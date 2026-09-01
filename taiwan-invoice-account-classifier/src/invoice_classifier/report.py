# -*- coding: utf-8 -*-
"""
invoice_classifier.report
============================
月度稅務合規報告生成器（整合自 M7-2 monthly_report.py）。

彙整當月處理的發票分類結果（invoices_processed）、人工修正紀錄
（corrections_made）與稅法合規規則觸發紀錄（compliance_violations），
產出一份供非技術人員（如公司負責人、會計主管）閱讀的繁體中文
月度報告（Markdown 格式）。

報告六大章節：
    1. 執行摘要（Executive Summary）
    2. 會計科目分佈（Account Distribution，含與上月比較）
    3. 稅務風險分析（Risk Analysis，含高風險發票明細）
    4. 合規規則觸發統計（Compliance Rule Trigger Stats）
    5. 修正紀錄分析（Correction Analysis）
    6. 建議行動（Recommended Actions）

⚠️ 重要聲明：
    本報告為「稅務輔助與風險提示工具」之輸出，僅供內部管理與風險
    自我檢核參考，不構成稅務或法律意見。報告中不使用「避免逃稅」
    等措辭，一律以「降低錯誤申報、重複列報、不得扣抵及憑證不完整
    風險」描述系統目的；高風險發票一律於報告中標示建議人工覆核。

非功能需求：
    效能：1000 筆發票 < 10 秒（純記憶體聚合運算，無外部 I/O）。
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["MonthlyReportGenerator", "ReportGenerationError"]

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_ACCOUNT_SUBJECTS_PATH = _THIS_DIR.parent.parent / "config" / "account_subjects.csv"
DEFAULT_TAX_RULES_PATH = _THIS_DIR.parent.parent / "config" / "tax_rules.json"

RISK_LEVEL_ORDER = ["critical", "high", "medium", "low"]
RISK_LEVEL_LABEL = {
    "critical": "極高風險", "high": "高風險", "medium": "中風險", "low": "低風險",
}


class ReportGenerationError(Exception):
    """報告生成過程中發生無法復原的錯誤時拋出。"""


def _load_account_subjects(path: Path) -> Dict[str, str]:
    import csv
    subjects: Dict[str, str] = {}
    if not path.exists():
        return subjects
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("account_code", "")).strip()
            if code:
                subjects[code] = row.get("account_name", "")
    return subjects


def _load_rule_notes(path: Path) -> Dict[str, Dict[str, str]]:
    """從 config/tax_rules.json 讀取規則名稱/法規依據/說明，供報告查詢使用。"""
    import json
    notes: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return notes
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules", {})
    if isinstance(rules, list):
        rules = {r["rule_id"]: r for r in rules}
    for rule_id, rule in rules.items():
        notes[rule_id] = {
            "name": rule_id,
            "law": rule.get("law_reference", ""),
            "note": rule.get("note", ""),
        }
    return notes


@dataclass
class _ReportStats:
    """報告生成過程中累積的中介統計資料（內部使用）。"""

    total_invoices: int = 0
    confidences: List[float] = field(default_factory=list)
    risk_counter: Counter = field(default_factory=Counter)
    account_counter: Counter = field(default_factory=Counter)
    high_risk_invoices: List[Dict[str, Any]] = field(default_factory=list)
    low_confidence_invoices: List[Dict[str, Any]] = field(default_factory=list)
    data_quality_notes: List[str] = field(default_factory=list)


class MonthlyReportGenerator:
    """
    月度稅務合規報告生成器。

    使用方式：
        generator = MonthlyReportGenerator()
        markdown = generator.generate_report(
            invoices_processed=[...],
            corrections_made=[...],
            compliance_violations=[...],
            company_name="範例股份有限公司A",
            report_month="2026-08",
        )
    """

    LOW_CONFIDENCE_THRESHOLD = 0.6
    MAX_HIGH_RISK_ROWS = 30
    TOP_N = 5

    def __init__(
        self,
        account_subjects: Optional[Dict[str, str]] = None,
        rule_notes: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.account_subjects = dict(
            account_subjects if account_subjects is not None
            else _load_account_subjects(DEFAULT_ACCOUNT_SUBJECTS_PATH)
        )
        self.rule_notes = dict(
            rule_notes if rule_notes is not None
            else _load_rule_notes(DEFAULT_TAX_RULES_PATH)
        )

    def generate_report(
        self,
        invoices_processed: List[Dict[str, Any]],
        corrections_made: List[Dict[str, Any]],
        compliance_violations: List[Dict[str, Any]],
        company_name: str = "貴公司",
        report_month: Optional[str] = None,
        previous_month_distribution: Optional[Dict[str, int]] = None,
        compliance_weight: Optional[float] = None,
    ) -> str:
        """
        產生完整月度報告（Markdown 字串）。

        compliance_weight: 目前公司設定的合規檢查權重（0-1，對應
            CompanyProfile.compliance_weight），用於「建議行動」章節
            提示是否需要調整 delta 權重。
        """
        start_time = time.perf_counter()

        invoices_processed = invoices_processed or []
        corrections_made = corrections_made or []
        compliance_violations = compliance_violations or []

        resolved_month = report_month or self._infer_report_month(invoices_processed)

        stats = self._aggregate_invoices(invoices_processed)
        rule_stats = self._aggregate_compliance(compliance_violations)
        correction_stats = self._aggregate_corrections(corrections_made)

        sections = [
            self._build_header(company_name, resolved_month),
            self._build_executive_summary(stats),
            self._build_account_distribution(stats, previous_month_distribution),
            self._build_risk_analysis(stats),
            self._build_compliance_section(rule_stats, len(compliance_violations)),
            self._build_correction_section(correction_stats),
            self._build_recommended_actions(stats, rule_stats, correction_stats, compliance_weight),
            self._build_footer(stats, start_time),
        ]

        report = "\n\n".join(section for section in sections if section)

        elapsed = time.perf_counter() - start_time
        if elapsed > 10.0:
            stats.data_quality_notes.append(
                f"⚠️ 報告生成耗時 {elapsed:.2f} 秒，超過 10 秒效能目標，建議檢查資料量或伺服器負載。"
            )
            report += (
                f"\n\n> ⚠️ **效能提醒**：本次報告生成耗時 {elapsed:.2f} 秒，"
                f"超過系統目標（<10 秒），建議確認發票筆數或系統資源。"
            )

        return report

    # ------------------------------------------------------------------
    # 聚合運算（內部方法）
    # ------------------------------------------------------------------

    def _infer_report_month(self, invoices: List[Dict[str, Any]]) -> str:
        for inv in invoices:
            date_str = inv.get("date") or inv.get("invoice_date")
            if date_str:
                try:
                    return str(date_str)[:7]
                except Exception:
                    continue
        return datetime.now().strftime("%Y-%m")

    def _account_name(self, code: Optional[str], fallback_name: Optional[str] = None) -> str:
        if fallback_name:
            return fallback_name
        if code and code in self.account_subjects:
            return self.account_subjects[code]
        return "未知科目" if not code else f"未登錄科目（{code}）"

    def _aggregate_invoices(self, invoices: List[Dict[str, Any]]) -> _ReportStats:
        stats = _ReportStats()
        stats.total_invoices = len(invoices)

        for idx, inv in enumerate(invoices):
            invoice_id = inv.get("invoice_id", f"UNKNOWN-{idx}")
            account_code = inv.get("predicted_account")
            risk_level = str(inv.get("risk_level", "low")).lower()

            if risk_level not in RISK_LEVEL_LABEL:
                stats.data_quality_notes.append(
                    f"發票 {invoice_id} 的風險等級「{inv.get('risk_level')}」無法辨識，"
                    f"已預設為 low 處理，建議覆核來源資料。"
                )
                risk_level = "low"

            confidence = inv.get("confidence")
            if isinstance(confidence, (int, float)):
                stats.confidences.append(float(confidence))
            else:
                confidence = None
                stats.data_quality_notes.append(
                    f"發票 {invoice_id} 缺少置信度（confidence）資料，未列入平均置信度計算。"
                )

            stats.risk_counter[risk_level] += 1

            if account_code:
                stats.account_counter[account_code] += 1
            else:
                stats.data_quality_notes.append(
                    f"發票 {invoice_id} 缺少預測科目（predicted_account），未列入科目分佈統計。"
                )

            if risk_level in ("high", "critical"):
                stats.high_risk_invoices.append({
                    "invoice_id": invoice_id,
                    "summary": inv.get("summary", ""),
                    "predicted_account": account_code or "-",
                    "account_name": self._account_name(account_code, inv.get("account_name")),
                    "risk_level": risk_level,
                    "risk_reason": inv.get("risk_reason") or "系統判定為高風險科目或關鍵字組合，建議人工複核法規適用性。",
                    "confidence": confidence,
                })

            if confidence is not None and confidence < self.LOW_CONFIDENCE_THRESHOLD:
                stats.low_confidence_invoices.append({
                    "invoice_id": invoice_id, "confidence": confidence,
                    "predicted_account": account_code or "-",
                })

        stats.high_risk_invoices.sort(
            key=lambda x: (
                RISK_LEVEL_ORDER.index(x["risk_level"]) if x["risk_level"] in RISK_LEVEL_ORDER else 99,
                x["confidence"] if x["confidence"] is not None else 0.0,
            )
        )
        return stats

    def _aggregate_compliance(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        counter: Counter = Counter()
        invoice_map: Dict[str, List[str]] = defaultdict(list)
        for v in violations:
            rule_id = v.get("rule_id", "unknown_rule")
            counter[rule_id] += 1
            if v.get("invoice_id"):
                invoice_map[rule_id].append(v["invoice_id"])
        return {"counter": counter, "invoice_map": invoice_map}

    def _aggregate_corrections(self, corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        corrected_to_counter: Counter = Counter()
        durations_seconds: List[float] = []
        notes: List[str] = []

        for c in corrections:
            corrected_to = c.get("corrected_to")
            if corrected_to:
                corrected_to_counter[corrected_to] += 1

            predicted_at = c.get("predicted_at")
            corrected_at = c.get("timestamp") or c.get("corrected_at")
            if predicted_at and corrected_at:
                duration = self._safe_duration_seconds(predicted_at, corrected_at)
                if duration is not None:
                    durations_seconds.append(duration)
                else:
                    notes.append(
                        f"修正紀錄 {c.get('invoice_id', '未知發票')} 的時間格式無法解析，未列入平均修正時間計算。"
                    )

        return {
            "total": len(corrections), "corrected_to_counter": corrected_to_counter,
            "durations_seconds": durations_seconds, "notes": notes,
        }

    @staticmethod
    def _safe_duration_seconds(start: str, end: str) -> Optional[float]:
        try:
            t_start = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            t_end = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
            return max((t_end - t_start).total_seconds(), 0.0)
        except Exception:
            return None

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f} 秒"
        if seconds < 3600:
            return f"{seconds / 60:.1f} 分鐘"
        if seconds < 86400:
            return f"{seconds / 3600:.1f} 小時"
        return f"{seconds / 86400:.1f} 天"

    # ------------------------------------------------------------------
    # 章節產生（內部方法）
    # ------------------------------------------------------------------

    def _build_header(self, company_name: str, report_month: str) -> str:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"# {company_name} 月度稅務合規報告（{report_month}）\n\n"
            f"報告生成時間：{generated_at}\n\n"
            f"> 本報告由「台灣電子發票會計科目分類系統」自動產生，"
            f"為稅務輔助與風險提示工具之輸出，**不構成稅務或法律意見**。\n"
        )

    def _build_executive_summary(self, stats: _ReportStats) -> str:
        total = stats.total_invoices
        avg_conf = statistics.mean(stats.confidences) if stats.confidences else 0.0
        high_risk_count = stats.risk_counter.get("high", 0) + stats.risk_counter.get("critical", 0)
        high_risk_pct = (high_risk_count / total * 100) if total else 0.0

        lines = [
            "## 一、執行摘要", "",
            f"- **當月處理發票總數**：{total} 張",
            f"- **平均置信度**：{avg_conf * 100:.1f}%",
            f"- **高風險發票數量**：{high_risk_count} 張（佔比 {high_risk_pct:.1f}%，含「高風險」與「極高風險」，"
            f"已標示建議轉人工覆核）",
        ]
        if total == 0:
            lines.append("")
            lines.append("> 本月無發票資料，以下各章節將顯示為空值。")
        return "\n".join(lines)

    def _build_account_distribution(
        self, stats: _ReportStats, previous_month_distribution: Optional[Dict[str, int]]
    ) -> str:
        total = sum(stats.account_counter.values())
        lines = ["## 二、會計科目分佈", ""]

        if total == 0:
            lines.append("本月無有效科目分佈資料。")
            return "\n".join(lines)

        header = "| 科目代碼 | 科目名稱 | 使用次數 | 佔比 |"
        divider = "| --- | --- | --- | --- |"
        if previous_month_distribution:
            header = "| 科目代碼 | 科目名稱 | 使用次數 | 佔比 | 與上月比較 |"
            divider = "| --- | --- | --- | --- | --- |"
        lines.append(header)
        lines.append(divider)

        for code, count in stats.account_counter.most_common():
            name = self._account_name(code)
            pct = count / total * 100
            row = f"| {code} | {name} | {count} | {pct:.1f}% |"
            if previous_month_distribution:
                prev = previous_month_distribution.get(code, 0)
                diff = count - prev
                if diff > 0:
                    change = f" 🔺 +{diff}"
                elif diff < 0:
                    change = f" 🔻 {diff}"
                else:
                    change = " 持平"
                row = f"| {code} | {name} | {count} | {pct:.1f}% |{change} |"
            lines.append(row)

        if previous_month_distribution:
            new_codes = set(stats.account_counter) - set(previous_month_distribution)
            vanished_codes = set(previous_month_distribution) - set(stats.account_counter)
            if new_codes:
                lines.append("")
                lines.append(
                    "本月新增使用科目：" + "、".join(f"{c}（{self._account_name(c)}）" for c in sorted(new_codes))
                )
            if vanished_codes:
                lines.append(
                    "本月未再使用（上月有使用）科目："
                    + "、".join(f"{c}（{self._account_name(c)}）" for c in sorted(vanished_codes))
                )
        return "\n".join(lines)

    def _build_risk_analysis(self, stats: _ReportStats) -> str:
        total = stats.total_invoices
        lines = ["## 三、稅務風險分析", "", "### 風險等級分佈", ""]

        if total == 0:
            lines.append("本月無發票資料。")
            return "\n".join(lines)

        lines.append("| 風險等級 | 發票數 | 佔比 |")
        lines.append("| --- | --- | --- |")
        for level in RISK_LEVEL_ORDER:
            count = stats.risk_counter.get(level, 0)
            pct = count / total * 100
            lines.append(f"| {RISK_LEVEL_LABEL[level]} | {count} | {pct:.1f}% |")

        lines.append("")
        lines.append("### 高風險發票明細（高風險／極高風險，建議轉人工覆核）")
        lines.append("")

        if not stats.high_risk_invoices:
            lines.append("本月無高風險或極高風險發票。")
        else:
            shown = stats.high_risk_invoices[: self.MAX_HIGH_RISK_ROWS]
            lines.append("| 發票編號 | 摘要 | 預測科目 | 風險等級 | 風險原因 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for item in shown:
                summary = self._truncate(item["summary"], 30)
                lines.append(
                    f"| {item['invoice_id']} | {summary} | "
                    f"{item['predicted_account']}（{item['account_name']}） | "
                    f"{RISK_LEVEL_LABEL.get(item['risk_level'], item['risk_level'])} | "
                    f"{item['risk_reason']} |"
                )
            remaining = len(stats.high_risk_invoices) - len(shown)
            if remaining > 0:
                lines.append("")
                lines.append(f"（其餘 {remaining} 張高風險發票因篇幅限制未逐筆列出，請於系統中查詢完整清單。）")
        return "\n".join(lines)

    def _build_compliance_section(self, rule_stats: Dict[str, Any], total_violations: int) -> str:
        counter: Counter = rule_stats["counter"]
        lines = ["## 四、合規規則觸發統計", ""]

        if total_violations == 0:
            lines.append("本月未觸發任何稅法合規規則。")
            return "\n".join(lines)

        lines.append(f"本月共觸發合規規則 {total_violations} 次，涉及 {len(counter)} 條規則。")
        lines.append("")
        lines.append("| 規則名稱 | 觸發次數 | 法規依據 |")
        lines.append("| --- | --- | --- |")
        for rule_id, count in counter.most_common():
            info = self.rule_notes.get(rule_id, {})
            name = info.get("name", rule_id)
            law = info.get("law", "（無登錄法規依據，請人工確認規則來源）")
            lines.append(f"| {name} | {count} | {law} |")

        lines.append("")
        lines.append(f"### 最常觸發規則 Top {self.TOP_N}")
        lines.append("")
        top_n = counter.most_common(self.TOP_N)
        for rank, (rule_id, count) in enumerate(top_n, start=1):
            info = self.rule_notes.get(rule_id, {})
            name = info.get("name", rule_id)
            note = info.get("note", "")
            law = info.get("law", "")
            lines.append(f"{rank}. **{name}**（{count} 次）— {law}。{note}")
        return "\n".join(lines)

    def _build_correction_section(self, correction_stats: Dict[str, Any]) -> str:
        total = correction_stats["total"]
        lines = ["## 五、修正紀錄分析", "", f"- **修正總數**：{total} 筆"]

        if total == 0:
            lines.append("")
            lines.append("本月無人工修正紀錄。")
            return "\n".join(lines)

        durations = correction_stats["durations_seconds"]
        if durations:
            avg_seconds = statistics.mean(durations)
            lines.append(f"- **平均修正時間**（自預測至人工修正）：{self._format_duration(avg_seconds)}")
        else:
            lines.append("- **平均修正時間**：無足夠時間戳資料可計算")

        lines.append("")
        lines.append(f"### 最常被修正的科目 Top {self.TOP_N}")
        lines.append("")
        top_n = correction_stats["corrected_to_counter"].most_common(self.TOP_N)
        if not top_n:
            lines.append("無有效的修正目標科目資料。")
        else:
            lines.append("| 修正後科目 | 科目名稱 | 次數 |")
            lines.append("| --- | --- | --- |")
            for code, count in top_n:
                lines.append(f"| {code} | {self._account_name(code)} | {count} |")
        return "\n".join(lines)

    def _build_recommended_actions(
        self,
        stats: _ReportStats,
        rule_stats: Dict[str, Any],
        correction_stats: Dict[str, Any],
        compliance_weight: Optional[float],
    ) -> str:
        lines = ["## 六、建議行動", "", "### 需優先覆核的發票"]

        if stats.high_risk_invoices:
            top_review = stats.high_risk_invoices[:10]
            for item in top_review:
                lines.append(
                    f"- 發票 {item['invoice_id']}（{item['predicted_account']} "
                    f"{item['account_name']}）：{item['risk_reason']}"
                )
        else:
            lines.append("- 本月無高風險發票，維持現行覆核頻率即可。")

        lines.append("")
        lines.append("### 需調整的記帳習慣")
        habit_notes: List[str] = []
        counter: Counter = rule_stats["counter"]
        if counter:
            top_rule_id, top_rule_count = counter.most_common(1)[0]
            info = self.rule_notes.get(top_rule_id, {})
            if top_rule_count >= 3:
                habit_notes.append(
                    f"「{info.get('name', top_rule_id)}」本月觸發 {top_rule_count} 次，"
                    f"建議與記帳人員確認相關摘要之科目判斷邏輯，降低重複觸發之錯誤申報風險"
                    f"（{info.get('law', '')}）。"
                )
        corrected_to_counter: Counter = correction_stats["corrected_to_counter"]
        if corrected_to_counter:
            top_corrected, top_count = corrected_to_counter.most_common(1)[0]
            if top_count >= 3:
                habit_notes.append(
                    f"科目「{top_corrected}（{self._account_name(top_corrected)}）」本月被人工修正 {top_count} 次，"
                    "建議檢視母版模型或客戶偏好權重是否需要調整。"
                )
        if not habit_notes:
            habit_notes.append("本月無明顯異常記帳習慣，建議持續觀察。")
        lines.extend(f"- {note}" for note in habit_notes)

        lines.append("")
        lines.append("### 需更新的公司設定")
        config_notes: List[str] = []
        total = stats.total_invoices
        high_risk_count = stats.risk_counter.get("high", 0) + stats.risk_counter.get("critical", 0)
        high_risk_ratio = (high_risk_count / total) if total else 0.0

        if compliance_weight is not None:
            if high_risk_ratio > 0.1 and compliance_weight < 0.4:
                config_notes.append(
                    f"目前 compliance_weight 為 {compliance_weight:.2f}，"
                    f"而本月高風險發票佔比達 {high_risk_ratio * 100:.1f}%，"
                    f"建議調高至 0.4 以上以強化合規檢查權重（delta），降低重複列報與不得扣抵風險。"
                )
            else:
                config_notes.append(
                    f"目前 compliance_weight 為 {compliance_weight:.2f}，依本月風險比例研判暫不需調整。"
                )
        else:
            config_notes.append("未提供目前 compliance_weight 設定值，建議於下次報告提供以利評估是否需調整。")

        if not config_notes:
            config_notes.append("本月無需更新公司設定。")
        lines.extend(f"- {note}" for note in config_notes)
        return "\n".join(lines)

    def _build_footer(self, stats: _ReportStats, start_time: float) -> str:
        lines = ["---", "", "### 資料品質提醒"]
        if stats.data_quality_notes:
            unique_notes = list(dict.fromkeys(stats.data_quality_notes))[:20]
            lines.extend(f"- {note}" for note in unique_notes)
            if len(stats.data_quality_notes) > len(unique_notes):
                lines.append(f"- （其餘 {len(stats.data_quality_notes) - len(unique_notes)} 筆提醒因重複已省略）")
        else:
            lines.append("- 本月資料完整，無缺漏欄位提醒。")

        lines.append("")
        lines.append(
            "> 本報告由「台灣電子發票會計科目分類系統」自動產生，內容僅供內部管理與風險自我檢核參考，"
            "為降低錯誤申報、重複列報、不得扣抵及憑證不完整風險而設計，**不構成稅務或法律意見**，"
            "實際稅務處理仍請以會計師或稅務顧問之專業判斷為準。"
        )
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = text or ""
        return text if len(text) <= max_len else text[: max_len - 1] + "…"
