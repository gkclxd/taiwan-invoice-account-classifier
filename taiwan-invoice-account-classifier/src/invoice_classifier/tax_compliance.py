# -*- coding: utf-8 -*-
"""
invoice_classifier.tax_compliance
====================================
台灣稅法合規檢查模組（整合自 M4-1 tax_rules.py + M4-2 risk_scorer.py）。

⚠️ 系統定位重要聲明：
    本模組為「稅務輔助與風險提示工具」，所有規則比對、風險評分與提醒文字，
    僅供使用者內部記帳與風險自我檢核參考，**不構成正式稅務或法律意見**。
    - 系統不會、也不應被用於協助規避稅捐（本模組不使用「避免逃稅」一類
      措辭，一律改以「降低錯誤申報、重複列報、不得扣抵及憑證不完整風險」
      描述系統目的）。
    - 系統不會依單一關鍵字或規則命中，逕自宣稱使用者違法或有逃漏稅意圖；
      任何規則觸發僅代表「存在需要留意的稅務風險情境」，實際是否違規
      仍須由企業會計/稅務顧問依實際交易事實判斷。
    - 高風險（risk_level = high / critical）結果一律標示「建議轉人工覆核」。

功能：
    1. TaxRuleEngine：載入 config/tax_rules.json 規則庫，依發票摘要 /
       貿易條件比對觸發之規則（進項稅額分離、CIF 運費不得重複列報、
       報關費併入進貨成本、交際費列支上限、員工福利 vs 交際費、
       自用乘人小汽車進項稅額不得扣抵等情境）。
    2. assess_invoice_risk()：五大風險維度評分（科目風險、關鍵字風險、
       貿易條件風險、金額異常風險、憑證風險），輸出風險分數（0-1）、
       風險等級（low/medium/high/critical）、風險旗標與合規提醒文字。

自訂科目對照表：
    本模組使用的科目代碼（如 5121、1268 等）為系統預設值，若企業提供
    CompanyProfile.custom_account_mapping，應由呼叫端（classifier.py）
    在取得本模組輸出後，套用對照表轉換為企業自有科目代碼／名稱，
    本模組本身不假設任何科目代碼為所有公司之唯一標準。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

_THIS_DIR = Path(__file__).resolve().parent
# repo 結構：src/invoice_classifier/tax_compliance.py -> ../../config/tax_rules.json
DEFAULT_RULES_PATH = _THIS_DIR.parent.parent / "config" / "tax_rules.json"


# ============================================================================
# 1. 稅法合規規則庫（TaxRuleEngine）
# ============================================================================

class TaxRuleDict(TypedDict, total=False):
    rule_id: str
    trigger_keywords: List[str]
    trigger_condition: Optional[str]
    preferred_account: Optional[str]
    forbidden_accounts: List[str]
    weight: float
    penalty: float
    note: str
    law_reference: str
    effective_date: str
    scope: str
    manual_review_required: str
    resolution_note: str


class TaxRuleEngine:
    """
    稅法合規規則引擎：載入 config/tax_rules.json，提供規則查詢與比對。
    """

    def __init__(self, rules_path: Optional[Path] = None):
        self.rules_path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
        self._rules: Dict[str, TaxRuleDict] = {}
        self.version: str = ""
        self.last_updated: str = ""
        self.description: str = ""
        self._load()

    def _load(self) -> None:
        if not self.rules_path.exists():
            raise FileNotFoundError(
                f"找不到稅法合規規則檔案：{self.rules_path}。"
                f"請確認 config/tax_rules.json 存在於專案根目錄。"
            )
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self.version = payload.get("version", "")
        self.last_updated = payload.get("last_updated", "")
        self.description = payload.get("description", "")
        rules = payload.get("rules", {})
        if isinstance(rules, list):
            # 相容 list 結構（如 M4-1 原始格式），轉為 dict 結構
            rules = {r["rule_id"]: r for r in rules}
        self._rules = rules

    def get_rule(self, rule_id: str) -> TaxRuleDict:
        return self._rules[rule_id]

    def list_rules(self) -> List[TaxRuleDict]:
        return list(self._rules.values())

    def all_rules_dict(self) -> Dict[str, TaxRuleDict]:
        """回傳 {rule_id: rule_dict}，供 fusion_engine 直接使用。"""
        return self._rules

    def match_rules(
        self,
        summary: str,
        trade_condition: Optional[str] = None,
    ) -> List[TaxRuleDict]:
        """
        依發票摘要（及可選的貿易條件）比對觸發之規則清單。

        比對邏輯：
            1. trigger_keywords：任一關鍵字出現於 summary 即視為關鍵字命中。
            2. trigger_condition：
                 - None / "AMOUNT_CHECK" → 不限制是否觸發（AMOUNT_CHECK 僅
                   額外標記需人工檢查全年累計金額，實際比對邏輯由呼叫端
                   依全年累計數判斷）。
                 - "CIF" → 需 trade_condition == "CIF" 且關鍵字命中。

        Returns:
            依規則定義順序排列，命中的規則字典清單（可能為空清單）。
        """
        matched: List[TaxRuleDict] = []
        summary = summary or ""
        for rule in self._rules.values():
            keywords = rule.get("trigger_keywords", [])
            keyword_hit = any(kw in summary for kw in keywords) if keywords else False
            if keywords and not keyword_hit:
                continue

            condition = rule.get("trigger_condition")
            if condition == "CIF" and trade_condition != "CIF":
                continue
            # AMOUNT_CHECK 與 None 皆不影響是否觸發，僅影響後續處理提示

            matched.append(rule)
        return matched

    # -- 特定情境檢查（任務要求的專屬檢查函式） --------------------------------

    def check_input_tax_deduction_issues(
        self, summary: str, predicted_account: Optional[str] = None
    ) -> List[str]:
        """
        「不得扣抵」情境檢查：檢查是否觸發自用乘人小汽車進項稅額不得扣抵
        （rule_006），或進項稅額未單獨列帳（rule_001）等不得扣抵情境。

        回傳提醒文字清單（可能為空）。
        """
        notes: List[str] = []
        matched = self.match_rules(summary)
        for rule in matched:
            forbidden = rule.get("forbidden_accounts", [])
            if "1268" in forbidden:
                notes.append(
                    f"[{rule.get('rule_id')}] {rule.get('note', '')}"
                    f"（本情境涉及進項稅額不得扣抵，建議轉人工覆核）"
                )
            if predicted_account and predicted_account in forbidden:
                notes.append(
                    f"[{rule.get('rule_id')}] 目前預測科目 {predicted_account} "
                    f"屬本規則之禁止科目，建議轉人工覆核後再行入帳。"
                )
        return notes

    def check_duplicate_cost_reporting(
        self, summary: str, trade_condition: Optional[str] = None
    ) -> List[str]:
        """
        進貨成本重複列報檢查：主要針對 CIF 條件下運費/保險費是否與貨價
        重複列報（rule_002），以及報關費是否已併入進貨成本（rule_003）。
        """
        notes: List[str] = []
        matched = self.match_rules(summary, trade_condition=trade_condition)
        for rule in matched:
            rule_id = rule.get("rule_id", "")
            if "cif_freight" in rule_id.lower() or "customs" in rule_id.lower():
                notes.append(f"[{rule_id}] {rule.get('note', '')}")
        return notes

    def suggest_input_tax_handling(self, summary: str) -> Optional[str]:
        """
        進項稅額處理提醒：若摘要顯示涉及營業稅/進項稅額字樣，
        提醒應單獨列帳（rule_001），不可併入進貨成本或費用科目。
        """
        matched = self.match_rules(summary)
        for rule in matched:
            if "input_tax" in rule.get("rule_id", "").lower():
                return f"[{rule.get('rule_id')}] {rule.get('note', '')}"
        return None


_default_engine: Optional[TaxRuleEngine] = None


def get_default_engine() -> TaxRuleEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = TaxRuleEngine()
    return _default_engine


# ============================================================================
# 2. 五大風險維度評分（assess_invoice_risk）
# ============================================================================

ACCOUNT_RISK_BASE_SCORE: Dict[str, float] = {
    "critical": 0.75,
    "high": 0.55,
    "medium": 0.35,
    "low": 0.05,
}

ACCOUNT_RISK_MAP: Dict[str, str] = {
    "1268": "critical",  # 進項稅額：若未分離列帳視為 critical
    "5213": "medium",    # 交際費：有列支上限
    "5215": "medium",    # 差旅費：需證明業務相關
    "5221": "high",       # 運費：易與 CIF 貨價重複列報
    "5222": "high",       # 保險費：易與 CIF 貨價重複列報
    "5121": "low",        # 進貨
    "5216": "low",        # 辦公費
}

ACCOUNT_RISK_NOTE: Dict[str, str] = {
    "1268": "進項稅額（1268）為系統預設之高關注科目：進項稅額應單獨列帳，"
            "若與進貨/費用科目併記，可能構成重複列報或不得扣抵之風險。",
    "5213": "交際費（5213）依規定有列支上限，超過部分將被剔除，請留意金額是否合理。",
    "5215": "差旅費（5215）建議檢附出差事由、業務相關證明，以利因應查核。",
    "5221": "運費（5221）於 CIF 貿易條件下可能已內含於貨價中，"
            "重複列報將構成重複認列費用之風險。",
    "5222": "保險費（5222）於 CIF 貿易條件下可能已內含於貨價中，"
            "重複列報將構成重複認列費用之風險。",
}

KEYWORD_RISK_SCORE = 0.15
HIGH_RISK_KEYWORDS: List[str] = ["交際", "應酬", "員工", "自用", "收據", "估單"]

KEYWORD_RISK_NOTE: Dict[str, str] = {
    "交際": "摘要含「交際」：請確認是否屬交際費（5213，有列支上限）"
            "或應改列職工福利（5226）。",
    "應酬": "摘要含「應酬」：屬交際性質支出，有列支上限，超過部分可能被剔除。",
    "員工": "摘要含「員工」：若為員工旅遊/聚餐/尾牙等集體性質支出，"
            "應優先考慮列為職工福利（5226）而非交際費（5213）。",
    "自用": "摘要含「自用」：若涉及自用乘人小汽車，其進項稅額可能不得扣抵，"
            "請確認是否誤列 1268，建議轉人工覆核。",
    "收據": "摘要含「收據」：收據非統一發票，憑證不完整可能影響費用認列，"
            "須人工複核憑證合法性。",
    "估單": "摘要含「估單」：估價單非正式交易憑證，請確認是否已取得正式發票或收據。",
}

CIF_FREIGHT_KEYWORDS: List[str] = ["運費", "保險費"]
CIF_FREIGHT_RISK_SCORE = 0.4
FOB_NO_FREIGHT_DOC_RISK_SCORE = 0.2

AMOUNT_ZERO_OR_NEGATIVE_RISK_SCORE = 0.5
AMOUNT_INDUSTRY_OUTLIER_RISK_SCORE = 0.3
INDUSTRY_STANDARD_AMOUNT: Dict[str, float] = {
    "5121": 50000.0, "5122": 8000.0, "5213": 5000.0, "5214": 1000.0,
    "5215": 15000.0, "5216": 3000.0, "5217": 2000.0, "5218": 4000.0,
    "5219": 20000.0, "5221": 10000.0, "5222": 5000.0, "5226": 10000.0,
    "5253": 30000.0,
}
INDUSTRY_OUTLIER_MULTIPLIER = 2.0

DOCUMENT_RISK_KEYWORDS: List[str] = ["收據", "估單", "證明單"]
DOCUMENT_RISK_SCORE = 0.2
DOCUMENT_RISK_NOTE = (
    "摘要顯示可能非正式統一發票憑證（收據/估價單/證明單），"
    "憑證不完整可能影響費用認列，請優先取得買受人為公司抬頭之三聯式或電子發票。"
)

MAX_RISK_SCORE = 1.0
RISK_LEVEL_THRESHOLDS = (
    (0.7, "critical"),
    (0.5, "high"),
    (0.3, "medium"),
    (0.0, "low"),
)


class RiskDict(TypedDict):
    risk_score: float
    risk_level: str
    risk_flags: List[str]
    compliance_notes: List[str]
    manual_review_recommended: bool


@dataclass
class _InvoiceLike:
    """內部標準化容器：將任意輸入（InvoiceData / dict / 具屬性物件）正規化。"""
    summary: str = ""
    seller_ban: str = ""
    buyer_ban: str = ""
    amount: Optional[float] = None
    trade_condition: Optional[str] = None
    account_code: Optional[str] = None
    invoice_date: Optional[str] = None
    flags: List[str] = field(default_factory=list)


def _coerce_invoice(invoice: Any) -> _InvoiceLike:
    def _get(name: str, default=None):
        if isinstance(invoice, dict):
            return invoice.get(name, default)
        return getattr(invoice, name, default)

    trade_condition = _get("trade_condition")
    if trade_condition is not None and hasattr(trade_condition, "value"):
        trade_condition = trade_condition.value

    return _InvoiceLike(
        summary=_get("summary", "") or "",
        seller_ban=_get("seller_ban", "") or "",
        buyer_ban=_get("buyer_ban", "") or "",
        amount=_get("amount"),
        trade_condition=trade_condition,
        account_code=_get("account_code"),
        invoice_date=_get("invoice_date"),
    )


def _score_account_risk(inv: _InvoiceLike) -> float:
    if not inv.account_code:
        return 0.0
    level = ACCOUNT_RISK_MAP.get(inv.account_code)
    if level is None:
        return 0.0
    score = ACCOUNT_RISK_BASE_SCORE[level]
    inv.flags.append(f"科目風險：{inv.account_code} 屬 {level} 風險科目（+{score:.2f}）")
    note = ACCOUNT_RISK_NOTE.get(inv.account_code)
    if note:
        inv.flags.append(f"__note__:{note}")
    return score


def _score_keyword_risk(inv: _InvoiceLike) -> float:
    total = 0.0
    summary = inv.summary or ""
    for kw in HIGH_RISK_KEYWORDS:
        if kw in summary:
            total += KEYWORD_RISK_SCORE
            inv.flags.append(f"關鍵字風險：摘要包含「{kw}」（+{KEYWORD_RISK_SCORE:.2f}）")
            note = KEYWORD_RISK_NOTE.get(kw)
            if note:
                inv.flags.append(f"__note__:{note}")
    return total


def _score_trade_condition_risk(inv: _InvoiceLike) -> float:
    total = 0.0
    summary = inv.summary or ""
    tc = (inv.trade_condition or "").upper()

    if tc == "CIF":
        hit_keywords = [kw for kw in CIF_FREIGHT_KEYWORDS if kw in summary]
        if hit_keywords:
            total += CIF_FREIGHT_RISK_SCORE
            inv.flags.append(
                f"貿易條件風險：CIF 條件下摘要含「{'、'.join(hit_keywords)}」，"
                f"疑似與貨價重複列報（+{CIF_FREIGHT_RISK_SCORE:.2f}）"
            )
            inv.flags.append(
                "__note__:CIF 條件下運費及保險費可能已內含於貨價中，"
                "不建議再單獨列報，避免重複列報風險。"
            )
    elif tc == "FOB":
        hit_keywords = [kw for kw in CIF_FREIGHT_KEYWORDS if kw in summary]
        if not hit_keywords:
            total += FOB_NO_FREIGHT_DOC_RISK_SCORE
            inv.flags.append(
                f"貿易條件風險：FOB 條件下未見運費/保險費相關單據，"
                f"疑似進貨成本認列不完整（+{FOB_NO_FREIGHT_DOC_RISK_SCORE:.2f}）"
            )
            inv.flags.append(
                "__note__:FOB 條件下運費、保險費由買方負擔，"
                "應取得對應單據併入或單獨列報，避免成本認列不完整。"
            )
    return total


def _score_amount_anomaly_risk(inv: _InvoiceLike) -> float:
    total = 0.0
    amount = inv.amount

    if amount is None:
        inv.flags.append("__note__:發票金額缺失，無法進行金額異常風險評估，建議人工複核金額欄位。")
        return 0.0

    if amount <= 0:
        total += AMOUNT_ZERO_OR_NEGATIVE_RISK_SCORE
        inv.flags.append(
            f"金額異常風險：發票金額為 0 或負數（{amount}）（+{AMOUNT_ZERO_OR_NEGATIVE_RISK_SCORE:.2f}）"
        )
        inv.flags.append("__note__:金額為 0 或負數之發票應暫緩入帳，並確認是否為折讓單、作廢發票或資料輸入錯誤。")
        return total

    standard = INDUSTRY_STANDARD_AMOUNT.get(inv.account_code) if inv.account_code else None
    if standard is not None and amount > standard * INDUSTRY_OUTLIER_MULTIPLIER:
        total += AMOUNT_INDUSTRY_OUTLIER_RISK_SCORE
        inv.flags.append(
            f"金額異常風險：金額 {amount:,.0f} 超過同業標準 "
            f"({standard:,.0f}) 的 {INDUSTRY_OUTLIER_MULTIPLIER:.0f} 倍（+{AMOUNT_INDUSTRY_OUTLIER_RISK_SCORE:.2f}）"
        )
        inv.flags.append("__note__:金額顯著偏離同業水準，建議確認是否有分割發票、金額誤植或科目誤植之情形。")

    return total


def _score_document_risk(inv: _InvoiceLike) -> float:
    summary = inv.summary or ""
    hit_keywords = [kw for kw in DOCUMENT_RISK_KEYWORDS if kw in summary]
    if not hit_keywords:
        return 0.0
    inv.flags.append(
        f"憑證風險：摘要包含「{'、'.join(hit_keywords)}」，疑似非正式憑證（+{DOCUMENT_RISK_SCORE:.2f}）"
    )
    inv.flags.append(f"__note__:{DOCUMENT_RISK_NOTE}")
    return DOCUMENT_RISK_SCORE


def _risk_level_from_score(score: float) -> str:
    for threshold, level in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"  # pragma: no cover


def assess_invoice_risk(invoice: Any, account_code: Optional[str] = None) -> RiskDict:
    """
    評估單張發票的稅務風險分數與等級（五大風險維度）。

    Args:
        invoice: InvoiceData 實例、dict，或任何具備
            summary / seller_ban / buyer_ban / amount / trade_condition 屬性的物件。
        account_code: 選填，此發票「已知或預測」之會計科目代碼（4 碼）。

    Returns:
        RiskDict，其中 manual_review_recommended 在 risk_level 為
        high / critical 時為 True（依任務要求：高風險結果必須標示
        「建議轉人工覆核」）。

    效能：純字串比對與四則運算，單筆評估遠低於 0.1 秒。
    """
    inv = _coerce_invoice(invoice)
    if account_code:
        inv.account_code = account_code

    dimension_scores = [
        _score_account_risk(inv),
        _score_keyword_risk(inv),
        _score_trade_condition_risk(inv),
        _score_amount_anomaly_risk(inv),
        _score_document_risk(inv),
    ]

    raw_score = sum(dimension_scores)
    risk_score = min(raw_score, MAX_RISK_SCORE)
    risk_level = _risk_level_from_score(risk_score)

    risk_flags = [f for f in inv.flags if not f.startswith("__note__:")]
    compliance_notes = [f[len("__note__:"):] for f in inv.flags if f.startswith("__note__:")]

    if not risk_flags:
        risk_flags.append("未觸發任何風險維度，屬低風險發票。")
    if not compliance_notes:
        compliance_notes.append("本筆發票未觸發特定合規提醒，仍建議依內部控制程序留存憑證正本。")

    if raw_score > MAX_RISK_SCORE:
        risk_flags.append(f"（原始加總分數 {raw_score:.2f} 已超過上限 1.0，已裁切）")

    manual_review_recommended = risk_level in ("high", "critical")
    if manual_review_recommended:
        compliance_notes.append("⚠️ 本筆發票風險等級較高，建議轉人工覆核，本結果不構成稅務或法律意見。")

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "risk_flags": risk_flags,
        "compliance_notes": compliance_notes,
        "manual_review_recommended": manual_review_recommended,
    }


__all__ = [
    "TaxRuleEngine",
    "get_default_engine",
    "assess_invoice_risk",
    "RiskDict",
    "TaxRuleDict",
]
