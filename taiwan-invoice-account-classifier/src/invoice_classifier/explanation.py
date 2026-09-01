# -*- coding: utf-8 -*-
"""
invoice_classifier.explanation
=================================
解釋模組整合層（本專案新設計，無對應舊模組原型）。

功能：
    讀取 fusion_engine.dynamic_weighted_prediction() 輸出的
    score_breakdown / explanation 資料，組裝成一份人類可讀（繁體中文）
    的預測解釋，供前端或報告呈現，說明：
        - 為什麼系統預測這個科目（分數拆解，四來源各佔多少）
        - 觸發了哪些稅法合規規則、對應法規依據
        - 建議是否轉人工覆核，以及理由

⚠️ 重要聲明：
    本模組產生之解釋文字僅說明系統計算過程與風險提示依據，
    **不構成稅務或法律意見**。任何標示「建議轉人工覆核」的情境，
    使用者不應僅依系統解釋逕自認定入帳科目或稅務處理方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------
# 資料結構：PredictionExplanation
# ----------------------------------------------------------------------------

@dataclass
class PredictionExplanation:
    """
    單筆發票預測結果的完整解釋（供 classifier.py / report.py 使用）。
    """

    predicted_account: str
    account_name: str
    confidence: float
    risk_level: str
    manual_review_recommended: bool
    suggested_action: str

    # 分數拆解：{account_code: {master_score, correction_score, seller_score,
    #                            compliance_score, final_score}}
    score_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 主導本次預測結果的來源（依 alpha*master / beta*correction / gamma*seller /
    # delta*compliance 中，對 predicted_account 貢獻最大者判斷）
    dominant_source: str = ""
    dominant_source_label: str = ""

    # 觸發的稅法合規規則（rule_id 清單）與對應說明文字
    triggered_rules: List[str] = field(default_factory=list)
    compliance_notes: List[str] = field(default_factory=list)

    # 其他候選科目（top-k，不含最終預測本身）
    alternative_candidates: List[Dict[str, Any]] = field(default_factory=list)

    # 人類可讀的完整說明文字（繁體中文，多行）
    human_readable_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_account": self.predicted_account,
            "account_name": self.account_name,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "manual_review_recommended": self.manual_review_recommended,
            "suggested_action": self.suggested_action,
            "score_breakdown": self.score_breakdown,
            "dominant_source": self.dominant_source,
            "dominant_source_label": self.dominant_source_label,
            "triggered_rules": self.triggered_rules,
            "compliance_notes": self.compliance_notes,
            "alternative_candidates": self.alternative_candidates,
            "human_readable_summary": self.human_readable_summary,
        }


_SOURCE_LABELS = {
    "master_score": "母版模型（歷史案例相似度）",
    "correction_score": "客戶個人化修正紀錄",
    "seller_score": "賣方歷史科目偏好",
    "compliance_score": "稅法合規規則",
}

_WEIGHT_KEY_TO_SCORE_KEY = {
    "alpha": "master_score",
    "beta": "correction_score",
    "gamma": "seller_score",
    "delta": "compliance_score",
}


def _determine_dominant_source(
    predicted_account: str,
    score_breakdown: Dict[str, Dict[str, float]],
    weights_used: Dict[str, float],
) -> str:
    """依「權重 x 分項分數」找出對最終分數貢獻最大的來源（four-way tie 時取第一個非零者）。"""
    row = score_breakdown.get(predicted_account, {})
    contributions = {}
    for weight_key, score_key in _WEIGHT_KEY_TO_SCORE_KEY.items():
        w = float(weights_used.get(weight_key, 0.0))
        s = float(row.get(score_key, 0.0))
        contributions[score_key] = w * s
    if not contributions or max(contributions.values()) <= 0:
        return "master_score"
    return max(contributions, key=contributions.get)


def build_explanation(fusion_result: Dict[str, Any]) -> PredictionExplanation:
    """
    將 fusion_engine.dynamic_weighted_prediction() 的輸出組裝為
    PredictionExplanation。

    Parameters
    ----------
    fusion_result : dict
        fusion_engine.dynamic_weighted_prediction() 的回傳值，需含
        "predicted_account", "confidence", "score_breakdown", "explanation"。

    Returns
    -------
    PredictionExplanation
    """
    predicted_account = fusion_result["predicted_account"]
    confidence = fusion_result["confidence"]
    score_breakdown = fusion_result.get("score_breakdown", {})
    exp = fusion_result.get("explanation", {})

    risk_level = exp.get("risk_level", "low")
    account_name = exp.get("account_name", "")
    suggested_action = exp.get("suggested_action", "")
    triggered_rules = exp.get("triggered_rules", [])
    compliance_notes = exp.get("compliance_notes", [])
    weights_used = exp.get("weights_used", {})
    top_k_candidates = exp.get("top_k_candidates", [])

    manual_review_recommended = risk_level in ("high", "critical")

    dominant_source = _determine_dominant_source(predicted_account, score_breakdown, weights_used)
    dominant_source_label = _SOURCE_LABELS.get(dominant_source, dominant_source)

    alternatives = [c for c in top_k_candidates if c.get("account_code") != predicted_account]

    summary_lines = _build_human_readable_lines(
        predicted_account=predicted_account,
        account_name=account_name,
        confidence=confidence,
        risk_level=risk_level,
        dominant_source_label=dominant_source_label,
        triggered_rules=triggered_rules,
        compliance_notes=compliance_notes,
        suggested_action=suggested_action,
        alternatives=alternatives,
        manual_review_recommended=manual_review_recommended,
    )

    return PredictionExplanation(
        predicted_account=predicted_account,
        account_name=account_name,
        confidence=confidence,
        risk_level=risk_level,
        manual_review_recommended=manual_review_recommended,
        suggested_action=suggested_action,
        score_breakdown=score_breakdown,
        dominant_source=dominant_source,
        dominant_source_label=dominant_source_label,
        triggered_rules=triggered_rules,
        compliance_notes=compliance_notes,
        alternative_candidates=alternatives,
        human_readable_summary="\n".join(summary_lines),
    )


def _build_human_readable_lines(
    predicted_account: str,
    account_name: str,
    confidence: float,
    risk_level: str,
    dominant_source_label: str,
    triggered_rules: List[str],
    compliance_notes: List[str],
    suggested_action: str,
    alternatives: List[Dict[str, Any]],
    manual_review_recommended: bool,
) -> List[str]:
    risk_label = {"low": "低風險", "medium": "中風險", "high": "高風險", "critical": "極高風險"}.get(
        risk_level, risk_level
    )

    lines: List[str] = []
    lines.append(f"系統建議科目：{predicted_account}（{account_name}），置信度 {confidence:.2%}")
    lines.append(f"主要判斷依據：{dominant_source_label}")
    lines.append(f"稅務風險等級：{risk_label}")

    if triggered_rules:
        lines.append(f"觸發稅法合規規則：{', '.join(triggered_rules)}")
        for note in compliance_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("未觸發任何稅法合規規則。")

    if alternatives:
        alt_desc = "、".join(
            f"{c.get('account_code')}（{c.get('account_name', '')}，分數 {c.get('final_score')}）"
            for c in alternatives
        )
        lines.append(f"其他候選科目：{alt_desc}")

    lines.append(f"建議行動：{suggested_action}")

    if manual_review_recommended:
        lines.append("⚠️ 建議轉人工覆核：本結果僅供風險提示參考，不構成稅務或法律意見。")

    return lines


__all__ = ["PredictionExplanation", "build_explanation"]
