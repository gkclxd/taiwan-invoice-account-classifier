# -*- coding: utf-8 -*-
"""
invoice_classifier.fusion_engine
===================================
動態加權預測引擎（整合自 M5-1 fusion_engine.py）。

融合四個預測來源，輸出最終會計科目預測建議：
    1. 母版模型（Master Model）        — 摘要向量 vs. 科目原型向量的 cosine similarity
    2. 客戶修正原型（Correction Prototypes） — 摘要向量 vs. 該公司歷史修正紀錄學到的原型向量
    3. 賣方偏好（Seller Preferences）  — 該賣方統編過去被記過的科目分佈
    4. 稅法合規規則（Tax Compliance Rules） — 規則觸發後對科目分數的加分/懲罰

    final_score[i] = alpha * master_score[i]
                   + beta  * correction_score[i]
                   + gamma * seller_score[i]
                   + delta * compliance_score[i]

    predicted_account = argmax(final_score)
    confidence        = final_score[predicted_account]（正規化至 0-1）

⚠️ 重要聲明：
    本函式輸出之 predicted_account、confidence、risk_level 僅為系統依
    歷史資料與規則計算之建議值，**不構成稅務或法律意見**。risk_level 為
    high/critical，或 confidence 偏低時，皆會於 suggested_action 中
    標示「建議轉人工覆核」，實際入帳科目應由企業會計/稅務顧問確認。

⚠️ 向量空間一致性提醒：
    correction_prototypes（客戶修正原型向量）須與 master_model 使用
    「相同的向量化器（embedder/vectorizer）與維度」產生，否則
    cosine_similarity 會因維度不一致而回傳 0（該來源分數視同無證據，
    不會中斷流程，但也不會提供個人化資訊）。實務上建議：
        1. storage.SummaryVectorizer 預設嘗試載入
           preprocessor.InvoiceVectorizer(backend="tfidf_svd")；
        2. 但該向量化器若未與 master_model 訓練時使用的向量化器共用
           同一份已 fit 的模型（vocabulary / SVD 投影矩陣），兩者輸出
           的向量空間並不相通，即使維度剛好相同也可能語意不一致。
        3. 正式部署時，應讓 CorrectionManager 的 SummaryVectorizer 與
           MasterModel 共用同一份已訓練向量化器（例如都指向
           models/master_model_vectorizer.pkl），以確保個人化學習層
           的向量與母版模型的向量落在同一空間，可直接比較 cosine similarity。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 工具函式
# ----------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        # 向量維度不一致（例如個人化修正紀錄的向量空間與目前母版模型不同版本），
        # 無法計算相似度時保守回傳 0（不貢獻分數），並由呼叫端決定是否記錄警告，
        # 避免整體預測流程中斷。
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------------------------------------------------------
# 各來源分數計算
# ----------------------------------------------------------------------------

def compute_master_scores(summary_vector: np.ndarray, master_model) -> Dict[str, float]:
    """母版模型分數：摘要向量與每個科目原型向量的 cosine similarity（映射到 0-1）。"""
    scores = {}
    for code in master_model.account_codes:
        proto = master_model.account_vectors.get(code)
        if proto is None:
            scores[code] = 0.0
            continue
        sim = cosine_similarity(summary_vector, proto)
        scores[code] = (sim + 1.0) / 2.0
    return scores


def compute_correction_scores(
    summary_vector: np.ndarray,
    correction_prototypes: Dict[str, np.ndarray],
    account_codes: List[str],
) -> Dict[str, float]:
    """
    客戶修正原型分數：對每個「該客戶曾修正過」的科目，計算摘要向量與該科目
    修正原型向量的 cosine similarity。沒有修正紀錄的科目分數為 0。
    """
    scores = {code: 0.0 for code in account_codes}
    for code, proto_vec in correction_prototypes.items():
        if code not in scores:
            scores[code] = 0.0
        sim = cosine_similarity(summary_vector, proto_vec)
        scores[code] = max(0.0, (sim + 1.0) / 2.0)
    return scores


def compute_seller_scores(
    seller_ban: str,
    seller_preferences: Dict[str, Dict[str, float]],
    account_codes: List[str],
) -> Dict[str, float]:
    """
    賣方偏好分數：若賣方統編有偏好紀錄，依偏好比例對該科目加分；
    無偏好紀錄的賣方，所有科目分數為 0。

    seller_preferences 結構範例：
        {"12345678": {"5121": 0.8, "5122": 0.2}}  # 賣方 -> {科目: 偏好比例}
    """
    scores = {code: 0.0 for code in account_codes}
    seller_pref = seller_preferences.get(seller_ban, {})
    for code, ratio in seller_pref.items():
        if code not in scores:
            scores[code] = 0.0
        scores[code] = float(ratio)
    return scores


def compute_compliance_scores(
    invoice_data: Any,
    compliance_rules: Dict[str, Dict[str, Any]],
    account_codes: List[str],
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    稅法合規分數：掃描所有規則，若觸發：
      - 對 preferred_account 加分（+weight），若 preferred_account 為 None
        則不加分（本規則性質為否決/禁止性，見規則庫 resolution_note）。
      - 對 forbidden_accounts 扣分（-penalty，可使分數降為極低甚至歸零）。

    回傳 (compliance_scores, compliance_notes, triggered_rule_ids)
    """
    scores = {code: 0.0 for code in account_codes}
    notes: List[str] = []
    triggered_rules: List[str] = []

    summary = getattr(invoice_data, "summary", None)
    if summary is None and isinstance(invoice_data, dict):
        summary = invoice_data.get("summary", "")
    summary = summary or ""

    trade_condition = getattr(invoice_data, "trade_condition", None)
    if trade_condition is None and isinstance(invoice_data, dict):
        trade_condition = invoice_data.get("trade_condition")
    if trade_condition is not None and hasattr(trade_condition, "value"):
        trade_condition = trade_condition.value

    for rule_id, rule in compliance_rules.items():
        keywords = rule.get("trigger_keywords", [])
        condition = rule.get("trigger_condition")
        weight = float(rule.get("weight", 0.0))
        penalty = float(rule.get("penalty", 0.0))
        preferred = rule.get("preferred_account")
        forbidden = rule.get("forbidden_accounts", [])
        note = rule.get("note", "")

        keyword_hit = any(kw in summary for kw in keywords) if keywords else False
        condition_hit = True
        if condition == "CIF":
            condition_hit = (trade_condition == "CIF")

        triggered = keyword_hit and condition_hit if keywords else condition_hit

        if not triggered:
            continue

        triggered_rules.append(rule_id)
        if note:
            notes.append(f"[{rule_id}] {note}")

        if preferred and preferred in scores:
            scores[preferred] += weight

        for fb in forbidden:
            if fb in scores:
                scores[fb] -= penalty
                notes.append(f"[{rule_id}] 科目 {fb} 遭合規規則扣分 (-{penalty:.2f})")

    # compliance_score 允許為負，最終加權融合時會被 delta 縮放；
    # 這裡不做 0-1 裁切，讓「違反規則」能真正壓低 final_score。
    return scores, notes, triggered_rules


# ----------------------------------------------------------------------------
# 主函式：動態加權預測
# ----------------------------------------------------------------------------

def dynamic_weighted_prediction(
    invoice_data: Any,
    master_model,
    correction_prototypes: Dict[str, np.ndarray],
    seller_preferences: Dict[str, Dict[str, float]],
    compliance_rules: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    多來源動態加權融合預測。

    Parameters
    ----------
    invoice_data : InvoiceData（或 dict / 具同名屬性物件）
    master_model : master_model.MasterModel（含 account_vectors、account_codes、encode()）
    correction_prototypes : Dict[str, np.ndarray]  該客戶的修正原型向量字典
    seller_preferences : Dict[str, Dict[str, float]]  賣方偏好字典
    compliance_rules : Dict[str, Dict[str, Any]]  稅法合規規則字典 {rule_id: rule_dict}
    weights : Dict[str, float]  {"alpha":..., "beta":..., "gamma":..., "delta":...}
    top_k : int  score_breakdown 中額外附上的候選科目數

    Returns
    -------
    Dict[str, Any]，包含 predicted_account, confidence, score_breakdown,
    explanation（含 suggested_action，高風險時明確標示建議轉人工覆核）。
    """
    t0 = time.perf_counter()

    alpha = float(weights.get("alpha", 0.0))
    beta = float(weights.get("beta", 0.0))
    gamma = float(weights.get("gamma", 0.0))
    delta = float(weights.get("delta", 0.0))

    account_codes = list(master_model.account_codes)

    summary = getattr(invoice_data, "summary", None)
    if summary is None and isinstance(invoice_data, dict):
        summary = invoice_data.get("summary", "")
    summary = summary or ""

    seller_ban = getattr(invoice_data, "seller_ban", None)
    if seller_ban is None and isinstance(invoice_data, dict):
        seller_ban = invoice_data.get("seller_ban", "")
    seller_ban = seller_ban or ""

    # Step 1: 向量化摘要
    summary_vector = master_model.encode(summary)

    # Step 2-5: 各來源分數
    master_scores = compute_master_scores(summary_vector, master_model)
    correction_scores = compute_correction_scores(summary_vector, correction_prototypes, account_codes)
    seller_scores = compute_seller_scores(seller_ban, seller_preferences, account_codes)
    compliance_scores, compliance_notes, triggered_rules = compute_compliance_scores(
        invoice_data, compliance_rules, account_codes
    )

    # Step 6: 加權融合
    final_scores: Dict[str, float] = {}
    breakdown: Dict[str, Dict[str, float]] = {}
    for code in account_codes:
        m = master_scores.get(code, 0.0)
        c = correction_scores.get(code, 0.0)
        s = seller_scores.get(code, 0.0)
        t = compliance_scores.get(code, 0.0)
        f = alpha * m + beta * c + gamma * s + delta * t
        final_scores[code] = f
        breakdown[code] = {
            "master_score": round(m, 4),
            "correction_score": round(c, 4),
            "seller_score": round(s, 4),
            "compliance_score": round(t, 4),
            "final_score": round(f, 4),
        }

    # Step 7: argmax 選出預測科目
    predicted_account = max(final_scores, key=final_scores.get)
    raw_confidence = final_scores[predicted_account]
    confidence = max(0.0, min(1.0, raw_confidence))

    # Top-k 候選科目
    ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_k_candidates = [
        {"account_code": code, "account_name": master_model.account_names.get(code, ""),
         "final_score": round(score, 4)}
        for code, score in ranked[:top_k]
    ]

    risk_level = _assess_risk_level(compliance_scores, predicted_account, triggered_rules, compliance_rules)
    suggested_action = _suggest_action(risk_level, confidence, compliance_notes)

    elapsed = time.perf_counter() - t0

    explanation = {
        "predicted_account": predicted_account,
        "account_name": master_model.account_names.get(predicted_account, ""),
        "confidence": round(confidence, 4),
        "risk_level": risk_level,
        "score_breakdown": breakdown,
        "top_k_candidates": top_k_candidates,
        "compliance_notes": compliance_notes,
        "triggered_rules": triggered_rules,
        "suggested_action": suggested_action,
        "weights_used": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "elapsed_seconds": round(elapsed, 6),
    }

    return {
        "predicted_account": predicted_account,
        "confidence": round(confidence, 4),
        "score_breakdown": breakdown,
        "explanation": explanation,
    }


def _assess_risk_level(
    compliance_scores: Dict[str, float],
    predicted_account: str,
    triggered_rules: List[str],
    compliance_rules: Dict[str, Dict[str, Any]],
) -> str:
    """依觸發規則的權重與是否命中預測科目的懲罰，評估風險等級（僅供風險提示參考）。"""
    if not triggered_rules:
        return "low"

    max_weight = 0.0
    predicted_penalized = False
    for rule_id in triggered_rules:
        rule = compliance_rules.get(rule_id, {})
        max_weight = max(max_weight, float(rule.get("weight", 0.0)))
        if predicted_account in rule.get("forbidden_accounts", []):
            predicted_penalized = True

    if predicted_penalized or max_weight >= 0.9:
        return "critical"
    if max_weight >= 0.8:
        return "high"
    if max_weight >= 0.6:
        return "medium"
    return "low"


def _suggest_action(risk_level: str, confidence: float, notes: List[str]) -> str:
    """
    建議行動文字。任務規定：高風險結果必須標示「建議轉人工覆核」，
    且不得使用「避免逃稅」等措辭，一律以「降低錯誤申報、重複列報、
    不得扣抵及憑證不完整風險」描述系統目的。本函式輸出僅為風險提示，
    不構成稅務或法律意見。
    """
    if risk_level == "critical":
        return ("⚠️ 高風險：建議轉人工覆核。本情境可能涉及進項稅額分離列帳、"
                "重複列報或不得扣抵等規定，系統無法自行判定是否違規，"
                "請由企業會計/稅務顧問確認實際處理方式。")
    if risk_level == "high":
        return "建議轉人工覆核：請確認科目是否符合稅務規範後再入帳，本結果不構成稅務或法律意見。"
    if confidence < 0.4:
        return "系統信心不足，建議人工判斷並記錄修正供學習，以降低錯誤申報風險。"
    if risk_level == "medium":
        return "建議留意金額是否超過法定列支限額，並保留相關憑證備查。"
    return "可參考採用系統預測科目，仍建議定期由會計人員複核。"


__all__ = [
    "dynamic_weighted_prediction",
    "compute_master_scores",
    "compute_correction_scores",
    "compute_seller_scores",
    "compute_compliance_scores",
    "cosine_similarity",
]
