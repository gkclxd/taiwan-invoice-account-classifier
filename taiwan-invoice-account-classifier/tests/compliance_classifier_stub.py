# -*- coding: utf-8 -*-
"""
測試用分類器包裝（整合自 M8-1 classifier_stub.py，改名避免與 M8-2 版本衝突）

模組：tests/compliance_classifier_stub.py
說明：
    test_compliance_cases.py / test_compliance_runner.py 需要一個具備
    `predict(invoice_dict) -> dict` 介面的物件來執行合規正向測試案例。

    改寫重點（相較原始 M8-1 classifier_stub.py）：
        - 不再需要猜測 M4-1/M5-1 資料夾路徑，直接 import 正式套件
          invoice_classifier.fusion_engine / invoice_classifier.tax_compliance。
        - 母版模型改用 invoice_classifier.master_model.MasterModel
          （正式版本），仍搭配確定性假 encoder（hashlib.md5）以確保測試
          結果可重現，不代表正式訓練後模型的分類品質。
        - 稅法合規規則改由 config/tax_rules.json（合併後版本）載入，
          rule_id 前綴為大寫 RULE_00X（見任務決策 #1 與 #3）。
        - 金額上限檢查（交際費 2‰）邏輯仍由本檔案的 _amount_cap_check()
          額外實作（fusion_engine 不內建全年累計金額比對）。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np

from invoice_classifier.data_models import InvoiceData
from invoice_classifier.fusion_engine import dynamic_weighted_prediction
from invoice_classifier.master_model import MasterModel
from invoice_classifier.tax_compliance import TaxRuleEngine

ACCOUNT_CODES = [
    "5121", "5122", "1268", "5213", "5214", "5215", "5216",
    "5217", "5218", "5219", "5221", "5222", "5226", "5253",
]

ACCOUNT_NAMES = {
    "5121": "進貨", "5122": "進貨費用", "1268": "進項稅額", "5213": "交際費",
    "5214": "交通費", "5215": "差旅費", "5216": "辦公費", "5217": "郵電費",
    "5218": "水電費", "5219": "廣告費", "5221": "運費", "5222": "保險費",
    "5226": "職工福利", "5253": "折舊費用",
}

_KEYWORD_PROTOTYPES = {
    "5121": ["進貨", "貨物", "商品", "原物料", "原料"],
    "5122": ["進貨費用", "報關", "通關"],
    "1268": ["稅", "營業稅", "VAT", "稅額"],
    "5213": ["交際", "應酬", "宴客", "送禮", "客戶"],
    "5215": ["差旅", "出差", "住宿", "機票"],
    "5216": ["辦公", "文具", "耗材"],
    "5221": ["運費", "運送"],
    "5222": ["保險費", "保險"],
    "5226": ["員工", "旅遊", "聚餐", "尾牙", "福利"],
}
_DIM = 64


def _fake_embed(text: str) -> np.ndarray:
    """確定性假向量產生器（hashlib.md5，任何程序、任何時間執行結果皆相同，
    確保測試自動化且可重現，不受 Python 字串雜湊隨機化影響）。"""
    vec = np.zeros(_DIM, dtype=float)
    text = text or ""
    grams = [text[i:i + 2] for i in range(len(text) - 1)] or [text]
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % _DIM
        vec[h] += 1.0
    return vec


class _DeterministicEmbedder:
    def encode(self, texts):
        return np.array([_fake_embed(t) for t in texts])


def _build_master_model() -> MasterModel:
    """建立簡化母版模型：每個科目原型向量 = 該科目關鍵字樣本句的假向量平均。"""
    account_vectors: Dict[str, np.ndarray] = {}
    for code in ACCOUNT_CODES:
        kws = _KEYWORD_PROTOTYPES.get(code, [ACCOUNT_NAMES.get(code, code)])
        vecs = [_fake_embed(kw) for kw in kws]
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        account_vectors[code] = avg / norm if norm > 1e-12 else avg

    return MasterModel(
        account_codes=ACCOUNT_CODES,
        account_vectors=account_vectors,
        account_names=ACCOUNT_NAMES,
        embedder=_DeterministicEmbedder(),
    )


# 假設情境：測試用進貨淨額基準（供計算 1.5‰-2‰ 上限之對照示範）
_ASSUMED_NET_PURCHASE_BASE = 10_000_000.0
_ENTERTAINMENT_CAP_RATIO = 0.002
_ENTERTAINMENT_CAP = _ASSUMED_NET_PURCHASE_BASE * _ENTERTAINMENT_CAP_RATIO


def _amount_cap_check(summary: str, amount: Optional[float]) -> Optional[str]:
    entertainment_kws = ["交際", "應酬", "宴客", "送禮"]
    if not any(kw in (summary or "") for kw in entertainment_kws):
        return None
    if amount is not None and amount > _ENTERTAINMENT_CAP:
        return (
            f"[amount_check] 交際費金額 {amount:,.0f} 元已超過假設列支上限 "
            f"{_ENTERTAINMENT_CAP:,.0f} 元（進貨淨額 2‰），可能於申報時遭剔除，"
            f"請核對全年累計金額並轉人工覆核。"
        )
    return None


class ComplianceTestClassifier:
    """
    包裝 fusion_engine，提供 predict(invoice: dict) -> dict 介面給測試執行函式使用。

    weights 採用「一般中小企業」預設權重（alpha=0.5, beta=0.2, gamma=0.1, delta=0.2），
    測試不涉及客戶修正／賣方偏好資料，beta、gamma 來源分數恆為 0。
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, rules_path=None):
        self.weights = weights or {"alpha": 0.5, "beta": 0.2, "gamma": 0.1, "delta": 0.2}
        self.master_model = _build_master_model()
        self.rule_engine = TaxRuleEngine(rules_path) if rules_path else TaxRuleEngine()
        self.compliance_rules = self.rule_engine.all_rules_dict()

    def predict(self, invoice: Dict[str, Any]) -> Dict[str, Any]:
        invoice_obj = InvoiceData(
            buyer_ban=invoice.get("buyer_ban", "0000000000"),
            seller_ban=invoice.get("seller_ban", "0000000000"),
            summary=invoice.get("summary", ""),
            amount=invoice.get("amount"),
            trade_condition=invoice.get("trade_condition"),
            invoice_date=invoice.get("invoice_date", "2026-08-31"),
        )

        result = dynamic_weighted_prediction(
            invoice_data=invoice_obj, master_model=self.master_model,
            correction_prototypes={}, seller_preferences={},
            compliance_rules=self.compliance_rules, weights=self.weights,
        )

        amount_note = _amount_cap_check(invoice.get("summary", ""), invoice.get("amount"))
        if amount_note:
            result["explanation"]["compliance_notes"].append(amount_note)
            result["amount_cap_warning"] = True
        else:
            result["amount_cap_warning"] = False

        result["risk_level"] = result["explanation"]["risk_level"]
        result["compliance_notes"] = result["explanation"]["compliance_notes"]

        triggered_rule_ids = result["explanation"].get("triggered_rules", [])
        forbidden_set = set()
        for rule_id in triggered_rule_ids:
            rule = self.compliance_rules.get(rule_id, {})
            for fb in rule.get("forbidden_accounts", []):
                forbidden_set.add(fb)
        result["forbidden_accounts_penalized"] = sorted(forbidden_set)

        return result


__all__ = ["ComplianceTestClassifier", "ACCOUNT_CODES", "ACCOUNT_NAMES"]
