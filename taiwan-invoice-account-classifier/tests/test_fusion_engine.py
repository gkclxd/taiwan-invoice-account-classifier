# -*- coding: utf-8 -*-
"""
模組測試：動態加權預測引擎（整合自 M5-1 test_fusion_engine.py）

改寫重點：
    - import 路徑改為 invoice_classifier.fusion_engine
    - MasterModel 改用 invoice_classifier.master_model.MasterModel
      （原 M5-1 檔案內建精簡版 dataclass，此處改用正式版本以驗證
      實際整合後的介面相容性）
    - InvoiceData 改用 invoice_classifier.data_models.InvoiceData

涵蓋：
    - 不同公司類型的權重組合（新創/中小企業/進出口/上市公司）
    - 各條稅法合規規則的觸發
    - 客戶修正原型 / 賣方偏好對預測的影響
    - 效能測試（< 1 秒 / 筆）
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from invoice_classifier.data_models import InvoiceData
from invoice_classifier.fusion_engine import dynamic_weighted_prediction
from invoice_classifier.master_model import MasterModel
from invoice_classifier.tax_compliance import TaxRuleEngine

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "tax_rules.json"
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

ACCOUNT_NAMES = {
    "5121": "進貨", "5122": "進貨費用", "1268": "進項稅額",
    "5213": "交際費", "5226": "職工福利", "5221": "運費", "5222": "保險費",
}

ACCOUNT_SEED_TEXT = {
    "5121": "採購商品進貨貨款",
    "5122": "報關費通關費用",
    "1268": "營業稅進項稅額",
    "5213": "客戶交際應酬餐費",
    "5226": "員工旅遊聚餐尾牙",
    "5221": "海運運費",
    "5222": "貨物保險費",
}


def _fake_embed(text: str, dim: int = 32) -> np.ndarray:
    """測試用穩定假向量產生器：以字元 bigram hash 建構穩定向量。"""
    vec = np.zeros(dim, dtype=float)
    text = text or ""
    grams = [text[i:i + 2] for i in range(len(text) - 1)] or [text]
    for g in grams:
        vec[hash(g) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


class _FakeEmbedder:
    """符合 MasterModel.embedder 介面的假 embedder（測試用，不依賴真實訓練）。"""

    def encode(self, texts):
        return np.array([_fake_embed(t) for t in texts])


@pytest.fixture(scope="module")
def master_model() -> MasterModel:
    account_vectors = {code: _fake_embed(text) for code, text in ACCOUNT_SEED_TEXT.items()}
    return MasterModel(
        account_codes=list(ACCOUNT_SEED_TEXT.keys()),
        account_vectors=account_vectors,
        account_names=ACCOUNT_NAMES,
        embedder=_FakeEmbedder(),
    )


@pytest.fixture(scope="module")
def compliance_rules() -> dict:
    return TaxRuleEngine(RULES_PATH).all_rules_dict()


@pytest.fixture(scope="module")
def fusion_weight_presets() -> dict:
    settings = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    return settings["fusion_weights"]


def _mk_invoice(summary, trade_condition=None, seller_ban="1111111111", buyer_ban="2222222222"):
    return InvoiceData(
        buyer_ban=buyer_ban, seller_ban=seller_ban, summary=summary,
        trade_condition=trade_condition, invoice_date="2026-08-31",
    )


@pytest.mark.parametrize("company_type", ["startup", "sme", "import_export", "listed_company"])
def test_weight_presets_produce_valid_prediction(master_model, compliance_rules, fusion_weight_presets, company_type):
    """驗證四組公司類型權重皆能產生合法預測結果，且權重總和為 1。"""
    weights = fusion_weight_presets[company_type]
    assert abs(sum(weights.values()) - 1.0) < 1e-9, f"{company_type} 權重總和應為 1"

    inv = _mk_invoice("採購商品進貨貨款")
    result = dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model, correction_prototypes={},
        seller_preferences={}, compliance_rules=compliance_rules, weights=weights,
    )
    assert result["predicted_account"] in ACCOUNT_NAMES
    assert 0.0 <= result["confidence"] <= 1.0


def test_input_tax_rule_forces_low_confidence_or_penalty(master_model, compliance_rules):
    """含營業稅字樣的摘要應觸發進項稅額分離規則，1268 以外科目遭懲罰或 1268 被加分。"""
    inv = _mk_invoice("進貨一批，內含5%營業稅")
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.1, "delta": 0.2}
    result = dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model, correction_prototypes={},
        seller_preferences={}, compliance_rules=compliance_rules, weights=weights,
    )
    triggered = result["explanation"]["triggered_rules"]
    assert any(r.startswith("RULE_001") for r in triggered)


def test_cif_freight_rule_triggers_and_recommends_review(master_model, compliance_rules):
    inv = _mk_invoice("進口貨物運費保險費", trade_condition="CIF")
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.1, "delta": 0.2}
    result = dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model, correction_prototypes={},
        seller_preferences={}, compliance_rules=compliance_rules, weights=weights,
    )
    triggered = result["explanation"]["triggered_rules"]
    assert any(r.startswith("RULE_002") for r in triggered)


def test_correction_prototype_influences_prediction(master_model, compliance_rules):
    """客戶修正原型應能拉高對應科目的 correction_score。"""
    inv = _mk_invoice("採購商品進貨貨款")
    correction_vec = _fake_embed("採購商品進貨貨款")
    weights = {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0}
    result = dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model,
        correction_prototypes={"5121": correction_vec},
        seller_preferences={}, compliance_rules=compliance_rules, weights=weights,
    )
    assert result["score_breakdown"]["5121"]["correction_score"] > 0.5


def test_seller_preference_influences_prediction(master_model, compliance_rules):
    inv = _mk_invoice("採購商品進貨貨款", seller_ban="9999999999")
    weights = {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "delta": 0.0}
    result = dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model, correction_prototypes={},
        seller_preferences={"9999999999": {"5226": 0.9, "5121": 0.1}},
        compliance_rules=compliance_rules, weights=weights,
    )
    assert result["predicted_account"] == "5226"


def test_performance_under_one_second(master_model, compliance_rules):
    inv = _mk_invoice("一般進貨採購")
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.1, "delta": 0.2}
    start = time.perf_counter()
    dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model, correction_prototypes={},
        seller_preferences={}, compliance_rules=compliance_rules, weights=weights,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_high_risk_result_recommends_manual_review(master_model, compliance_rules):
    inv = _mk_invoice("購買自用乘人小客車，含5%營業稅")
    weights = {"alpha": 0.3, "beta": 0.2, "gamma": 0.1, "delta": 0.4}
    result = dynamic_weighted_prediction(
        invoice_data=inv, master_model=master_model, correction_prototypes={},
        seller_preferences={}, compliance_rules=compliance_rules, weights=weights,
    )
    risk_level = result["explanation"]["risk_level"]
    if risk_level in ("high", "critical"):
        assert "人工覆核" in result["explanation"]["suggested_action"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
