# -*- coding: utf-8 -*-
"""
邊界情況測試用分類器包裝（整合自 M8-2 classifier_stub.py，改名避免與 M8-1 版本衝突）

模組：tests/boundary_classifier_stub.py
說明：
    驗證系統在「不正常／極端輸入」下的穩定性 —— 是否會拋出未捕捉例外、
    是否能在合理時間內完成、是否回傳結構完整的錯誤或降級結果。

    改寫重點（相較原始 M8-2 classifier_stub.py）：
        - 不再需要猜測 M1-2/M5-1/M5-2 資料夾路徑，直接 import 正式套件：
          invoice_classifier.data_models（Pydantic InvoiceData 嚴格驗證）、
          invoice_classifier.fusion_engine、invoice_classifier.storage。
        - 驗證優先於呼叫核心演算法：使用 Pydantic InvoiceData 做嚴格驗證
          （10 碼統編、amount >= 0、ISO 8601 日期、trade_condition 僅接受
          FOB/CIF）。驗證失敗時回傳結構化錯誤，不拋出未捕捉例外。
        - 任何未預期例外都被最外層 try/except 攔截，轉為
          {"error": True, "error_type":..., "error_message":...}，
          確保 predict() 本身「絕不」讓呼叫端程式崩潰。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pydantic

from invoice_classifier.data_models import InvoiceData
from invoice_classifier.fusion_engine import dynamic_weighted_prediction
from invoice_classifier.master_model import MasterModel
from invoice_classifier.tax_compliance import TaxRuleEngine
from invoice_classifier.storage import CorrectionManager, CorrectionRecord

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

_DIM = 64
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


def _fake_embed(text: str) -> np.ndarray:
    """確定性假向量產生器（hashlib.md5），對空字串／emoji／特殊字元一律安全。"""
    vec = np.zeros(_DIM, dtype=float)
    text = text or ""
    chars = list(text)  # 以 unicode code point 逐字處理，emoji 也不會造成 slicing 例外
    grams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)] or (chars or [" "])
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8", errors="ignore")).hexdigest(), 16) % _DIM
        vec[h] += 1.0
    return vec


class _DeterministicEmbedder:
    def encode(self, texts):
        return np.array([_fake_embed(t) for t in texts])


def _build_master_model() -> MasterModel:
    account_vectors: Dict[str, np.ndarray] = {}
    for code in ACCOUNT_CODES:
        kws = _KEYWORD_PROTOTYPES.get(code, [ACCOUNT_NAMES.get(code, code)])
        vecs = [_fake_embed(kw) for kw in kws]
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        account_vectors[code] = avg / norm if norm > 1e-12 else avg
    return MasterModel(
        account_codes=ACCOUNT_CODES, account_vectors=account_vectors,
        account_names=ACCOUNT_NAMES, embedder=_DeterministicEmbedder(),
    )


class InvoiceValidationError(Exception):
    """代表輸入驗證失敗（非系統崩潰），攜帶結構化錯誤欄位清單。"""

    def __init__(self, errors: List[Dict[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in errors))


def _validate_invoice_dict(invoice: Any) -> InvoiceData:
    """
    使用 Pydantic InvoiceData 驗證原始 dict 輸入。

    驗證失敗時將 pydantic.ValidationError 轉為 InvoiceValidationError
    （結構化欄位錯誤清單），由呼叫端 predict() 統一捕捉，絕不讓例外
    往外洩漏造成崩潰。
    """
    if invoice is None:
        raise InvoiceValidationError([{"field": "invoice", "message": "輸入不可為 None"}])
    if not isinstance(invoice, dict):
        raise InvoiceValidationError(
            [{"field": "invoice", "message": f"輸入必須為 dict，實際收到 {type(invoice).__name__}"}]
        )
    try:
        return InvoiceData(**invoice)
    except pydantic.ValidationError as exc:
        errors = [
            {"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        raise InvoiceValidationError(errors) from exc
    except TypeError as exc:
        # 例如傳入未知欄位、或 extra="forbid" 觸發的非 ValidationError 情境
        raise InvoiceValidationError([{"field": "invoice", "message": str(exc)}]) from exc


class BoundaryTestClassifier:
    """
    邊界測試專用分類器包裝。

    predict(invoice) -> dict 永遠回傳一個 dict，絕不拋出未捕捉例外：
        - 驗證失敗：{"error": True, "error_type": "ValidationError",
                     "error_message": ..., "field_errors": [...]}
        - 核心演算法拋例外（不應發生，但仍加保護）：
                    {"error": True, "error_type": "InternalError", "error_message": ...}
        - 正常：{"error": False, "predicted_account":..., "confidence":...,
                 "risk_level":..., "compliance_notes": [...], "elapsed_seconds":...}
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        correction_prototypes: Optional[Dict[str, Any]] = None,
        seller_preferences: Optional[Dict[str, Any]] = None,
        rules_path=None,
    ):
        self.weights = weights or {"alpha": 0.5, "beta": 0.2, "gamma": 0.1, "delta": 0.2}
        self.rule_engine = TaxRuleEngine(rules_path) if rules_path else TaxRuleEngine()
        self.compliance_rules = self.rule_engine.all_rules_dict()
        self.master_model = _build_master_model()
        self.correction_prototypes = correction_prototypes or {}
        self.seller_preferences = seller_preferences or {}

    def predict(self, invoice: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            invoice_obj = _validate_invoice_dict(invoice)
        except InvoiceValidationError as exc:
            return {
                "error": True, "error_type": "ValidationError", "error_message": str(exc),
                "field_errors": exc.errors, "elapsed_seconds": round(time.perf_counter() - t0, 6),
            }
        except Exception as exc:  # noqa: BLE001 — 最終防線
            return {
                "error": True, "error_type": "UnexpectedValidationError",
                "error_message": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - t0, 6),
            }

        try:
            result = dynamic_weighted_prediction(
                invoice_data=invoice_obj, master_model=self.master_model,
                correction_prototypes=self.correction_prototypes,
                seller_preferences=self.seller_preferences,
                compliance_rules=self.compliance_rules, weights=self.weights,
            )
            elapsed = time.perf_counter() - t0
            return {
                "error": False, "degraded": False,
                "predicted_account": result["predicted_account"],
                "confidence": result["confidence"],
                "risk_level": result["explanation"]["risk_level"],
                "compliance_notes": result["explanation"]["compliance_notes"],
                "triggered_rules": result["explanation"].get("triggered_rules", []),
                "score_breakdown": result["score_breakdown"],
                "elapsed_seconds": round(elapsed, 6),
            }
        except Exception as exc:  # noqa: BLE001 — predict() 對外承諾絕不崩潰
            return {
                "error": True, "error_type": "InternalError",
                "error_message": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - t0, 6),
            }


def get_correction_manager_cls():
    return CorrectionManager


def get_correction_record_cls():
    return CorrectionRecord


__all__ = [
    "BoundaryTestClassifier", "InvoiceValidationError",
    "get_correction_manager_cls", "get_correction_record_cls",
    "ACCOUNT_CODES", "ACCOUNT_NAMES",
]
