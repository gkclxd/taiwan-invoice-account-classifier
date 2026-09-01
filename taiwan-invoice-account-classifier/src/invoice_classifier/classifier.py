# -*- coding: utf-8 -*-
"""
invoice_classifier.classifier
================================
整合全流程的主要對外 API（本專案新設計，串接以下所有模組）：

    InvoiceData
        │
        ▼
    preprocessor.InvoicePreprocessor  ── 斷詞 / 停用詞過濾
        │
        ▼
    preprocessor.InvoiceVectorizer / master_model.MasterModel.encode()
        │  （母版模型使用自身 embedder 對「原始摘要」向量化，
        │    以與訓練時的向量空間保持一致）
        ▼
    personalization.PrototypeStore     ── 客戶修正原型 + 賣方偏好
        │
        ▼
    fusion_engine.dynamic_weighted_prediction()  ── 四來源加權融合
        │
        ▼
    tax_compliance.assess_invoice_risk()          ── 五維度風險評分（補充）
        │
        ▼
    explanation.build_explanation()               ── 組裝人類可讀解釋
        │
        ▼
    AccountPrediction + PredictionExplanation

InvoiceClassifier 為對外主要介面，提供單張發票 predict() 方法。

⚠️ 系統定位重要聲明：
    本系統為「稅務輔助與風險提示工具」，predict() 之輸出結果
    （包含建議科目、置信度、風險等級、是否建議人工覆核）僅供內部
    記帳與風險自我檢核參考，**不構成正式稅務或法律意見**。系統不會
    自行判定使用者有無逃漏稅意圖，也不會僅依單一關鍵字直接宣稱違法；
    任何高風險或系統無法確定之判斷，均會標示「建議轉人工覆核」，
    請務必經企業會計人員或稅務顧問確認後再行採用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .data_models import AccountPrediction, CompanyProfile, InvoiceData, RiskLevel
from .explanation import PredictionExplanation, build_explanation
from .fusion_engine import dynamic_weighted_prediction
from .master_model import MasterModel
from .personalization import PrototypeStore, compute_seller_preferences
from .preprocessor import InvoicePreprocessor
from .storage import CorrectionManager
from .tax_compliance import TaxRuleEngine, assess_invoice_risk, get_default_engine

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = _THIS_DIR.parent.parent / "models" / "master_model.pkl"

DEFAULT_WEIGHTS = {"alpha": 0.4, "beta": 0.3, "gamma": 0.1, "delta": 0.2}


@dataclass
class ClassificationResult:
    """InvoiceClassifier.predict() 的完整回傳結果。"""

    prediction: AccountPrediction
    explanation: PredictionExplanation
    fusion_raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        pred_dict = (
            self.prediction.model_dump() if hasattr(self.prediction, "model_dump")
            else self.prediction.dict()
        )
        return {
            "prediction": pred_dict,
            "explanation": self.explanation.to_dict(),
        }


class InvoiceClassifier:
    """
    台灣電子發票會計科目分類系統 — 對外主要 API。

    使用範例：
        classifier = InvoiceClassifier(
            master_model_path="models/master_model.pkl",
            company_profile=CompanyProfile(...),
        )
        result = classifier.predict(invoice_data)
        print(result.explanation.human_readable_summary)

    注意：
        - 本 repo 不包含已訓練好的 master_model.pkl，使用前請先執行
          master_model.train_from_csv() 以你自己準備的訓練資料訓練並
          save() 出模型檔案（見 examples/ 與 docs/ 說明）。
        - custom_account_mapping：若 company_profile 提供了自訂會計科目
          對照表，predict() 會在最終輸出前套用轉換（見 _apply_custom_mapping）。
    """

    def __init__(
        self,
        master_model_path: Optional[Path] = None,
        master_model: Optional[MasterModel] = None,
        company_profile: Optional[CompanyProfile] = None,
        corrections_data_dir: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        rules_path: Optional[Path] = None,
    ):
        self.company_profile = company_profile
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)

        # 依 CompanyProfile.compliance_weight 動態調整 delta（合規規則權重），
        # 其餘三項等比例縮放，確保四項總和維持為 1。
        if company_profile is not None:
            self.weights = self._rebalance_weights(self.weights, company_profile.compliance_weight)

        self.preprocessor = InvoicePreprocessor()
        self.rule_engine: TaxRuleEngine = (
            TaxRuleEngine(rules_path) if rules_path else get_default_engine()
        )

        if master_model is not None:
            self.master_model = master_model
        else:
            path = Path(master_model_path) if master_model_path else DEFAULT_MODEL_PATH
            self.master_model = self._load_or_raise(path)

        data_dir = corrections_data_dir or "./data/corrections"
        self.correction_manager = CorrectionManager(data_dir=data_dir)
        self._prototype_store = PrototypeStore()
        self._seller_preferences: Dict[str, Dict[str, float]] = {}
        self._corrections_loaded_for: Optional[str] = None

    @staticmethod
    def _load_or_raise(path: Path) -> MasterModel:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到母版模型檔案：{path}。\n"
                f"本 repo 不包含已訓練完成的模型二進位檔（.pkl），請先執行訓練："
                f"\n\n    from invoice_classifier.master_model import train_from_csv\n"
                f"    model = train_from_csv('your_training_data.csv', "
                f"'config/account_subjects.csv')\n"
                f"    model.save('models/master_model.pkl')\n\n"
                f"詳見 docs/ 或 examples/run_classification_demo.py 中的說明。"
            )
        return MasterModel.load(path)

    @staticmethod
    def _rebalance_weights(base_weights: Dict[str, float], compliance_weight: float) -> Dict[str, float]:
        """
        依公司設定的 compliance_weight（對應 delta）重新分配四個權重，
        使 alpha+beta+gamma+delta 維持為 1。alpha/beta/gamma 依原比例縮放。
        """
        delta = max(0.0, min(1.0, float(compliance_weight)))
        remaining = 1.0 - delta
        base_abc_sum = base_weights["alpha"] + base_weights["beta"] + base_weights["gamma"]
        if base_abc_sum <= 0:
            alpha = beta = gamma = remaining / 3.0
        else:
            alpha = base_weights["alpha"] / base_abc_sum * remaining
            beta = base_weights["beta"] / base_abc_sum * remaining
            gamma = base_weights["gamma"] / base_abc_sum * remaining
        return {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}

    # -- 修正紀錄 / 賣方偏好載入 --------------------------------------------

    def load_corrections(self, company_tax_id: str) -> None:
        """
        載入指定公司統編的歷史修正紀錄，建立個人化原型向量與賣方偏好。
        建議在 predict() 前先呼叫一次；若未呼叫，predict() 會在需要時
        自動載入（惰性初始化）。
        """
        records_raw = self.correction_manager.query_corrections(company_tax_id)
        # storage.CorrectionRecord -> data_models.CorrectionRecord 相容轉換
        from .data_models import CorrectionRecord as SchemaCorrectionRecord

        records = [
            SchemaCorrectionRecord(
                invoice_id=r.invoice_id, timestamp=r.timestamp, summary=r.summary,
                summary_vector=r.summary_vector, original_pred=r.original_pred,
                corrected_to=r.corrected_to, confidence_weight=r.confidence_weight,
                seller_ban=r.seller_ban, buyer_ban=r.buyer_ban,
            )
            for r in records_raw
        ]
        self._prototype_store.bulk_load(records)
        self._seller_preferences = compute_seller_preferences(records)
        self._corrections_loaded_for = company_tax_id

    def record_correction(
        self,
        company_tax_id: str,
        invoice_id: str,
        summary: str,
        original_pred: str,
        corrected_to: str,
        seller_ban: str,
        buyer_ban: str,
        confidence_weight: float = 1.0,
    ) -> None:
        """
        記錄一筆人工修正，並立即以增量方式更新個人化原型向量與賣方偏好
        （不需重新呼叫 load_corrections() 重算全部歷史資料）。
        """
        self.correction_manager.add_correction(
            company_tax_id=company_tax_id, invoice_id=invoice_id, summary=summary,
            original_pred=original_pred, corrected_to=corrected_to,
            seller_ban=seller_ban, buyer_ban=buyer_ban, confidence_weight=confidence_weight,
        )
        from .data_models import CorrectionRecord as SchemaCorrectionRecord

        vector = self.correction_manager.vectorizer.encode(summary)
        record = SchemaCorrectionRecord(
            invoice_id=invoice_id, timestamp=datetime.now(timezone.utc).isoformat(),
            summary=summary, summary_vector=vector, original_pred=original_pred,
            corrected_to=corrected_to, confidence_weight=confidence_weight,
            seller_ban=seller_ban, buyer_ban=buyer_ban,
        )
        if self._corrections_loaded_for != company_tax_id:
            self.load_corrections(company_tax_id)
        else:
            self._prototype_store.add_record(record)
            seller_stat = self._seller_preferences.setdefault(seller_ban, {})
            # 簡易增量更新賣方偏好比例（重新正規化）
            counts = {k: v for k, v in seller_stat.items()}
            counts[corrected_to] = counts.get(corrected_to, 0.0) + 1.0
            total = sum(counts.values())
            self._seller_preferences[seller_ban] = {k: v / total for k, v in counts.items()}

    # -- 自訂科目對照表 --------------------------------------------------

    def _apply_custom_mapping(self, account_code: str, account_name: str) -> tuple:
        """
        套用 CompanyProfile.custom_account_mapping（企業自訂會計科目對照表）。
        系統內建科目代碼並非所有公司唯一標準，若企業提供對照表，
        以企業自訂代碼／名稱覆蓋系統預設值。
        """
        if self.company_profile and self.company_profile.custom_account_mapping:
            mapped = self.company_profile.custom_account_mapping.get(account_code)
            if mapped:
                return mapped, account_name
        return account_code, account_name

    # -- 主要預測介面 --------------------------------------------------

    def predict(
        self,
        invoice: InvoiceData,
        company_tax_id: Optional[str] = None,
        top_k: int = 3,
    ) -> ClassificationResult:
        """
        對單張發票進行會計科目預測。

        流程：preprocessor → master_model(encode) → personalization →
              fusion_engine → tax_compliance → explanation

        Parameters
        ----------
        invoice : InvoiceData
        company_tax_id : 選填。若提供且尚未載入該公司的修正紀錄，
            會自動呼叫 load_corrections()。若未提供，則不套用個人化
            學習層（correction_score 全數為 0，退化為母版模型+賣方偏好+合規規則）。
        top_k : score_breakdown 附帶的候選科目數。

        Returns
        -------
        ClassificationResult
        """
        # 1. 前處理（斷詞 + 停用詞過濾），主要用於稅法規則關鍵字比對的一致性；
        #    母版模型向量化直接對「原始摘要」進行（與訓練時一致，見 MasterModel.encode）。
        _ = self.preprocessor.preprocess(invoice.summary)

        # 2. 個人化資料載入（惰性）
        if company_tax_id and self._corrections_loaded_for != company_tax_id:
            self.load_corrections(company_tax_id)

        correction_prototypes = {
            code: proto["prototype_vector"]
            for code, proto in self._prototype_store.get_prototypes().items()
        }

        # 3. 稅法合規規則庫
        compliance_rules = self.rule_engine.all_rules_dict()

        # 4. Fusion Engine：四來源加權融合
        fusion_result = dynamic_weighted_prediction(
            invoice_data=invoice,
            master_model=self.master_model,
            correction_prototypes=correction_prototypes,
            seller_preferences=self._seller_preferences,
            compliance_rules=compliance_rules,
            weights=self.weights,
            top_k=top_k,
        )

        predicted_account = fusion_result["predicted_account"]
        confidence = fusion_result["confidence"]

        # 5. 稅法合規補充檢查：五維度風險評分（結合 fusion 的規則觸發風險等級，
        #    取兩者較高者作為最終風險等級，避免僅靠規則觸發而遺漏金額/憑證類風險）
        risk_result = assess_invoice_risk(invoice, account_code=predicted_account)
        final_risk_level = _max_risk_level(
            fusion_result["explanation"]["risk_level"], risk_result["risk_level"]
        )
        fusion_result["explanation"]["risk_level"] = final_risk_level
        fusion_result["explanation"]["compliance_notes"] = list(dict.fromkeys(
            fusion_result["explanation"]["compliance_notes"] + risk_result["compliance_notes"]
        ))

        # 6. 套用自訂科目對照表
        account_name = fusion_result["explanation"]["account_name"]
        mapped_code, mapped_name = self._apply_custom_mapping(predicted_account, account_name)

        risk_level_enum = RiskLevel(final_risk_level)
        prediction = AccountPrediction(
            account_code=predicted_account,  # 保留系統標準代碼（4 碼驗證需求）
            account_name=mapped_name or account_name or predicted_account,
            confidence=confidence,
            risk_level=risk_level_enum,
        )

        # 7. 組裝解釋
        explanation = build_explanation(fusion_result)
        if mapped_code != predicted_account:
            explanation.human_readable_summary += (
                f"\n（企業自訂對照：系統科目 {predicted_account} 對應貴公司科目 {mapped_code}）"
            )

        return ClassificationResult(
            prediction=prediction, explanation=explanation, fusion_raw=fusion_result,
        )

    def predict_batch(
        self, invoices: List[InvoiceData], company_tax_id: Optional[str] = None, top_k: int = 3,
    ) -> List[ClassificationResult]:
        """批次預測多張發票（依序呼叫 predict()，個人化資料僅載入一次）。"""
        if company_tax_id and self._corrections_loaded_for != company_tax_id:
            self.load_corrections(company_tax_id)
        return [self.predict(inv, company_tax_id=company_tax_id, top_k=top_k) for inv in invoices]


_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _max_risk_level(a: str, b: str) -> str:
    ra = _RISK_ORDER.get(a, 0)
    rb = _RISK_ORDER.get(b, 0)
    return a if ra >= rb else b


__all__ = ["InvoiceClassifier", "ClassificationResult", "DEFAULT_WEIGHTS"]
