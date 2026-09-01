# -*- coding: utf-8 -*-
"""
invoice_classifier.data_models
================================
系統核心資料結構定義（整合自 M1-2 schemas.py）。

使用 Pydantic BaseModel 定義五大核心資料結構，並內建欄位驗證規則
（統一編號 10 碼數字、會計科目代碼 4 碼、0-1 區間的權重／置信度、
ISO 8601 時間格式等）。

結構清單：
    1. InvoiceData          — 發票輸入資料
    2. AccountPrediction    — 預測結果
    3. CorrectionRecord     — 客戶修正紀錄
    4. TaxComplianceRule    — 稅法合規規則
    5. CompanyProfile       — 公司設定檔（含自訂會計科目對照表擴充點）

重要聲明：
    本模組定義之「會計科目代碼」預設對照台灣《商業會計科目表》常見科目，
    僅為系統預設值，並非適用於所有公司之唯一標準。CompanyProfile 提供
    custom_account_mapping 欄位，允許企業自訂科目對照表以覆蓋系統預設值。

Pydantic 版本相容性：
    本檔案同時相容 Pydantic v1 與 v2，會自動偵測目前安裝的 Pydantic
    主版本並套用對應語法（v2 使用 field_validator / model_config；
    v1 使用 validator / Config class）。
"""

from __future__ import annotations

import re
from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict

import pydantic

PYDANTIC_V2 = pydantic.VERSION.startswith("2")

if PYDANTIC_V2:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
else:
    from pydantic import BaseModel, Field, validator  # type: ignore


# ---------------------------------------------------------------------------
# 共用常數與列舉（Enum）
# ---------------------------------------------------------------------------

class TradeCondition(str, Enum):
    """貿易條件（Incoterms 簡化版，用於判斷運費／保險費是否重複列報）"""
    FOB = "FOB"   # Free on Board：買方負擔主運費，未含在貨價中
    CIF = "CIF"   # Cost, Insurance and Freight：運費、保險費已含在貨價中


class RiskLevel(str, Enum):
    """稅務風險等級（僅供風險提示參考，非稅務或法律意見）"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 統一編號：台灣公司/商業登記統一編號，固定 10 碼數字
BAN_PATTERN = re.compile(r"^\d{10}$")

# 會計科目代碼：依《商業會計科目表》編碼規則，固定 4 碼數字（如 5121、1268）
ACCOUNT_CODE_PATTERN = re.compile(r"^\d{4}$")


def _validate_ban(value: str, field_name: str = "統一編號") -> str:
    if not isinstance(value, str) or not BAN_PATTERN.match(value):
        raise ValueError(f"{field_name}必須為 10 碼數字字串，實際收到：{value!r}")
    return value


def _validate_account_code(value: str, field_name: str = "會計科目代碼") -> str:
    if not isinstance(value, str) or not ACCOUNT_CODE_PATTERN.match(value):
        raise ValueError(f"{field_name}必須為 4 碼數字字串（依《商業會計科目表》預設格式，"
                          f"企業可自訂對照表覆寫），實際收到：{value!r}")
    return value


def _validate_iso8601(value: str, field_name: str = "時間欄位") -> str:
    try:
        normalized = value.replace("Z", "+00:00")
        if len(value) <= 10:
            date.fromisoformat(value)
        else:
            datetime.fromisoformat(normalized)
    except Exception as exc:
        raise ValueError(f"{field_name}必須為 ISO 8601 格式，實際收到：{value!r}（{exc}）")
    return value


def _validate_unit_interval(value: float, field_name: str) -> float:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{field_name}必須介於 0-1 之間，實際收到：{value!r}")
    return value


# ---------------------------------------------------------------------------
# 1. InvoiceData — 發票輸入資料
# ---------------------------------------------------------------------------

class InvoiceData(BaseModel):
    """
    發票輸入資料（進項電子發票的原始輸入）

    重要提醒：
        trade_condition 需標註 FOB/CIF，用於判斷運費、保險費是否
        已內含於貨價中，避免重複列報費用（詳見 tax_compliance 模組說明）。
    """

    invoice_id: Optional[str] = Field(default=None, description="發票編號（可選）")
    buyer_ban: str = Field(..., description="買方統一編號（10 碼數字）")
    seller_ban: str = Field(..., description="賣方統一編號（10 碼數字）")
    summary: str = Field(..., min_length=1, description="發票摘要（繁體中文）")
    amount: Optional[float] = Field(default=None, ge=0, description="發票金額（含稅），可選")
    trade_condition: Optional[TradeCondition] = Field(
        default=None, description="貿易條件（FOB/CIF），可選"
    )
    invoice_date: str = Field(..., description="發票日期（ISO 8601，如 2026-08-31）")

    if PYDANTIC_V2:
        model_config = ConfigDict(use_enum_values=True, extra="forbid")

        @field_validator("buyer_ban")
        @classmethod
        def _check_buyer_ban(cls, v: str) -> str:
            return _validate_ban(v, "買方統一編號")

        @field_validator("seller_ban")
        @classmethod
        def _check_seller_ban(cls, v: str) -> str:
            return _validate_ban(v, "賣方統一編號")

        @field_validator("invoice_date")
        @classmethod
        def _check_invoice_date(cls, v: str) -> str:
            return _validate_iso8601(v, "發票日期")
    else:
        class Config:
            use_enum_values = True
            extra = "forbid"

        @validator("buyer_ban")
        def _check_buyer_ban(cls, v):  # noqa: N805
            return _validate_ban(v, "買方統一編號")

        @validator("seller_ban")
        def _check_seller_ban(cls, v):  # noqa: N805
            return _validate_ban(v, "賣方統一編號")

        @validator("invoice_date")
        def _check_invoice_date(cls, v):  # noqa: N805
            return _validate_iso8601(v, "發票日期")


# ---------------------------------------------------------------------------
# 2. AccountPrediction — 預測結果
# ---------------------------------------------------------------------------

class AccountPrediction(BaseModel):
    """會計科目預測結果（輔助參考用，非最終稅務判斷）"""

    account_code: str = Field(..., description="會計科目代碼（4 碼，如 5121）")
    account_name: str = Field(..., min_length=1, description="會計科目名稱（如「進貨」）")
    confidence: float = Field(..., description="置信度（0-1）")
    risk_level: RiskLevel = Field(..., description="稅務風險等級")

    if PYDANTIC_V2:
        model_config = ConfigDict(use_enum_values=True, extra="forbid")

        @field_validator("account_code")
        @classmethod
        def _check_account_code(cls, v: str) -> str:
            return _validate_account_code(v)

        @field_validator("confidence")
        @classmethod
        def _check_confidence(cls, v: float) -> float:
            return _validate_unit_interval(v, "confidence")
    else:
        class Config:
            use_enum_values = True
            extra = "forbid"

        @validator("account_code")
        def _check_account_code(cls, v):  # noqa: N805
            return _validate_account_code(v)

        @validator("confidence")
        def _check_confidence(cls, v):  # noqa: N805
            return _validate_unit_interval(v, "confidence")


# ---------------------------------------------------------------------------
# 3. CorrectionRecord — 客戶修正紀錄
# ---------------------------------------------------------------------------

class CorrectionRecord(BaseModel):
    """
    客戶人工修正紀錄

    用途：
        個人化學習層（personalization 模組）依此紀錄學習「客戶偏好
        原型向量」與「賣方 → 科目」偏好，支援增量學習。
    """

    invoice_id: str = Field(..., min_length=1, description="發票編號")
    timestamp: str = Field(..., description="修正時間（ISO 8601）")
    summary: str = Field(..., min_length=1, description="發票摘要")
    summary_vector: List[float] = Field(..., min_length=1, description="摘要的向量表示")
    original_pred: str = Field(..., description="母版模型原始預測科目代碼")
    corrected_to: str = Field(..., description="人工修正後的科目代碼")
    confidence_weight: float = Field(..., description="修正可信度（0-1）")
    seller_ban: str = Field(..., description="賣方統一編號（10 碼數字）")
    buyer_ban: str = Field(..., description="買方統一編號（10 碼數字）")

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

        @field_validator("original_pred")
        @classmethod
        def _check_original_pred(cls, v: str) -> str:
            return _validate_account_code(v, "母版預測科目代碼")

        @field_validator("corrected_to")
        @classmethod
        def _check_corrected_to(cls, v: str) -> str:
            return _validate_account_code(v, "修正後科目代碼")

        @field_validator("confidence_weight")
        @classmethod
        def _check_confidence_weight(cls, v: float) -> float:
            return _validate_unit_interval(v, "confidence_weight")

        @field_validator("seller_ban")
        @classmethod
        def _check_seller_ban(cls, v: str) -> str:
            return _validate_ban(v, "賣方統一編號")

        @field_validator("buyer_ban")
        @classmethod
        def _check_buyer_ban(cls, v: str) -> str:
            return _validate_ban(v, "買方統一編號")

        @field_validator("timestamp")
        @classmethod
        def _check_timestamp(cls, v: str) -> str:
            return _validate_iso8601(v, "修正時間")
    else:
        class Config:
            extra = "forbid"

        @validator("original_pred")
        def _check_original_pred(cls, v):  # noqa: N805
            return _validate_account_code(v, "母版預測科目代碼")

        @validator("corrected_to")
        def _check_corrected_to(cls, v):  # noqa: N805
            return _validate_account_code(v, "修正後科目代碼")

        @validator("confidence_weight")
        def _check_confidence_weight(cls, v):  # noqa: N805
            return _validate_unit_interval(v, "confidence_weight")

        @validator("seller_ban")
        def _check_seller_ban(cls, v):  # noqa: N805
            return _validate_ban(v, "賣方統一編號")

        @validator("buyer_ban")
        def _check_buyer_ban(cls, v):  # noqa: N805
            return _validate_ban(v, "買方統一編號")

        @validator("timestamp")
        def _check_timestamp(cls, v):  # noqa: N805
            return _validate_iso8601(v, "修正時間")


# ---------------------------------------------------------------------------
# 4. TaxComplianceRule — 稅法合規規則
# ---------------------------------------------------------------------------

class TaxComplianceRule(BaseModel):
    """
    稅法合規規則（規則引擎使用）。

    重要提醒：
        本規則庫僅提供常見情境之風險提示，不涵蓋所有稅務情境，
        亦不構成稅務或法律意見。preferred_account 可能為 None
        （代表本規則性質為否決/禁止性，不主動建議正面科目）。
    """

    rule_id: str = Field(..., min_length=1, description="規則識別碼")
    trigger_keywords: List[str] = Field(default_factory=list, description="觸發關鍵字清單")
    trigger_condition: Optional[str] = Field(default=None, description="觸發條件（如 CIF），可選")
    preferred_account: Optional[str] = Field(default=None, description="建議科目代碼（4 碼），可選")
    forbidden_accounts: List[str] = Field(default_factory=list, description="禁止科目代碼清單")
    weight: float = Field(..., description="規則權重（0-1）")
    penalty: float = Field(..., description="違反懲罰係數（0-1）")
    note: str = Field(..., min_length=1, description="法規依據說明")
    law_reference: Optional[str] = Field(default=None, description="法規來源條文")
    effective_date: Optional[str] = Field(default=None, description="規則版本／生效日期")
    scope: Optional[str] = Field(default=None, description="適用範圍")
    manual_review_required: Optional[str] = Field(default=None, description="建議人工覆核之條件說明")
    resolution_note: Optional[str] = Field(
        default=None, description="規則整合時的判斷說明（若該規則存在資料來源分歧）"
    )

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")

        @field_validator("preferred_account")
        @classmethod
        def _check_preferred_account(cls, v: Optional[str]) -> Optional[str]:
            if v is None:
                return v
            return _validate_account_code(v, "建議科目代碼")

        @field_validator("forbidden_accounts")
        @classmethod
        def _check_forbidden_accounts(cls, v: List[str]) -> List[str]:
            for code in v:
                _validate_account_code(code, "禁止科目代碼")
            return v

        @field_validator("weight")
        @classmethod
        def _check_weight(cls, v: float) -> float:
            return _validate_unit_interval(v, "weight")

        @field_validator("penalty")
        @classmethod
        def _check_penalty(cls, v: float) -> float:
            return _validate_unit_interval(v, "penalty")
    else:
        class Config:
            extra = "allow"

        @validator("preferred_account")
        def _check_preferred_account(cls, v):  # noqa: N805
            if v is None:
                return v
            return _validate_account_code(v, "建議科目代碼")

        @validator("forbidden_accounts", each_item=True)
        def _check_forbidden_accounts(cls, v):  # noqa: N805
            return _validate_account_code(v, "禁止科目代碼")

        @validator("weight")
        def _check_weight(cls, v):  # noqa: N805
            return _validate_unit_interval(v, "weight")

        @validator("penalty")
        def _check_penalty(cls, v):  # noqa: N805
            return _validate_unit_interval(v, "penalty")


# ---------------------------------------------------------------------------
# 5. CompanyProfile — 公司設定檔
# ---------------------------------------------------------------------------

class CompanyProfile(BaseModel):
    """
    公司設定檔

    用途：
        描述單一客戶公司的個人化設定，包含合規檢查權重、行業別
        （用於風險評估）、修正紀錄檔案路徑等，供 fusion_engine 與
        tax_compliance 模組讀取。

    自訂會計科目對照表擴充點：
        系統內建的科目代碼（如 5121/1268 等）僅為《商業會計科目表》
        常見科目之預設值，並非所有公司唯一標準。custom_account_mapping
        允許企業提供「系統科目代碼 -> 企業自有科目代碼/名稱」的對照，
        由 tax_compliance / fusion_engine 於輸出前套用轉換，
        不需修改系統內建規則庫本身。
    """

    company_tax_id: str = Field(..., description="公司統一編號（10 碼數字）")
    compliance_weight: float = Field(
        default=0.9,
        description="合規檢查權重（0-1），數值越高代表稅法合規檢查在最終融合分數中的比重越高。"
                     "此為系統風險控管參數，並非用於判斷使用者意圖。",
    )
    industry_type: str = Field(..., min_length=1, description="行業別（用於風險評估，如「進出口貿易」「一般中小企業」）")
    corrections_file: str = Field(..., min_length=1, description="修正紀錄檔案路徑（JSON）")
    last_updated: str = Field(..., description="最後更新時間（ISO 8601）")
    custom_account_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="企業自訂會計科目對照表（系統科目代碼 -> 企業自有科目代碼），可為空",
    )

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")

        @field_validator("company_tax_id")
        @classmethod
        def _check_company_tax_id(cls, v: str) -> str:
            return _validate_ban(v, "公司統一編號")

        @field_validator("compliance_weight")
        @classmethod
        def _check_weight(cls, v: float) -> float:
            return _validate_unit_interval(v, "compliance_weight")

        @field_validator("last_updated")
        @classmethod
        def _check_last_updated(cls, v: str) -> str:
            return _validate_iso8601(v, "最後更新時間")
    else:
        class Config:
            extra = "forbid"

        @validator("company_tax_id")
        def _check_company_tax_id(cls, v):  # noqa: N805
            return _validate_ban(v, "公司統一編號")

        @validator("compliance_weight")
        def _check_weight(cls, v):  # noqa: N805
            return _validate_unit_interval(v, "compliance_weight")

        @validator("last_updated")
        def _check_last_updated(cls, v):  # noqa: N805
            return _validate_iso8601(v, "最後更新時間")


# ---------------------------------------------------------------------------
# 範例資料（每個結構至少 1 筆，符合台灣稅法情境；統編為虛構範例值）
# ---------------------------------------------------------------------------

def build_examples() -> Dict[str, BaseModel]:
    """回傳每個結構的範例資料，供測試 / 文件使用（皆為虛構範例資料）。"""

    example_invoice = InvoiceData(
        invoice_id="EX-0001",
        buyer_ban="1234567890",
        seller_ban="0987654321",
        summary="進口原物料一批，CIF 運費及保險費已含於貨價",
        amount=105000.0,
        trade_condition=TradeCondition.CIF,
        invoice_date="2026-08-15",
    )

    example_prediction = AccountPrediction(
        account_code="5121",
        account_name="進貨",
        confidence=0.87,
        risk_level=RiskLevel.LOW,
    )

    example_correction = CorrectionRecord(
        invoice_id="EX-0002",
        timestamp="2026-08-20T10:30:00+08:00",
        summary="辦公室員工尾牙聚餐費用",
        summary_vector=[0.012, -0.034, 0.256, 0.101],
        original_pred="5213",
        corrected_to="5226",
        confidence_weight=0.95,
        seller_ban="0987654321",
        buyer_ban="1234567890",
    )

    example_rule = TaxComplianceRule(
        rule_id="rule_002_cif_freight_no_duplicate",
        trigger_keywords=["運費", "保險費"],
        trigger_condition="CIF",
        preferred_account="5122",
        forbidden_accounts=["5221", "5222"],
        weight=0.8,
        penalty=0.8,
        note="CIF 條件下運費及保險費已含在進貨價格中，不可再列報。",
    )

    example_profile = CompanyProfile(
        company_tax_id="1234567890",
        compliance_weight=0.9,
        industry_type="進出口貿易",
        corrections_file="data/corrections/company_1234567890_corrections.json",
        last_updated="2026-08-31T09:00:00+08:00",
        custom_account_mapping={},
    )

    return {
        "InvoiceData": example_invoice,
        "AccountPrediction": example_prediction,
        "CorrectionRecord": example_correction,
        "TaxComplianceRule": example_rule,
        "CompanyProfile": example_profile,
    }


if __name__ == "__main__":
    import json

    examples = build_examples()
    for name, obj in examples.items():
        print(f"\n===== {name} 範例 =====")
        if PYDANTIC_V2:
            print(obj.model_dump_json(indent=2, ensure_ascii=False))
        else:
            print(json.dumps(obj.dict(), indent=2, ensure_ascii=False))
