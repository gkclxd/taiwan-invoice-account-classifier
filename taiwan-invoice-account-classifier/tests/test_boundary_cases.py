# -*- coding: utf-8 -*-
"""
邊界情況測試案例庫（整合自 M8-2 test_cases.py，改名避免與 M8-1 版本衝突）

涵蓋 8 大邊界情境，共 28 個案例：
    1. 空輸入（summary / seller_ban / buyer_ban 為空）        BC001-BC004
    2. 異常格式（統編非 10 碼、amount 負數或 0、trade_condition 未知值）
                                                              BC005-BC010
    3. 極端金額（1 億元 / 1 元 / None）                        BC011-BC013
    4. 特殊字元（emoji、HTML 標籤、SQL 注入字串）              BC014-BC016
    5. 無修正紀錄（客戶首次使用）                              BC017
    6. 單一修正紀錄                                            BC018
    7. 衝突規則（同時觸發多條規則，建議不同科目）              BC019-BC021
    8. 大量修正紀錄（10,000 筆，分類器層級案例）               BC022

    額外補充：型別誤用、None 輸入、超長字串、非 dict 輸入、缺少必填欄位
    （BC023-BC028）。

改寫重點（相較原始 M8-2 test_cases.py）：
    - rule_id 相關敘述改為大寫 RULE_00X 命名（對應合併後的
      config/tax_rules.json，見任務決策 #1 與 #3）。
    - BC019（自用小客車 vs 進項稅額分離規則）與 BC021（交際費 vs 職工福利）
      的案例設計本身即驗證本專案 config/tax_rules.json 中 RULE_002 / RULE_006
      的整合判斷版本（見 resolution_note）。
    - 基準發票已使用本專案 InvoiceData 驗證通過之 10 碼合法統編。
"""

from __future__ import annotations

from typing import Any, Dict, List

_VALID_BASE: Dict[str, Any] = {
    "buyer_ban": "1234567890",
    "seller_ban": "0987654321",
    "summary": "採購辦公用品一批",
    "amount": 5000.0,
    "trade_condition": None,
    "invoice_date": "2026-08-15",
}


def _base(**overrides: Any) -> Dict[str, Any]:
    d = dict(_VALID_BASE)
    d.update(overrides)
    return d


BOUNDARY_TEST_CASES: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # 1. 空輸入
    # ------------------------------------------------------------------
    {
        "test_id": "BC001",
        "category": "空輸入",
        "description": "summary 為空字串",
        "input": _base(summary=""),
        "expected_behavior": "回傳結構化驗證錯誤（error=True），不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "error_type", "error_message"],
    },
    {
        "test_id": "BC002",
        "category": "空輸入",
        "description": "seller_ban 為空字串",
        "input": _base(seller_ban=""),
        "expected_behavior": "回傳結構化驗證錯誤（error=True），不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "error_type", "field_errors"],
    },
    {
        "test_id": "BC003",
        "category": "空輸入",
        "description": "buyer_ban 為空字串",
        "input": _base(buyer_ban=""),
        "expected_behavior": "回傳結構化驗證錯誤（error=True），不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "error_type", "field_errors"],
    },
    {
        "test_id": "BC004",
        "category": "空輸入",
        "description": "buyer_ban 與 seller_ban 皆為 None",
        "input": _base(buyer_ban=None, seller_ban=None),
        "expected_behavior": "回傳結構化驗證錯誤，field_errors 含兩個欄位，不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },

    # ------------------------------------------------------------------
    # 2. 異常格式
    # ------------------------------------------------------------------
    {
        "test_id": "BC005",
        "category": "異常格式",
        "description": "統編非 10 碼數字（少於 10 碼）",
        "input": _base(buyer_ban="12345"),
        "expected_behavior": "回傳驗證錯誤，明確指出 buyer_ban 格式錯誤",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },
    {
        "test_id": "BC006",
        "category": "異常格式",
        "description": "統編含英文字母（非純數字）",
        "input": _base(seller_ban="AB345678CD"),
        "expected_behavior": "回傳驗證錯誤，明確指出 seller_ban 格式錯誤",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },
    {
        "test_id": "BC007",
        "category": "異常格式",
        "description": "amount 為負數",
        "input": _base(amount=-1000.0),
        "expected_behavior": "回傳驗證錯誤，明確指出 amount 不可為負數",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },
    {
        "test_id": "BC008",
        "category": "異常格式",
        "description": "amount 為 0（合法邊界值，非負數但屬極端情況）",
        "input": _base(amount=0.0),
        "expected_behavior": "視為合法輸入正常處理（0 元發票如折讓單常見），不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC009",
        "category": "異常格式",
        "description": "trade_condition 為未知值（EXW，非 FOB/CIF）",
        "input": _base(trade_condition="EXW"),
        "expected_behavior": "回傳驗證錯誤，明確指出 trade_condition 僅接受 FOB/CIF",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },
    {
        "test_id": "BC010",
        "category": "異常格式",
        "description": "invoice_date 非 ISO 8601 格式（如 2026/08/31）",
        "input": _base(invoice_date="2026/08/31"),
        "expected_behavior": "回傳驗證錯誤，明確指出日期格式錯誤",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },

    # ------------------------------------------------------------------
    # 3. 極端金額
    # ------------------------------------------------------------------
    {
        "test_id": "BC011",
        "category": "極端金額",
        "description": "amount 極大（1 億元）",
        "input": _base(summary="進口大型機器設備一批", amount=100_000_000.0),
        "expected_behavior": "正常處理並回傳預測結果，不崩潰、無數值溢位",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC012",
        "category": "極端金額",
        "description": "amount 極小（1 元）",
        "input": _base(summary="文具用品採購", amount=1.0),
        "expected_behavior": "正常處理並回傳預測結果，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC013",
        "category": "極端金額",
        "description": "amount 未提供（None，欄位為可選）",
        "input": _base(amount=None),
        "expected_behavior": "正常處理（amount 為可選欄位），不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },

    # ------------------------------------------------------------------
    # 4. 特殊字元
    # ------------------------------------------------------------------
    {
        "test_id": "BC014",
        "category": "特殊字元",
        "description": "summary 含 emoji",
        "input": _base(summary="辦公室聚餐🍱🎉尾牙活動"),
        "expected_behavior": "正常處理，不因多位元組字元造成例外，可正確判斷職工福利科目",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC015",
        "category": "特殊字元",
        "description": "summary 含 HTML 標籤（潛在 XSS 注入嘗試）",
        "input": _base(summary="<script>alert('xss')</script>採購辦公用品"),
        "expected_behavior": "正常處理為純文字摘要，不執行任何嵌入內容，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC016",
        "category": "特殊字元",
        "description": "summary 含 SQL 注入字串",
        "input": _base(summary="'; DROP TABLE invoices; --"),
        "expected_behavior": "正常處理為純文字摘要，不觸發任何資料庫操作，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },

    # ------------------------------------------------------------------
    # 5 & 6. 修正紀錄數量邊界（無 / 單一 / 大量）
    # ------------------------------------------------------------------
    {
        "test_id": "BC017",
        "category": "無修正紀錄",
        "description": "客戶首次使用，無任何修正紀錄（correction_prototypes 為空 dict）",
        "input": _base(summary="進貨原物料一批"),
        "expected_behavior": "僅使用母版模型（及合規規則）預測，不因缺少修正紀錄而崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
        "correction_prototypes": {},
    },
    {
        "test_id": "BC018",
        "category": "單一修正紀錄",
        "description": "客戶僅有 1 筆修正紀錄可用於個人化學習",
        "input": _base(summary="進貨原物料一批"),
        "expected_behavior": "正常套用該筆修正紀錄的原型向量，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
        "correction_prototypes": "SINGLE",
    },

    # ------------------------------------------------------------------
    # 7. 衝突規則
    # ------------------------------------------------------------------
    {
        "test_id": "BC019",
        "category": "衝突規則",
        "description": "同時觸發規則1（進項稅額分離→建議1268）與規則6（自用小客車→禁止1268）",
        "input": _base(summary="自用乘人小汽車採購含5%營業稅", amount=1_500_000.0),
        "expected_behavior": "取最嚴格規則（1268 遭雙重限制／禁止），risk_level 應為 high 或 critical，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level", "compliance_notes"],
        "expect_risk_level_in": ["high", "critical"],
    },
    {
        "test_id": "BC020",
        "category": "衝突規則",
        "description": "同時觸發規則2（CIF運費不可重複→建議5122/禁止5221,5222）與規則3（報關費→建議5122）",
        "input": _base(
            summary="進口貨物CIF條件下之報關費與運費",
            amount=80_000.0,
            trade_condition="CIF",
        ),
        "expected_behavior": "兩條規則建議科目一致（5122）可疊加加分，取最嚴格處理不衝突項，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level", "compliance_notes"],
    },
    {
        "test_id": "BC021",
        "category": "衝突規則",
        "description": "同時觸發規則4（交際費→建議5213）與規則5（員工聚餐→建議5226/禁止5213），科目互斥",
        "input": _base(summary="員工尾牙聚餐交際應酬費用", amount=25_000.0),
        "expected_behavior": "5213 同時被建議又被規則5禁止，系統應取最嚴格規則（禁止優先），不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level", "compliance_notes"],
    },

    # ------------------------------------------------------------------
    # 8. 大量修正紀錄（分類器層級案例；10,000 筆壓力測試見 test_boundary_runner.py）
    # ------------------------------------------------------------------
    {
        "test_id": "BC022",
        "category": "大量修正紀錄",
        "description": "客戶有大量（模擬 10,000 筆彙總後）修正原型向量可用",
        "input": _base(summary="進貨原物料一批"),
        "expected_behavior": "預測時間 < 2 秒，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
        "correction_prototypes": "BULK",
        "max_elapsed_seconds": 2.0,
    },

    # ------------------------------------------------------------------
    # 額外邊界案例（型別誤用 / None 輸入 / 超長字串 / 非 dict）
    # ------------------------------------------------------------------
    {
        "test_id": "BC023",
        "category": "型別誤用",
        "description": "整體輸入為 None（呼叫端傳遞錯誤）",
        "input": None,
        "expected_behavior": "回傳結構化驗證錯誤，不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "error_message"],
    },
    {
        "test_id": "BC024",
        "category": "型別誤用",
        "description": "整體輸入為字串而非 dict",
        "input": "this is not a dict",
        "expected_behavior": "回傳結構化驗證錯誤，不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "error_message"],
    },
    {
        "test_id": "BC025",
        "category": "型別誤用",
        "description": "amount 為字串型別（如 \"5000\"）",
        "input": _base(amount="5000"),
        "expected_behavior": "Pydantic 嘗試型別轉換成功，正常處理，不崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC026",
        "category": "型別誤用",
        "description": "amount 為無法轉換的字串（如 \"abc\"）",
        "input": _base(amount="abc"),
        "expected_behavior": "回傳驗證錯誤，明確指出 amount 必須為數字，不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },
    {
        "test_id": "BC027",
        "category": "超長輸入",
        "description": "summary 為超長字串（約 10,000 字元）",
        "input": _base(summary="進貨原物料採購" * 1250),
        "expected_behavior": "正常處理，不因超長字串造成效能異常或崩潰",
        "should_not_crash": True,
        "expect_error": False,
        "expect_fields": ["predicted_account", "confidence", "risk_level"],
    },
    {
        "test_id": "BC028",
        "category": "缺少必填欄位",
        "description": "缺少 invoice_date 欄位（完全未提供該 key）",
        "input": {
            "buyer_ban": "1234567890",
            "seller_ban": "0987654321",
            "summary": "採購辦公用品一批",
            "amount": 3000.0,
        },
        "expected_behavior": "回傳驗證錯誤，明確指出 invoice_date 為必填，不崩潰",
        "should_not_crash": True,
        "expect_error": True,
        "expect_fields": ["error", "field_errors"],
    },
]


def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    return [tc for tc in BOUNDARY_TEST_CASES if tc["category"] == category]


def summarize_categories() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tc in BOUNDARY_TEST_CASES:
        counts[tc["category"]] = counts.get(tc["category"], 0) + 1
    return counts


__all__ = ["BOUNDARY_TEST_CASES", "get_test_cases_by_category", "summarize_categories"]


if __name__ == "__main__":
    print(f"測試案例總數：{len(BOUNDARY_TEST_CASES)}")
    for cat, count in summarize_categories().items():
        print(f"  {cat}: {count} 案例")
