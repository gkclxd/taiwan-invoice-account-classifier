# -*- coding: utf-8 -*-
"""
稅法合規測試案例庫（整合自 M8-1 test_cases.py，25 個正向測試案例）

改寫重點：
    - rule_refs 中的 rule_id 前綴改為大寫 RULE_00X（配合合併後 config/tax_rules.json）
    - 其餘案例邏輯、預期輸出與原始 M8-1 版本完全一致（規則 2 的建議科目本就是
      5122/5221/5222，規則 6 的 preferred_account 本就不斷言特定科目，
      與本專案合併後的 tax_rules.json 判斷一致，無需調整期望值）

說明：
    測試「分類器最終預測結果」是否正確反映規則的加分／懲罰／禁止效果
    （predicted_account、risk_level、compliance_notes），驗證 Fusion
    Engine 融合後的可觀察行為，並涵蓋金額檢查（交際費上限）等專屬邏輯。
    與 test_tax_rules.py 的差異：test_tax_rules.py 測試「規則比對」本身
    （match_rules 命中哪些規則），本檔案測試「分類器最終預測結果」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComplianceTestCase:
    test_id: str
    description: str
    input: Dict[str, Any]
    expected_output: Dict[str, Any]
    rule_refs: List[str] = field(default_factory=list)


def _invoice(
    summary: str, seller_ban: str = "0987654321", buyer_ban: str = "1234567890",
    amount: Optional[float] = None, trade_condition: Optional[str] = None,
    invoice_date: str = "2026-08-31",
) -> Dict[str, Any]:
    return {
        "summary": summary, "seller_ban": seller_ban, "buyer_ban": buyer_ban,
        "amount": amount, "trade_condition": trade_condition, "invoice_date": invoice_date,
    }


TEST_CASES: List[ComplianceTestCase] = [
    # --- 規則 1：進項稅額分離 --------------------------------------------
    ComplianceTestCase(
        test_id="TC001",
        description="摘要含「營業稅 5%」，應建議獨立列帳科目 1268（進項稅額）",
        input=_invoice("進貨一批，內含營業稅 5%，貨款另計"),
        expected_output={"predicted_account": "1268", "risk_level": None,
                          "compliance_notes_contains": ["進項稅額", "1268"]},
        rule_refs=["營業稅法第15條", "營業稅法第19條", "RULE_001_input_tax_separation"],
    ),
    ComplianceTestCase(
        test_id="TC002",
        description="摘要含「VAT」英文字樣，亦應觸發進項稅額分離規則",
        input=_invoice("顧問服務費用，VAT 另計"),
        expected_output={"predicted_account": "1268", "risk_level": None,
                          "compliance_notes_contains": ["進項稅額"]},
        rule_refs=["營業稅法第15條", "營業稅法第19條", "RULE_001_input_tax_separation"],
    ),
    ComplianceTestCase(
        test_id="TC003",
        description="若稅額被誤併入進貨科目（5121）列帳，應被扣分並標記為禁止科目",
        input=_invoice("進貨含稅金額一批，內含 5% 營業稅"),
        expected_output={"predicted_account": "1268", "risk_level": None,
                          "compliance_notes_contains": [], "forbidden_account_penalized": "5121"},
        rule_refs=["營業稅法第15條", "營業稅法第19條", "RULE_001_input_tax_separation"],
    ),
    ComplianceTestCase(
        test_id="TC004",
        description="控制案例：摘要僅含一般稅務字眼但無明確稅額字樣，仍應命中（含「稅」字即觸發）",
        input=_invoice("代收代付稅金一筆"),
        expected_output={"predicted_account": "1268", "risk_level": None,
                          "compliance_notes_contains": ["進項稅額"]},
        rule_refs=["營業稅法第15條", "RULE_001_input_tax_separation"],
    ),

    # --- 規則 2：CIF 條件下運費不可重複列報 -------------------------------
    ComplianceTestCase(
        test_id="TC005",
        description="trade_condition=CIF 且摘要含「運費」，應建議 5122（進貨費用），而非 5221（運費）",
        input=_invoice("進口原物料一批，運費及保險費已含於貨價", trade_condition="CIF"),
        expected_output={"predicted_account": "5122", "risk_level": None,
                          "compliance_notes_contains": ["CIF", "運費"],
                          "forbidden_account_penalized": "5221"},
        rule_refs=["營所稅查核準則第44條", "RULE_002_cif_freight_no_duplicate"],
    ),
    ComplianceTestCase(
        test_id="TC006",
        description="trade_condition=CIF 且摘要含「保險費」，同樣不可重複列報",
        input=_invoice("進口設備一批，內含保險費", trade_condition="CIF"),
        expected_output={"predicted_account": "5122", "risk_level": None,
                          "compliance_notes_contains": ["CIF"],
                          "forbidden_account_penalized": "5222"},
        rule_refs=["營所稅查核準則第44條", "RULE_002_cif_freight_no_duplicate"],
    ),
    ComplianceTestCase(
        test_id="TC007",
        description="對照組：trade_condition=FOB 且摘要含「運費」，FOB 條件下運費可單獨列報 5221，不應觸發 CIF 規則",
        input=_invoice("進口原料，運費另計", trade_condition="FOB"),
        expected_output={"predicted_account": "5221", "risk_level": "low", "compliance_notes_contains": []},
        rule_refs=["營所稅查核準則第44條（對照組，不觸發）"],
    ),
    ComplianceTestCase(
        test_id="TC008",
        description="對照組：未標示貿易條件（None）且摘要含「運費」，不應觸發 CIF 規則",
        input=_invoice("國內廠商運費一筆", trade_condition=None),
        expected_output={"predicted_account": None, "risk_level": "low", "compliance_notes_contains": []},
        rule_refs=["營所稅查核準則第44條（對照組，不觸發）"],
    ),

    # --- 規則 3：進口報關費應併入進貨成本 ----------------------------------
    ComplianceTestCase(
        test_id="TC009",
        description="摘要含「報關費」，應建議 5122（進貨費用），而非 5221（運費）",
        input=_invoice("進口貨物報關費一筆"),
        expected_output={"predicted_account": "5122", "risk_level": None,
                          "compliance_notes_contains": ["報關費", "進貨成本"]},
        rule_refs=["營所稅查核準則第37條", "RULE_003_customs_fee_into_cost"],
    ),
    ComplianceTestCase(
        test_id="TC010",
        description="摘要含「通關費」，同樣應併入進貨成本",
        input=_invoice("貨物通關費用一批"),
        expected_output={"predicted_account": "5122", "risk_level": None,
                          "compliance_notes_contains": ["進貨成本"]},
        rule_refs=["營所稅查核準則第37條", "RULE_003_customs_fee_into_cost"],
    ),
    ComplianceTestCase(
        test_id="TC011",
        description="摘要含英文「Customs」關鍵字，亦應命中報關費規則（中英混合情境）",
        input=_invoice("Customs clearance fee for import shipment"),
        expected_output={"predicted_account": "5122", "risk_level": None, "compliance_notes_contains": []},
        rule_refs=["營所稅查核準則第37條", "RULE_003_customs_fee_into_cost"],
    ),

    # --- 規則 4：交際費列支上限 --------------------------------------------
    ComplianceTestCase(
        test_id="TC012",
        description="摘要含「交際費」，金額超過假設之進貨淨額 2‰ 上限，應警告可能超過列支上限",
        input=_invoice("客戶交際應酬餐費", amount=500000.0),
        expected_output={"predicted_account": "5213", "risk_level": "medium",
                          "compliance_notes_contains": ["交際", "上限"], "amount_cap_warning": True},
        rule_refs=["營所稅查核準則第62條", "RULE_004_entertainment_expense_cap"],
    ),
    ComplianceTestCase(
        test_id="TC013",
        description="摘要含「送禮」關鍵字，亦應觸發交際費上限規則",
        input=_invoice("年節送禮客戶禮盒一批", amount=30000.0),
        expected_output={"predicted_account": "5213", "risk_level": "medium",
                          "compliance_notes_contains": ["交際"]},
        rule_refs=["營所稅查核準則第62條", "RULE_004_entertainment_expense_cap"],
    ),
    ComplianceTestCase(
        test_id="TC014",
        description="摘要含「宴客」關鍵字且金額較小（未超過上限），仍應標記交際費規則但風險較低",
        input=_invoice("業務宴客餐費", amount=2000.0),
        expected_output={"predicted_account": "5213", "risk_level": "medium",
                          "compliance_notes_contains": ["交際"]},
        rule_refs=["營所稅查核準則第62條", "RULE_004_entertainment_expense_cap"],
    ),

    # --- 規則 5：員工福利 vs 交際費 -----------------------------------------
    ComplianceTestCase(
        test_id="TC015",
        description="摘要含「員工旅遊」，應建議 5226（職工福利），而非 5213（交際費）",
        input=_invoice("員工旅遊活動費用"),
        expected_output={"predicted_account": "5226", "risk_level": None,
                          "compliance_notes_contains": ["職工福利"], "forbidden_account_penalized": "5213"},
        rule_refs=["營所稅查核準則第81條", "RULE_005_employee_welfare_vs_entertainment"],
    ),
    ComplianceTestCase(
        test_id="TC016",
        description="摘要含「尾牙」，應建議 5226（職工福利）",
        input=_invoice("公司年度尾牙聚餐"),
        expected_output={"predicted_account": "5226", "risk_level": None,
                          "compliance_notes_contains": ["職工福利"]},
        rule_refs=["營所稅查核準則第81條", "RULE_005_employee_welfare_vs_entertainment"],
    ),
    ComplianceTestCase(
        test_id="TC017",
        description="摘要含「員工」+「聚餐」，應建議 5226 而非 5213",
        input=_invoice("部門員工聚餐"),
        expected_output={"predicted_account": "5226", "risk_level": None,
                          "compliance_notes_contains": ["職工福利"], "forbidden_account_penalized": "5213"},
        rule_refs=["營所稅查核準則第81條", "RULE_005_employee_welfare_vs_entertainment"],
    ),

    # --- 規則 6：自用乘人小汽車進項稅額不得扣抵 -----------------------------
    ComplianceTestCase(
        test_id="TC018",
        description="摘要含「自用汽車」，1268（進項稅額）應為禁止科目，不得扣抵",
        input=_invoice("購買自用汽車一台", amount=1200000.0),
        expected_output={"predicted_account": None, "predicted_account_not_in": ["1268"],
                          "risk_level": "high", "compliance_notes_contains": ["不得扣抵"],
                          "forbidden_account_penalized": "1268"},
        rule_refs=["營業稅法第19條", "RULE_006_passenger_car_input_tax_not_deductible"],
    ),
    ComplianceTestCase(
        test_id="TC019",
        description="摘要含「轎車」關鍵字，亦應觸發不得扣抵規則",
        input=_invoice("購入自用乘人轎車一輛"),
        expected_output={"predicted_account": None, "predicted_account_not_in": ["1268"],
                          "risk_level": "high", "compliance_notes_contains": ["不得扣抵"],
                          "forbidden_account_penalized": "1268"},
        rule_refs=["營業稅法第19條", "RULE_006_passenger_car_input_tax_not_deductible"],
    ),
    ComplianceTestCase(
        test_id="TC020",
        description="摘要同時含「自用車」與「營業稅 5%」，應同時觸發規則1與規則6，risk_level 應升級為 critical",
        input=_invoice("購買自用車一台，內含營業稅 5%"),
        expected_output={"predicted_account": None, "predicted_account_not_in": ["1268"],
                          "risk_level": "critical", "compliance_notes_contains": ["不得扣抵", "進項稅額"],
                          "forbidden_account_penalized": "1268"},
        rule_refs=["營業稅法第15條", "營業稅法第19條", "RULE_001_input_tax_separation",
                   "RULE_006_passenger_car_input_tax_not_deductible"],
    ),

    # --- 額外案例：控制組與跨規則整合案例 -----------------------------------
    ComplianceTestCase(
        test_id="TC021",
        description="控制案例：一般辦公用品採購，不應命中任何稅法合規規則，風險等級應為 low",
        input=_invoice("辦公室文具用品採購"),
        expected_output={"predicted_account": None, "risk_level": "low", "compliance_notes_contains": []},
        rule_refs=["無（控制組）"],
    ),
    ComplianceTestCase(
        test_id="TC022",
        description="控制案例：一般差旅住宿費用，不應命中任何稅法合規規則",
        input=_invoice("國內出差住宿費用"),
        expected_output={"predicted_account": None, "risk_level": "low", "compliance_notes_contains": []},
        rule_refs=["無（控制組）"],
    ),
    ComplianceTestCase(
        test_id="TC023",
        description="跨規則案例：CIF 條件下同時含報關費與運費，應優先套用報關費（5122）規則且不重複列報運費",
        input=_invoice("進口貨物報關費及運費一批", trade_condition="CIF"),
        expected_output={"predicted_account": "5122", "risk_level": None,
                          "compliance_notes_contains": ["CIF"], "forbidden_account_penalized": "5221"},
        rule_refs=["營所稅查核準則第37條", "營所稅查核準則第44條",
                   "RULE_002_cif_freight_no_duplicate", "RULE_003_customs_fee_into_cost"],
    ),
    ComplianceTestCase(
        test_id="TC024",
        description="極端案例：金額為 0 的交際費發票，仍應標記交際費規則但不應誤判為超過上限",
        input=_invoice("交際應酬費用", amount=0.0),
        expected_output={"predicted_account": "5213", "risk_level": "medium",
                          "compliance_notes_contains": ["交際"]},
        rule_refs=["營所稅查核準則第62條", "RULE_004_entertainment_expense_cap"],
    ),
    ComplianceTestCase(
        test_id="TC025",
        description="邊界案例：買方統編與賣方統編相同（自開發票情境），仍應正確套用進項稅額分離規則",
        input=_invoice("內部調撥含營業稅 5%", seller_ban="1234567890", buyer_ban="1234567890"),
        expected_output={"predicted_account": "1268", "risk_level": None,
                          "compliance_notes_contains": ["進項稅額"]},
        rule_refs=["營業稅法第15條", "RULE_001_input_tax_separation"],
    ),
]


def get_test_case(test_id: str) -> ComplianceTestCase:
    for tc in TEST_CASES:
        if tc.test_id == test_id:
            return tc
    raise KeyError(f"找不到測試案例：{test_id}")


def rule_coverage_map() -> Dict[str, List[str]]:
    """回傳每條規則 ID 對應到哪些測試案例（依 rule_refs 中含 'RULE_' 前綴的項目統計）。"""
    coverage: Dict[str, List[str]] = {}
    for tc in TEST_CASES:
        for ref in tc.rule_refs:
            if ref.startswith("RULE_"):
                coverage.setdefault(ref, []).append(tc.test_id)
    return coverage


if __name__ == "__main__":
    print(f"共 {len(TEST_CASES)} 個測試案例")
    for rule_id, ids in rule_coverage_map().items():
        print(f"  {rule_id}: {len(ids)} 個案例 -> {ids}")
