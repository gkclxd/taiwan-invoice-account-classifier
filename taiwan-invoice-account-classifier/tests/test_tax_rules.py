# -*- coding: utf-8 -*-
"""
台灣稅法合規規則庫 — 測試案例（整合自 M4-1 test_tax_rules.py）

改寫重點：
    - import 路徑改為 invoice_classifier.tax_compliance.TaxRuleEngine.match_rules
    - rule_id 前綴改為配合合併後 config/tax_rules.json 的實際命名
      （採 M5-1 dict 結構之大寫 RULE_00X 前綴，見任務決策 #1）
    - 規則 2 的建議科目已於合併時修正為 5122（進貨費用），因此
      TC03 的 expected_preferred_account 由原 M4-1 版本相同的 5122 保留。

涵蓋六條規則各至少一個正向命中案例、CIF/FOB 對照案例、多規則同時觸發案例，
以及完全不命中的控制案例。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytest

from invoice_classifier.tax_compliance import TaxRuleEngine

RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "tax_rules.json"


@pytest.fixture(scope="module")
def engine() -> TaxRuleEngine:
    return TaxRuleEngine(RULES_PATH)


@dataclass
class TaxRuleTestCase:
    case_id: str
    summary: str
    trade_condition: Optional[str]
    expected_rule_prefixes: List[str]
    expected_preferred_account: Optional[str]
    description: str


TEST_CASES: List[TaxRuleTestCase] = [
    TaxRuleTestCase(
        case_id="TC01",
        summary="進貨一批，內含 5% 營業稅",
        trade_condition=None,
        expected_rule_prefixes=["RULE_001"],
        expected_preferred_account="1268",
        description="含稅率字樣，應觸發進項稅額分離規則，建議科目 1268。",
    ),
    TaxRuleTestCase(
        case_id="TC02",
        summary="顧問服務費，VAT 另計",
        trade_condition=None,
        expected_rule_prefixes=["RULE_001"],
        expected_preferred_account="1268",
        description="含 VAT 字樣，應觸發進項稅額分離規則。",
    ),
    TaxRuleTestCase(
        case_id="TC03",
        summary="進口機械設備，CIF 已含運費及保險費",
        trade_condition="CIF",
        expected_rule_prefixes=["RULE_002"],
        expected_preferred_account="5122",
        description="CIF 條件 + 運費/保險費關鍵字，應觸發 CIF 運費不可重複列報規則，建議科目 5122。",
    ),
    TaxRuleTestCase(
        case_id="TC04",
        summary="進口原料，運費另計",
        trade_condition="FOB",
        expected_rule_prefixes=[],
        expected_preferred_account=None,
        description="FOB 條件下含運費字樣，不應觸發 CIF 規則（對照組，驗證 trigger_condition 判斷正確）。",
    ),
    TaxRuleTestCase(
        case_id="TC05",
        summary="進口貨物報關費",
        trade_condition=None,
        expected_rule_prefixes=["RULE_003"],
        expected_preferred_account="5122",
        description="含報關費關鍵字，應觸發進口報關費併入進貨成本規則。",
    ),
    TaxRuleTestCase(
        case_id="TC06",
        summary="Customs clearance fee for shipment",
        trade_condition=None,
        expected_rule_prefixes=["RULE_003"],
        expected_preferred_account="5122",
        description="英文 Customs 關鍵字亦應命中報關費規則（中英混合情境）。",
    ),
    TaxRuleTestCase(
        case_id="TC07",
        summary="客戶交際應酬餐費",
        trade_condition=None,
        expected_rule_prefixes=["RULE_004"],
        expected_preferred_account="5213",
        description="含交際、應酬關鍵字，應觸發交際費上限規則。",
    ),
    TaxRuleTestCase(
        case_id="TC08",
        summary="年節送禮客戶禮盒一批",
        trade_condition=None,
        expected_rule_prefixes=["RULE_004"],
        expected_preferred_account="5213",
        description="含送禮關鍵字，應觸發交際費上限規則。",
    ),
    TaxRuleTestCase(
        case_id="TC09",
        summary="員工尾牙餐費",
        trade_condition=None,
        expected_rule_prefixes=["RULE_005"],
        expected_preferred_account="5226",
        description="含員工、尾牙關鍵字，應觸發員工福利規則，建議科目 5226 而非 5213。",
    ),
    TaxRuleTestCase(
        case_id="TC10",
        summary="部門員工聚餐",
        trade_condition=None,
        expected_rule_prefixes=["RULE_005"],
        expected_preferred_account="5226",
        description="含員工、聚餐關鍵字，應觸發員工福利規則。",
    ),
    TaxRuleTestCase(
        case_id="TC11",
        summary="購買自用乘人小客車一台",
        trade_condition=None,
        expected_rule_prefixes=["RULE_006"],
        expected_preferred_account=None,
        description="含乘人、自用車關鍵字，應觸發自用小客車進項稅額不得扣抵規則，1268 為禁止科目，"
                    "preferred_account 為 None（本規則性質為否決/禁止性，見 resolution_note）。",
    ),
    TaxRuleTestCase(
        case_id="TC12",
        summary="辦公室文具用品採購",
        trade_condition=None,
        expected_rule_prefixes=[],
        expected_preferred_account=None,
        description="控制案例：一般辦公用品採購，不應命中任何稅法合規規則。",
    ),
    TaxRuleTestCase(
        case_id="TC13",
        summary="購買自用乘人小客車，含 5% 營業稅",
        trade_condition=None,
        expected_rule_prefixes=["RULE_001", "RULE_006"],
        expected_preferred_account=None,
        description="多規則同時觸發案例：同時含稅率字樣與自用車字樣，"
                    "應同時觸發 RULE_001 與 RULE_006，1268 因兩規則皆列為相關/禁止科目。",
    ),
]


def _rule_prefix(rule_id: str) -> str:
    return "_".join(rule_id.split("_")[:2])


@pytest.mark.parametrize("tc", TEST_CASES, ids=[tc.case_id for tc in TEST_CASES])
def test_match_rules(engine: TaxRuleEngine, tc: TaxRuleTestCase) -> None:
    matched = engine.match_rules(tc.summary, trade_condition=tc.trade_condition)
    matched_prefixes = sorted(_rule_prefix(r["rule_id"]) for r in matched)
    assert matched_prefixes == sorted(tc.expected_rule_prefixes), (
        f"{tc.case_id} 失敗：預期觸發 {tc.expected_rule_prefixes}，實際觸發 {matched_prefixes}"
        f"（{tc.description}）"
    )


def test_rule_2_and_6_resolution_notes_present(engine: TaxRuleEngine) -> None:
    """驗證規則 2、規則 6 因資料來源分歧，已加註 resolution_note 說明整合判斷。"""
    rules = engine.all_rules_dict()
    rule2 = next(r for rid, r in rules.items() if rid.startswith("RULE_002"))
    rule6 = next(r for rid, r in rules.items() if rid.startswith("RULE_006"))
    assert "resolution_note" in rule2 and rule2["resolution_note"]
    assert "resolution_note" in rule6 and rule6["resolution_note"]
    assert rule2["preferred_account"] == "5122"
    assert rule6["preferred_account"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
