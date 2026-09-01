# -*- coding: utf-8 -*-
"""
月度報告生成器（MonthlyReportGenerator）測試案例（整合自 M7-2 test_monthly_report.py）

改寫重點：
    - import 路徑改為 invoice_classifier.report
    - rule_id 改為配合合併後 config/tax_rules.json 的命名（RULE_00X 大寫前綴）
    - anti_tax_evasion_weight 參數改名為 compliance_weight
    - DEFAULT_ACCOUNT_SUBJECTS 改用 config/account_subjects.csv 實際載入結果

測試案例清單：
    1. test_basic_report_generation        —— 基本情境：正常發票、修正、合規觸發皆有資料
    2. test_empty_month_report             —— 邊界情境：當月無任何資料
    3. test_high_risk_and_compliance_focus —— 高風險發票 + 合規規則觸發之報告內容正確性
    4. test_missing_fields_data_quality    —— 資料缺漏（無 confidence / risk_level 異常值）之容錯
    5. test_previous_month_comparison      —— 與上月科目分佈比較
    6. test_performance_1000_invoices      —— 效能測試：1000 筆發票 < 10 秒
"""

from __future__ import annotations

import random
import time
import unittest
from datetime import datetime, timedelta
from typing import Optional

from invoice_classifier.report import MonthlyReportGenerator


def _mock_invoice(
    invoice_id: str, date: str, summary: str, predicted_account: str, confidence: float,
    risk_level: str = "low", risk_reason: Optional[str] = None,
) -> dict:
    return {
        "invoice_id": invoice_id, "date": date, "summary": summary,
        "predicted_account": predicted_account, "confidence": confidence,
        "risk_level": risk_level, "risk_reason": risk_reason,
    }


class TestMonthlyReportGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = MonthlyReportGenerator()

    def test_basic_report_generation(self) -> None:
        invoices = [
            _mock_invoice("INV-0001", "2026-08-03", "辦公室文具用品採購", "5216", 0.92, "low"),
            _mock_invoice("INV-0002", "2026-08-05", "原物料進貨一批", "5121", 0.88, "low"),
            _mock_invoice("INV-0003", "2026-08-08", "客戶交際應酬餐費", "5213", 0.75, "medium"),
            _mock_invoice("INV-0004", "2026-08-10", "員工尾牙聚餐費用", "5226", 0.81, "low"),
            _mock_invoice(
                "INV-0005", "2026-08-12", "進口貨物 CIF 條件下運費", "5221", 0.55, "high",
                "CIF 條件下運費可能與進貨成本重複列報",
            ),
        ]
        corrections = [{
            "invoice_id": "INV-0004", "original_pred": "5213", "corrected_to": "5226",
            "predicted_at": "2026-08-10T09:00:00+08:00", "timestamp": "2026-08-10T15:30:00+08:00",
        }]
        violations = [{"rule_id": "RULE_002_cif_freight_no_duplicate", "invoice_id": "INV-0005"}]

        report = self.generator.generate_report(
            invoices_processed=invoices, corrections_made=corrections,
            compliance_violations=violations, company_name="測試股份有限公司A",
            report_month="2026-08", compliance_weight=0.5,
        )

        for heading in [
            "一、執行摘要", "二、會計科目分佈", "三、稅務風險分析",
            "四、合規規則觸發統計", "五、修正紀錄分析", "六、建議行動",
        ]:
            self.assertIn(heading, report)

        self.assertIn("測試股份有限公司A", report)
        self.assertIn("2026-08", report)
        self.assertIn("當月處理發票總數**：5 張", report)
        self.assertIn("INV-0005", report)
        self.assertIn("營利事業所得稅查核準則", report)  # 法規依據（第44條）
        self.assertIn("修正總數**：1 筆", report)
        self.assertIn("5226", report)

    def test_empty_month_report(self) -> None:
        report = self.generator.generate_report(
            invoices_processed=[], corrections_made=[], compliance_violations=[],
            company_name="無資料測試公司B", report_month="2026-07",
        )
        self.assertIn("無資料測試公司B", report)
        self.assertIn("本月無發票資料", report)
        self.assertIn("本月無人工修正紀錄", report)
        self.assertIn("本月未觸發任何稅法合規規則", report)
        self.assertIn("六、建議行動", report)

    def test_high_risk_and_compliance_focus(self) -> None:
        invoices = [
            _mock_invoice(
                "INV-1001", "2026-08-15", "營業稅 5% 進項稅額", "5121", 0.3, "critical",
                "進項稅額誤併入進貨成本，未單獨列帳",
            ),
            _mock_invoice(
                "INV-1002", "2026-08-16", "自用轎車進項稅額扣抵", "1268", 0.4, "critical",
                "自用乘人小汽車進項稅額不得扣抵",
            ),
            _mock_invoice("INV-1003", "2026-08-17", "報關費用支出", "5216", 0.6, "medium"),
            _mock_invoice(
                "INV-1004", "2026-08-18", "交際應酬送禮費用", "5213", 0.7, "high",
                "交際費可能超過列支上限",
            ),
        ]
        violations = [
            {"rule_id": "RULE_001_input_tax_separation", "invoice_id": "INV-1001"},
            {"rule_id": "RULE_001_input_tax_separation", "invoice_id": "INV-1001"},
            {"rule_id": "RULE_006_passenger_car_input_tax_not_deductible", "invoice_id": "INV-1002"},
            {"rule_id": "RULE_003_customs_fee_into_cost", "invoice_id": "INV-1003"},
            {"rule_id": "RULE_004_entertainment_expense_cap", "invoice_id": "INV-1004"},
        ]

        report = self.generator.generate_report(
            invoices_processed=invoices, corrections_made=[], compliance_violations=violations,
            company_name="高風險測試公司C", report_month="2026-08",
        )

        idx_1001 = report.index("INV-1001")
        idx_1004 = report.index("INV-1004")
        self.assertLess(idx_1001, idx_1004, "critical 風險發票應排在 high 風險發票之前")

        self.assertIn("最常觸發規則 Top 5", report)
        self.assertIn("加值型及非加值型營業稅法", report)
        self.assertIn("極高風險", report)

    def test_missing_fields_data_quality(self) -> None:
        invoices = [
            {"invoice_id": "INV-9001", "date": "2026-08-01", "summary": "無置信度發票", "predicted_account": "5121", "risk_level": "low"},
            {"invoice_id": "INV-9002", "date": "2026-08-02", "summary": "風險等級異常發票", "predicted_account": "5122", "confidence": 0.5, "risk_level": "unknown_level"},
            {"invoice_id": "INV-9003", "date": "2026-08-03", "summary": "缺科目發票", "confidence": 0.7, "risk_level": "medium"},
        ]
        report = self.generator.generate_report(
            invoices_processed=invoices, corrections_made=[], compliance_violations=[],
            company_name="容錯測試公司D",
        )
        self.assertIn("當月處理發票總數**：3 張", report)
        self.assertIn("資料品質提醒", report)
        self.assertIn("INV-9001", report)
        self.assertIn("INV-9002", report)
        self.assertIn("INV-9003", report)

    def test_previous_month_comparison(self) -> None:
        invoices = [
            _mock_invoice("INV-2001", "2026-08-01", "進貨", "5121", 0.9, "low"),
            _mock_invoice("INV-2002", "2026-08-02", "進貨", "5121", 0.9, "low"),
            _mock_invoice("INV-2003", "2026-08-03", "辦公費", "5216", 0.9, "low"),
        ]
        previous = {"5121": 5, "5213": 3}

        report = self.generator.generate_report(
            invoices_processed=invoices, corrections_made=[], compliance_violations=[],
            company_name="比較測試公司E", previous_month_distribution=previous,
        )
        self.assertIn("與上月比較", report)
        self.assertIn("本月新增使用科目", report)
        self.assertIn("5216", report)
        self.assertIn("本月未再使用", report)
        self.assertIn("5213", report)

    def test_performance_1000_invoices(self) -> None:
        random.seed(42)
        account_codes = list(self.generator.account_subjects.keys()) or ["5121", "5122", "5213"]
        risk_levels = ["low", "medium", "high", "critical"]
        base_date = datetime(2026, 8, 1)

        invoices = []
        for i in range(1000):
            invoices.append(_mock_invoice(
                invoice_id=f"INV-PERF-{i:05d}",
                date=(base_date + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S"),
                summary=f"模擬發票摘要 {i}",
                predicted_account=random.choice(account_codes),
                confidence=round(random.uniform(0.3, 0.99), 2),
                risk_level=random.choice(risk_levels),
            ))

        corrections = [
            {
                "invoice_id": f"INV-PERF-{i:05d}", "original_pred": "5213", "corrected_to": "5226",
                "predicted_at": "2026-08-01T09:00:00", "timestamp": "2026-08-01T09:30:00",
            }
            for i in range(0, 1000, 20)
        ]
        rule_ids = list(self.generator.rule_notes.keys()) or ["RULE_001_input_tax_separation"]
        violations = [
            {"rule_id": random.choice(rule_ids), "invoice_id": f"INV-PERF-{i:05d}"}
            for i in range(0, 1000, 10)
        ]

        start = time.perf_counter()
        report = self.generator.generate_report(
            invoices_processed=invoices, corrections_made=corrections,
            compliance_violations=violations, company_name="效能測試公司F",
        )
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 10.0, f"報告生成耗時 {elapsed:.2f} 秒，超過 10 秒效能目標")
        self.assertIn("當月處理發票總數**：1000 張", report)
        self.assertNotIn("⚠️ **效能提醒**", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
