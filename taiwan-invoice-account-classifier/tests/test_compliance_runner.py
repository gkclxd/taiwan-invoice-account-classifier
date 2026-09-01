# -*- coding: utf-8 -*-
"""
稅法合規測試執行函式（整合自 M8-1 test_runner.py，改名避免與 M8-2 版本衝突）

約束條件：
    - 25 個案例總執行時間需 < 10 秒
    - 全自動化，無需人工介入
    - 失敗時提供詳細錯誤訊息（含預期值/實際值/差異說明）

已知缺陷追蹤：
    原始 M5-1 fusion_engine.py 的 compute_compliance_scores() 對
    trigger_condition="AMOUNT_CHECK" 的判斷邏輯有誤（誤把 AMOUNT_CHECK
    當成需比對的 trade_condition 值，導致規則 4 永遠無法觸發）。
    本專案整合時已於 invoice_classifier/fusion_engine.py 修正此邏輯
    （僅 "CIF" 需比對 trade_condition，其餘條件值一律視為關鍵字命中即觸發），
    因此 TC012、TC013、TC014、TC024 於本專案應全數 PASS，
    不再是「預期內失敗」的已知缺陷案例。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from test_compliance_cases import TEST_CASES, ComplianceTestCase, rule_coverage_map


@dataclass
class TestResult:
    test_id: str
    description: str
    passed: bool
    elapsed_seconds: float
    errors: List[str] = field(default_factory=list)
    actual_output: Optional[Dict[str, Any]] = None


@dataclass
class TestReport:
    total: int
    passed: int
    failed: int
    total_elapsed_seconds: float
    results: List[TestResult] = field(default_factory=list)
    within_time_limit: bool = True

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    def summary_text(self) -> str:
        return "\n".join([
            f"測試總數：{self.total}", f"通過：{self.passed}", f"失敗：{self.failed}",
            f"通過率：{self.pass_rate:.1%}",
            f"總執行時間：{self.total_elapsed_seconds:.4f} 秒"
            f"（{'符合' if self.within_time_limit else '不符合'} <10秒限制）",
        ])

    def print_detailed(self) -> None:
        print("=" * 70)
        print("稅法合規測試報告")
        print("=" * 70)
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.test_id}: {r.description} ({r.elapsed_seconds*1000:.2f} ms)")
            if not r.passed:
                for err in r.errors:
                    print(f"         - {err}")
        print("-" * 70)
        print(self.summary_text())
        print("=" * 70)


def _check_case(tc: ComplianceTestCase, output: Dict[str, Any]) -> List[str]:
    """比對單一案例的預期輸出與實際輸出，回傳錯誤訊息清單（空清單代表通過）。"""
    errors: List[str] = []
    expected = tc.expected_output

    exp_account = expected.get("predicted_account")
    if exp_account is not None:
        actual_account = output.get("predicted_account")
        if actual_account != exp_account:
            errors.append(f"predicted_account 不符：預期 {exp_account!r}，實際 {actual_account!r}")

    exp_risk = expected.get("risk_level")
    if exp_risk is not None:
        actual_risk = output.get("risk_level")
        if actual_risk != exp_risk:
            errors.append(f"risk_level 不符：預期 {exp_risk!r}，實際 {actual_risk!r}")

    exp_keywords = expected.get("compliance_notes_contains", [])
    if exp_keywords:
        notes_joined = " | ".join(output.get("compliance_notes", []))
        missing = [kw for kw in exp_keywords if kw not in notes_joined]
        if missing:
            errors.append(f"compliance_notes 缺少預期關鍵字：{missing}（實際 notes：{output.get('compliance_notes', [])}）")

    exp_forbidden = expected.get("forbidden_account_penalized")
    if exp_forbidden is not None:
        actual_forbidden = output.get("forbidden_accounts_penalized", [])
        if exp_forbidden not in actual_forbidden:
            errors.append(f"預期科目 {exp_forbidden!r} 應被合規規則扣分，實際被扣分科目清單：{actual_forbidden}")

    exp_not_in = expected.get("predicted_account_not_in")
    if exp_not_in:
        actual_account = output.get("predicted_account")
        if actual_account in exp_not_in:
            errors.append(f"predicted_account 不應為禁止科目之一：實際為 {actual_account!r}，禁止清單：{exp_not_in}")

    exp_amount_warning = expected.get("amount_cap_warning")
    if exp_amount_warning is not None:
        actual_warning = output.get("amount_cap_warning", False)
        if bool(actual_warning) != bool(exp_amount_warning):
            errors.append(f"amount_cap_warning 不符：預期 {exp_amount_warning}，實際 {actual_warning}")

    return errors


def run_compliance_tests(test_cases: List[ComplianceTestCase], classifier: Any, verbose: bool = True) -> TestReport:
    results: List[TestResult] = []
    suite_start = time.perf_counter()

    for tc in test_cases:
        case_start = time.perf_counter()
        try:
            output = classifier.predict(tc.input)
            errors = _check_case(tc, output)
            passed = len(errors) == 0
        except Exception as exc:  # noqa: BLE001
            output = {}
            errors = [f"執行時發生例外（Exception）：{type(exc).__name__}: {exc}"]
            passed = False
        case_elapsed = time.perf_counter() - case_start

        results.append(TestResult(
            test_id=tc.test_id, description=tc.description, passed=passed,
            elapsed_seconds=case_elapsed, errors=errors, actual_output=output,
        ))

    total_elapsed = time.perf_counter() - suite_start
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count

    report = TestReport(
        total=len(results), passed=passed_count, failed=failed_count,
        total_elapsed_seconds=total_elapsed, results=results,
        within_time_limit=total_elapsed < 10.0,
    )
    if verbose:
        report.print_detailed()
    return report


if __name__ == "__main__":
    from compliance_classifier_stub import ComplianceTestClassifier

    classifier = ComplianceTestClassifier()
    report = run_compliance_tests(TEST_CASES, classifier, verbose=True)
    sys.exit(0 if report.failed == 0 else 1)
