# -*- coding: utf-8 -*-
"""
pytest 相容介面（整合自 M8-2 test_boundary_pytest.py）

執行方式：
    python -m pytest tests/test_boundary_pytest.py -v
"""

from __future__ import annotations

import time

import pytest

from test_boundary_cases import BOUNDARY_TEST_CASES
from boundary_classifier_stub import BoundaryTestClassifier
from test_boundary_runner import (
    _build_correction_prototypes_for_case,
    _verify_result,
    run_boundary_tests,
    run_stress_test,
)


@pytest.fixture(scope="module")
def classifier() -> BoundaryTestClassifier:
    return BoundaryTestClassifier()


@pytest.mark.parametrize(
    "tc",
    BOUNDARY_TEST_CASES,
    ids=[tc["test_id"] for tc in BOUNDARY_TEST_CASES],
)
def test_boundary_case(tc, classifier: BoundaryTestClassifier):
    """每個邊界案例都必須：(1) 不拋出未捕捉例外 (2) 回傳結果符合預期。"""
    correction_protos = _build_correction_prototypes_for_case(tc)
    classifier.correction_prototypes = correction_protos

    try:
        output = classifier.predict(tc["input"])
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{tc['test_id']} 導致未捕捉例外（系統崩潰）：{type(exc).__name__}: {exc}")

    failure_reason = _verify_result(tc, output)
    assert failure_reason is None, f"{tc['test_id']} 失敗：{failure_reason}"


def test_stress_10000_corrections_under_2_seconds():
    """任務要求：10,000 筆修正紀錄情境下，單筆預測時間 < 2 秒，且不崩潰。"""
    result = run_stress_test(n_records=10_000)
    assert result["predict_error"] is False, "大量修正紀錄情境下預測發生錯誤"
    assert result["predict_under_2s"] is True, (
        f"預測耗時 {result['predict_elapsed_seconds']} 秒，超過 2 秒門檻"
    )


def test_full_suite_completes_under_30_seconds():
    """任務約束：28 個案例總執行時間 < 30 秒。"""
    t0 = time.perf_counter()
    report = run_boundary_tests()
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"測試套件執行時間 {elapsed:.2f} 秒，超過 30 秒上限"
    assert report.no_crashes, "測試套件偵測到系統崩潰（CRASHED 案例數 > 0）"
