# -*- coding: utf-8 -*-
"""
pytest 相容介面（整合自 M8-1 test_compliance_pytest.py）

將 test_compliance_cases.TEST_CASES 以 pytest.mark.parametrize 形式暴露，
核心驗證邏輯與 test_compliance_runner._check_case() 共用。
"""

from __future__ import annotations

import pytest

from test_compliance_cases import TEST_CASES, ComplianceTestCase
from test_compliance_runner import _check_case
from compliance_classifier_stub import ComplianceTestClassifier


@pytest.fixture(scope="module")
def classifier():
    return ComplianceTestClassifier()


@pytest.mark.parametrize("tc", TEST_CASES, ids=[tc.test_id for tc in TEST_CASES])
def test_compliance_case(tc: ComplianceTestCase, classifier: ComplianceTestClassifier) -> None:
    output = classifier.predict(tc.input)
    errors = _check_case(tc, output)
    assert not errors, f"{tc.test_id} 失敗（{tc.description}）：\n" + "\n".join(f"  - {e}" for e in errors)
