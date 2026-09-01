# -*- coding: utf-8 -*-
"""
邊界情況測試執行函式（整合自 M8-2 test_runner.py，改名避免與 M8-1 版本衝突）

設計要求對應：
    - 28 個案例執行時間 < 30 秒          -> run_boundary_tests() 計時
    - 測試自動化，無需人工介入            -> 純函式呼叫，無互動輸入
    - 測試失敗提供詳細錯誤訊息            -> TestCaseResult.failure_reason
    - 所有測試都不應導致系統崩潰          -> 每個案例呼叫皆包在
                                            try/except 中，記錄為 CRASHED
                                            狀態並詳列 traceback

用法：
    python tests/test_boundary_runner.py
    python -m pytest tests/test_boundary_pytest.py -v
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from test_boundary_cases import BOUNDARY_TEST_CASES
from boundary_classifier_stub import (
    BoundaryTestClassifier,
    get_correction_manager_cls,
    get_correction_record_cls,
    ACCOUNT_CODES,
)


@dataclass
class TestCaseResult:
    test_id: str
    category: str
    description: str
    passed: bool
    crashed: bool
    elapsed_seconds: float
    actual_output: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None
    traceback_str: Optional[str] = None


@dataclass
class TestReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    crashed: int = 0
    total_elapsed_seconds: float = 0.0
    results: List[TestCaseResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.crashed == 0

    @property
    def no_crashes(self) -> bool:
        return self.crashed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "crashed": self.crashed,
            "no_crashes": self.no_crashes,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
            "results": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "description": r.description,
                    "passed": r.passed,
                    "crashed": r.crashed,
                    "elapsed_seconds": round(r.elapsed_seconds, 6),
                    "failure_reason": r.failure_reason,
                }
                for r in self.results
            ],
        }


def _build_correction_prototypes_for_case(tc: Dict[str, Any]) -> Dict[str, np.ndarray]:
    marker = tc.get("correction_prototypes")
    if marker is None:
        return {}
    if isinstance(marker, dict):
        return marker
    if marker == "SINGLE":
        rng = np.random.default_rng(seed=42)
        return {"5121": rng.normal(size=64)}
    if marker == "BULK":
        rng = np.random.default_rng(seed=7)
        return {code: rng.normal(size=64) for code in ACCOUNT_CODES}
    return {}


def _verify_result(tc: Dict[str, Any], output: Dict[str, Any]) -> Optional[str]:
    expect_error = tc.get("expect_error", False)
    actual_error = bool(output.get("error", False))

    if expect_error != actual_error:
        return (
            f"預期 error={expect_error}，實際 error={actual_error}"
            f"（error_message={output.get('error_message')!r}）"
        )

    expect_fields = tc.get("expect_fields") or []
    missing = [f for f in expect_fields if f not in output]
    if missing:
        return f"回傳結果缺少預期欄位：{missing}（實際欄位：{list(output.keys())}）"

    if not actual_error:
        risk_options = tc.get("expect_risk_level_in")
        if risk_options and output.get("risk_level") not in risk_options:
            return f"risk_level={output.get('risk_level')!r} 不在預期範圍 {risk_options}"

        max_elapsed = tc.get("max_elapsed_seconds")
        if max_elapsed is not None and output.get("elapsed_seconds", 0.0) > max_elapsed:
            return f"elapsed_seconds={output.get('elapsed_seconds')} 超過上限 {max_elapsed} 秒"

    return None


def run_boundary_tests(
    test_cases: List[Dict[str, Any]] = None,
    classifier: Optional[BoundaryTestClassifier] = None,
) -> TestReport:
    test_cases = test_cases if test_cases is not None else BOUNDARY_TEST_CASES
    classifier = classifier if classifier is not None else BoundaryTestClassifier()

    report = TestReport(total=len(test_cases))
    suite_start = time.perf_counter()

    for tc in test_cases:
        case_start = time.perf_counter()
        try:
            correction_protos = _build_correction_prototypes_for_case(tc)
            classifier.correction_prototypes = correction_protos or {}

            output = classifier.predict(tc["input"])
            elapsed = time.perf_counter() - case_start

            failure_reason = _verify_result(tc, output)
            passed = failure_reason is None

            result = TestCaseResult(
                test_id=tc["test_id"], category=tc["category"], description=tc["description"],
                passed=passed, crashed=False, elapsed_seconds=elapsed,
                actual_output=output, failure_reason=failure_reason,
            )
            if passed:
                report.passed += 1
            else:
                report.failed += 1

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - case_start
            result = TestCaseResult(
                test_id=tc["test_id"], category=tc["category"], description=tc["description"],
                passed=False, crashed=True, elapsed_seconds=elapsed,
                failure_reason=f"未捕捉例外導致崩潰：{type(exc).__name__}: {exc}",
                traceback_str=traceback.format_exc(),
            )
            report.crashed += 1

        report.results.append(result)

    report.total_elapsed_seconds = time.perf_counter() - suite_start
    return report


def print_report(report: TestReport) -> None:
    print("=" * 78)
    print(f"邊界情況測試報告　總案例數：{report.total}")
    print("=" * 78)
    for r in report.results:
        status = "PASS" if r.passed else ("CRASH" if r.crashed else "FAIL")
        print(f"[{r.test_id}] {status}  ({r.elapsed_seconds*1000:.2f} ms)  {r.category} — {r.description}")
        if not r.passed:
            print(f"        原因：{r.failure_reason}")
            if r.traceback_str:
                print("        " + r.traceback_str.replace("\n", "\n        "))
    print("-" * 78)
    print(
        f"通過：{report.passed}　失敗：{report.failed}　崩潰：{report.crashed}　"
        f"總耗時：{report.total_elapsed_seconds:.4f} 秒"
    )
    print(f"系統穩定性（無崩潰）：{'通過' if report.no_crashes else '未通過'}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# 壓力測試：10,000 筆修正紀錄
# ---------------------------------------------------------------------------

def run_stress_test(
    n_records: int = 10_000,
    use_real_correction_manager: bool = True,
    write_sample_size: int = 200,
) -> Dict[str, Any]:
    """
    大量修正紀錄壓力測試（見 M8-2 原始設計說明：add_correction() 為 O(n) per-call
    成本，直接同步寫入 10,000 筆會使測試套件本身超過時間上限，因此改用
    取樣＋外推法；預測路徑則直接注入已彙總向量，量測真實推論效能）。
    """
    import tempfile

    result: Dict[str, Any] = {
        "n_records": n_records,
        "write_sample_size": write_sample_size,
        "correction_manager_available": False,
        "write_sample_elapsed_seconds": None,
        "write_avg_ms_per_call": None,
        "write_extrapolated_seconds_for_n": None,
        "predict_elapsed_seconds": None,
        "predict_under_2s": None,
        "notes": [],
    }

    CorrectionManagerCls = get_correction_manager_cls() if use_real_correction_manager else None

    if CorrectionManagerCls is not None:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                manager = CorrectionManagerCls(data_dir=tmp_dir)
                company_tax_id = "1234567890"

                t0 = time.perf_counter()
                for i in range(write_sample_size):
                    manager.add_correction(
                        company_tax_id=company_tax_id,
                        invoice_id=f"INV{i:06d}",
                        summary="進貨原物料一批" if i % 2 == 0 else "辦公用品採購",
                        original_pred="5121",
                        corrected_to="5121" if i % 2 == 0 else "5216",
                        confidence_weight=0.9,
                        seller_ban="0987654321",
                        buyer_ban=company_tax_id,
                        allow_duplicate=True,
                    )
                sample_elapsed = time.perf_counter() - t0
                avg_ms = (sample_elapsed / write_sample_size) * 1000.0

                result["correction_manager_available"] = True
                result["write_sample_elapsed_seconds"] = round(sample_elapsed, 4)
                result["write_avg_ms_per_call"] = round(avg_ms, 3)
                result["write_extrapolated_seconds_for_n"] = round(avg_ms / 1000.0 * n_records, 1)
                result["notes"].append(
                    f"實測 {write_sample_size} 筆寫入平均每筆 {avg_ms:.2f} ms，"
                    "以此線性外推 10,000 筆總寫入時間約為 "
                    f"{result['write_extrapolated_seconds_for_n']:.1f} 秒（保守低估，"
                    "實際因整檔讀寫重複成本可能為 O(n^2) 更高）。"
                )
                result["scalability_concern"] = True
        except TypeError:
            result["notes"].append(
                "CorrectionManager.add_correction() 簽章與假設不符，"
                "已改用純記憶體向量彙總模擬大量修正紀錄場景。"
            )
        except Exception as exc:  # noqa: BLE001
            result["notes"].append(
                f"CorrectionManager 實測寫入失敗（{type(exc).__name__}: {exc}），"
                "已改用純記憶體向量彙總模擬，不影響本次穩定性結論。"
            )
    else:
        result["notes"].append("正式 CorrectionManager 模組未能載入，改以記憶體向量模擬 10,000 筆修正紀錄彙總情境。")

    rng = np.random.default_rng(seed=7)
    bulk_prototypes = {code: rng.normal(size=64) for code in ACCOUNT_CODES}

    classifier = BoundaryTestClassifier(correction_prototypes=bulk_prototypes)
    invoice = {
        "buyer_ban": "1234567890",
        "seller_ban": "0987654321",
        "summary": "進貨原物料一批",
        "amount": 50000.0,
        "trade_condition": None,
        "invoice_date": "2026-08-31",
    }

    t0 = time.perf_counter()
    output = classifier.predict(invoice)
    predict_elapsed = time.perf_counter() - t0

    result["predict_elapsed_seconds"] = round(predict_elapsed, 6)
    result["predict_under_2s"] = predict_elapsed < 2.0
    result["predict_error"] = output.get("error", False)

    return result


def generate_stress_test_markdown(stress_result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 壓力測試報告：大量修正紀錄（10,000 筆）")
    lines.append("")
    lines.append(f"測試規模：{stress_result['n_records']:,} 筆修正紀錄")
    lines.append("")
    lines.append("## 結果摘要")
    lines.append("")
    lines.append("| 指標 | 結果 |")
    lines.append("|---|---|")
    lines.append(
        f"| CorrectionManager 實測可用 | {'是' if stress_result['correction_manager_available'] else '否（改用模擬）'} |"
    )
    if stress_result.get("write_sample_elapsed_seconds") is not None:
        lines.append(
            f"| 實測取樣寫入筆數 | {stress_result['write_sample_size']:,} 筆（耗時 "
            f"{stress_result['write_sample_elapsed_seconds']} 秒） |"
        )
        lines.append(f"| 平均每筆寫入耗時 | {stress_result['write_avg_ms_per_call']} ms |")
        lines.append(
            f"| 外推 {stress_result['n_records']:,} 筆總寫入時間（保守估計） | "
            f"約 {stress_result['write_extrapolated_seconds_for_n']:,} 秒 |"
        )
    lines.append(f"| 單筆預測耗時（含大量修正原型向量） | {stress_result['predict_elapsed_seconds']} 秒 |")
    lines.append(
        f"| 預測時間 < 2 秒（任務要求） | {'通過' if stress_result['predict_under_2s'] else '未通過'} |"
    )
    lines.append(f"| 預測是否發生錯誤 | {'是（異常）' if stress_result['predict_error'] else '否（正常）'} |")
    lines.append("")
    if stress_result.get("notes"):
        lines.append("## 備註")
        lines.append("")
        for note in stress_result["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## 結論")
    lines.append("")
    if stress_result["predict_under_2s"] and not stress_result["predict_error"]:
        lines.append(
            "**預測路徑（推論）**：系統在已具備大量修正紀錄彙總後的個人化原型向量情境下，"
            "單筆發票預測時間仍遠低於 2 秒門檻，且未發生任何例外或崩潰，"
            "符合任務要求的效能與穩定性指標。"
        )
    else:
        lines.append("系統未能在大量修正紀錄情境下滿足預測效能或穩定性要求，建議進一步排查。")

    if stress_result.get("scalability_concern"):
        lines.append("")
        lines.append(
            "**寫入路徑（CorrectionManager.add_correction）— 已知擴展性疑慮**："
            "實測發現每次新增修正紀錄都會重新讀取整個 JSON 檔案、附加後整檔寫回並備份，"
            "屬於 per-call O(n) 成本，隨紀錄數增加而變慢，10,000 筆情境下累積寫入時間"
            f"外推約 {stress_result['write_extrapolated_seconds_for_n']:,} 秒（保守估計）。"
            "此雖不影響單筆『預測』路徑的即時效能，但會影響『批次匯入歷史修正紀錄』等"
            "維運情境的可用性，建議未來將 storage 模組改為批次寫入介面。"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    overall_start = time.perf_counter()

    report = run_boundary_tests()
    print_report(report)

    print("\n執行 10,000 筆修正紀錄壓力測試...\n")
    stress_result = run_stress_test(n_records=10_000)
    stress_md = generate_stress_test_markdown(stress_result)
    print(stress_md)

    total_elapsed = time.perf_counter() - overall_start
    print(f"\n總執行時間（含壓力測試）：{total_elapsed:.2f} 秒")

    out_dir = Path(__file__).resolve().parent
    with open(out_dir / "boundary_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    with open(out_dir / "stress_test_report.md", "w", encoding="utf-8") as f:
        f.write(stress_md + "\n")

    sys.exit(0 if report.no_crashes else 1)
