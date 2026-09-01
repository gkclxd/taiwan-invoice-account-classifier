# -*- coding: utf-8 -*-
"""
CorrectionManager 測試案例（整合自 M5-2 test_correction_manager.py）

改寫重點：
    - import 路徑改為 invoice_classifier.storage
    - 統編改用 10 碼數字（配合 data_models / storage 驗證規則）

涵蓋：
    1. 新增修正紀錄（正常情況）與自動生成欄位
    2. 重複 invoice_id 預設拒絕 / allow_duplicate=True 允許
    3. 查詢：invoice_id / corrected_to / seller_ban / date_range（含多條件 AND）
    4. 更新：confidence_weight 與 corrected_to，並確認 timestamp 有更新
    5. 刪除：需 confirm=True，且賣方偏好統計同步遞減
    6. 匯出：JSON 與 CSV
    7. 檔案格式錯誤應拋出 StorageError
    8. 輸入驗證錯誤
    9. 備份機制：寫入後產生備份檔，且超過 max_backups 會被裁剪
    10. 併發寫入：多執行緒同時 add_correction 不應遺失任何一筆紀錄
    11. 效能：單次操作 < 0.5 秒
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

from invoice_classifier.storage import (
    CorrectionManager,
    ConfirmationRequiredError,
    RecordNotFoundError,
    StorageError,
    ValidationError,
)

COMPANY = "1234567890"  # 測試用統編（10 碼數字，不代表真實公司）
SELLER_A = "2233445566"
SELLER_B = "9988776655"
BUYER = "1122334455"


class CorrectionManagerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="correction_manager_test_")
        self.manager = CorrectionManager(data_dir=self.tmpdir, max_backups=3)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestAddCorrection(CorrectionManagerTestBase):
    def test_add_correction_normal(self):
        record = self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="INV-0001", summary="辦公用品採購",
            original_pred="5101", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
        )
        self.assertEqual(record.invoice_id, "INV-0001")
        self.assertEqual(record.corrected_to, "5216")
        self.assertEqual(record.confidence_weight, 1.0)
        self.assertTrue(len(record.summary_vector) > 0)
        self.assertTrue(record.timestamp)

    def test_auto_generated_fields_reproducible(self):
        record = self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="INV-0002", summary="員工國內差旅費",
            original_pred="5213", corrected_to="5226", seller_ban=SELLER_A, buyer_ban=BUYER,
        )
        record2_vector = self.manager.vectorizer.encode("員工國內差旅費")
        self.assertEqual(record.summary_vector, record2_vector)
        self.assertAlmostEqual(record.confidence_weight, 1.0)

    def test_duplicate_invoice_id_rejected_by_default(self):
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="INV-DUP", summary="測試重複",
            original_pred="5101", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
        )
        with self.assertRaises(ValidationError):
            self.manager.add_correction(
                company_tax_id=COMPANY, invoice_id="INV-DUP", summary="測試重複二次",
                original_pred="5101", corrected_to="5217", seller_ban=SELLER_A, buyer_ban=BUYER,
            )

    def test_duplicate_allowed_when_flag_set(self):
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="INV-DUP2", summary="測試重複",
            original_pred="5101", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
        )
        record2 = self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="INV-DUP2", summary="測試重複二次",
            original_pred="5101", corrected_to="5217", seller_ban=SELLER_A, buyer_ban=BUYER,
            allow_duplicate=True,
        )
        self.assertEqual(record2.corrected_to, "5217")
        results = self.manager.query_corrections(COMPANY, invoice_id="INV-DUP2")
        self.assertEqual(len(results), 2)

    def test_validation_errors(self):
        with self.assertRaises(ValidationError):
            self.manager.add_correction(
                company_tax_id="bad_ban", invoice_id="X", summary="s",
                original_pred="5101", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
            )
        with self.assertRaises(ValidationError):
            self.manager.add_correction(
                company_tax_id=COMPANY, invoice_id="X", summary="s",
                original_pred="51", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
            )
        with self.assertRaises(ValidationError):
            self.manager.add_correction(
                company_tax_id=COMPANY, invoice_id="X", summary="s",
                original_pred="5101", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
                confidence_weight=1.5,
            )
        with self.assertRaises(ValidationError):
            self.manager.add_correction(
                company_tax_id=COMPANY, invoice_id="", summary="s",
                original_pred="5101", corrected_to="5216", seller_ban=SELLER_A, buyer_ban=BUYER,
            )


class TestQueryCorrections(CorrectionManagerTestBase):
    def setUp(self):
        super().setUp()
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="Q-1", summary="進貨-原物料",
            original_pred="5101", corrected_to="5121", seller_ban=SELLER_A, buyer_ban=BUYER,
        )
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="Q-2", summary="交際應酬餐費",
            original_pred="5213", corrected_to="5213", seller_ban=SELLER_B, buyer_ban=BUYER,
        )
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="Q-3", summary="員工尾牙聚餐",
            original_pred="5213", corrected_to="5226", seller_ban=SELLER_B, buyer_ban=BUYER,
        )

    def test_query_by_invoice_id(self):
        results = self.manager.query_corrections(COMPANY, invoice_id="Q-2")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].corrected_to, "5213")

    def test_query_by_multiple_conditions(self):
        results = self.manager.query_corrections(COMPANY, seller_ban=SELLER_B, corrected_to="5226")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].invoice_id, "Q-3")

        results_none = self.manager.query_corrections(COMPANY, seller_ban=SELLER_A, corrected_to="5226")
        self.assertEqual(len(results_none), 0)

    def test_query_empty_company_returns_empty_list(self):
        results = self.manager.query_corrections("0000000000")
        self.assertEqual(results, [])

    def test_query_by_date_range(self):
        all_records = self.manager.query_corrections(COMPANY)
        latest_ts = max(r.timestamp for r in all_records)
        results = self.manager.query_corrections(COMPANY, date_from=latest_ts)
        self.assertGreaterEqual(len(results), 1)
        results_future_only = self.manager.query_corrections(COMPANY, date_from="2999-01-01")
        self.assertEqual(results_future_only, [])


class TestUpdateCorrection(CorrectionManagerTestBase):
    def setUp(self):
        super().setUp()
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="U-1", summary="運費",
            original_pred="5101", corrected_to="5121", seller_ban=SELLER_A, buyer_ban=BUYER,
            confidence_weight=0.5,
        )

    def test_update_confidence_and_corrected_to(self):
        before = self.manager.query_corrections(COMPANY, invoice_id="U-1")[0]
        updated = self.manager.update_correction(
            COMPANY, invoice_id="U-1", confidence_weight=0.9, corrected_to="5122",
        )
        self.assertEqual(updated.confidence_weight, 0.9)
        self.assertEqual(updated.corrected_to, "5122")
        self.assertNotEqual(updated.timestamp, before.timestamp)

    def test_update_not_found_raises(self):
        with self.assertRaises(RecordNotFoundError):
            self.manager.update_correction(COMPANY, invoice_id="NO-SUCH-ID", confidence_weight=0.8)

    def test_update_requires_at_least_one_field(self):
        with self.assertRaises(ValidationError):
            self.manager.update_correction(COMPANY, invoice_id="U-1")


class TestDeleteCorrection(CorrectionManagerTestBase):
    def setUp(self):
        super().setUp()
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="D-1", summary="測試刪除",
            original_pred="5101", corrected_to="5121", seller_ban=SELLER_A, buyer_ban=BUYER,
        )

    def test_delete_without_confirm_raises(self):
        with self.assertRaises(ConfirmationRequiredError):
            self.manager.delete_correction(COMPANY, invoice_id="D-1")
        self.assertEqual(len(self.manager.query_corrections(COMPANY, invoice_id="D-1")), 1)

    def test_delete_with_confirm_succeeds_and_updates_seller_prefs(self):
        removed_count = self.manager.delete_correction(COMPANY, invoice_id="D-1", confirm=True)
        self.assertEqual(removed_count, 1)
        self.assertEqual(self.manager.query_corrections(COMPANY, invoice_id="D-1"), [])

        store = self.manager._store_for(COMPANY)
        payload = store.load()
        seller_stat = payload["seller_preferences"].get(SELLER_A, {})
        self.assertEqual(seller_stat.get("5121", 0), 0)

    def test_delete_not_found_raises(self):
        with self.assertRaises(RecordNotFoundError):
            self.manager.delete_correction(COMPANY, invoice_id="NO-SUCH-ID", confirm=True)


class TestExportCorrections(CorrectionManagerTestBase):
    def setUp(self):
        super().setUp()
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="E-1", summary="出口報關費",
            original_pred="5101", corrected_to="5122", seller_ban=SELLER_A, buyer_ban=BUYER,
        )

    def test_export_json_and_csv(self):
        json_path = Path(self.tmpdir) / "export.json"
        csv_path = Path(self.tmpdir) / "export.csv"

        out_json = self.manager.export_corrections(COMPANY, str(json_path), fmt="json")
        out_csv = self.manager.export_corrections(COMPANY, str(csv_path), fmt="csv")

        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["invoice_id"], "E-1")
        self.assertIn("summary_vector", data[0])

        csv_content = Path(out_csv).read_text(encoding="utf-8-sig")
        self.assertIn("E-1", csv_content)
        self.assertIn("invoice_id", csv_content.splitlines()[0])

    def test_export_invalid_format_raises(self):
        with self.assertRaises(ValidationError):
            self.manager.export_corrections(COMPANY, str(Path(self.tmpdir) / "x.xml"), fmt="xml")


class TestStorageRobustness(CorrectionManagerTestBase):
    def test_corrupt_json_file_raises_storage_error(self):
        store = self.manager._store_for(COMPANY)
        store.file_path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(StorageError):
            self.manager.query_corrections(COMPANY)

    def test_empty_file_treated_as_no_records(self):
        store = self.manager._store_for(COMPANY)
        store.file_path.write_text("", encoding="utf-8")
        results = self.manager.query_corrections(COMPANY)
        self.assertEqual(results, [])

    def test_backup_created_and_pruned(self):
        for i in range(5):
            self.manager.add_correction(
                company_tax_id=COMPANY, invoice_id=f"B-{i}", summary=f"備份測試{i}",
                original_pred="5101", corrected_to="5121", seller_ban=SELLER_A, buyer_ban=BUYER,
            )
        store = self.manager._store_for(COMPANY)
        backups = list(store.backup_dir.glob(f"company_{COMPANY}_corrections.*.bak.json"))
        self.assertLessEqual(len(backups), 3)  # max_backups=3
        self.assertGreater(len(backups), 0)


class TestConcurrency(CorrectionManagerTestBase):
    def test_concurrent_add_correction_no_data_loss(self):
        n_threads = 8
        errors = []

        def worker(i):
            try:
                self.manager.add_correction(
                    company_tax_id=COMPANY, invoice_id=f"CONC-{i}", summary=f"併發測試-{i}",
                    original_pred="5101", corrected_to="5121", seller_ban=SELLER_A, buyer_ban=BUYER,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        results = self.manager.query_corrections(COMPANY)
        self.assertEqual(len(results), n_threads)
        ids = {r.invoice_id for r in results}
        self.assertEqual(ids, {f"CONC-{i}" for i in range(n_threads)})


class TestPerformance(CorrectionManagerTestBase):
    def test_single_operation_under_500ms(self):
        start = time.perf_counter()
        self.manager.add_correction(
            company_tax_id=COMPANY, invoice_id="PERF-1", summary="效能測試",
            original_pred="5101", corrected_to="5121", seller_ban=SELLER_A, buyer_ban=BUYER,
        )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.5)


if __name__ == "__main__":
    unittest.main()
