# -*- coding: utf-8 -*-
"""
測試 invoice_classifier.storage 中的 VersionControl / BackupManager / AuditLogger
（整合自 M6-2 test_version_control.py）

改寫重點：
    - import 路徑改為 invoice_classifier.storage

涵蓋：
    - 版本號自動遞增、保留數量上限
    - 版本比較（新增/刪除/更新）與 Markdown 差異報告
    - 還原到指定版本（含還原前自動備份）
    - 每日備份（同步模式驗證邏輯）、保留天數清理
    - 審計日誌記錄、查詢、匯出（jsonl/json/csv）
    - 效能：版本操作 < 1 秒
"""

import json
import shutil
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

from invoice_classifier.storage import (
    VersionControl,
    BackupManager,
    AuditLogger,
    VersionNotFoundError,
    BackupNotFoundError,
)

TAX_ID = "1234567890"


def make_payload(corrections):
    return {
        "company_tax_id": TAX_ID, "corrections": corrections,
        "seller_preferences": {}, "last_updated": "2026-08-31T00:00:00+00:00",
    }


def rec(invoice_id, corrected_to, **extra):
    d = {"invoice_id": invoice_id, "corrected_to": corrected_to, "summary": f"summary-{invoice_id}"}
    d.update(extra)
    return d


def test_version_lifecycle(tmp_path):
    audit = AuditLogger(log_dir=tmp_path / "audit")
    vc = VersionControl(data_dir=tmp_path / "corrections", max_versions_kept=3, audit_logger=audit)

    v1 = vc.save_version(TAX_ID, make_payload([rec("A1", "5121")]), operator="alice", note="init")
    assert v1 == 1
    v2 = vc.save_version(TAX_ID, make_payload([rec("A1", "5121"), rec("A2", "5122")]), operator="alice")
    assert v2 == 2

    versions = vc.list_versions(TAX_ID)
    assert [m["version"] for m in versions] == [1, 2]

    diff = vc.compare_versions(TAX_ID, 1, 2)
    assert diff["summary"] == {"added": 1, "removed": 0, "updated": 0, "unchanged": 1}
    assert diff["added"][0]["invoice_id"] == "A2"

    md = VersionControl.diff_report_markdown(diff)
    assert "新增紀錄" in md and "A2" in md

    v3 = vc.save_version(TAX_ID, make_payload([rec("A1", "5122"), rec("A2", "5122")]), operator="bob")
    diff2 = vc.compare_versions(TAX_ID, 2, 3)
    assert diff2["summary"]["updated"] == 1
    assert diff2["updated"][0]["invoice_id"] == "A1"

    vc.save_version(TAX_ID, make_payload([rec("A1", "5122")]), operator="bob")
    v5 = vc.save_version(TAX_ID, make_payload([rec("A1", "5122")]), operator="bob")
    kept = [m["version"] for m in vc.list_versions(TAX_ID)]
    assert len(kept) == 3
    assert v5 in kept
    try:
        vc.get_version(TAX_ID, 1)
        assert False, "v1 should have been pruned"
    except VersionNotFoundError:
        pass


def test_restore_version(tmp_path):
    audit = AuditLogger(log_dir=tmp_path / "audit")
    vc = VersionControl(data_dir=tmp_path / "corrections", max_versions_kept=10, audit_logger=audit)

    p1 = make_payload([rec("A1", "5121")])
    v1 = vc.save_version(TAX_ID, p1, operator="alice")
    p2 = make_payload([rec("A1", "5121"), rec("A2", "5199")])
    vc.save_version(TAX_ID, p2, operator="alice")

    restored = vc.restore_version(TAX_ID, v1, operator="carol", current_payload=p2)
    assert restored == p1

    versions = vc.list_versions(TAX_ID)
    assert len(versions) == 3
    assert versions[-1]["note"].startswith("還原前自動備份")

    logs = audit.query_logs(company_tax_id=TAX_ID, action="restore")
    assert len(logs) == 1
    assert logs[0]["new_value"]["restored_to_version"] == v1


def test_backup_manager_sync_and_cleanup(tmp_path):
    audit = AuditLogger(log_dir=tmp_path / "audit")
    bm = BackupManager(backup_root=tmp_path / "backups", retention_days=30, audit_logger=audit)

    payload = make_payload([rec("A1", "5121")])
    path = bm.create_backup(TAX_ID, payload, operator="scheduler", async_mode=False)
    assert path.exists()

    backups = bm.list_backups(TAX_ID)
    assert len(backups) == 1
    assert backups[0]["date"] == date.today().isoformat()

    path2 = bm.create_backup(TAX_ID, payload, operator="scheduler", async_mode=False)
    assert path2 != path
    backups2 = bm.list_backups(TAX_ID)
    assert len(backups2) == 2

    restored = bm.restore_backup(TAX_ID, backups2[0]["file"], operator="carol")
    assert restored == payload

    try:
        bm.restore_backup(TAX_ID, "does_not_exist.json")
        assert False
    except BackupNotFoundError:
        pass

    company_dir = bm._company_dir(TAX_ID)
    old_date = (date.today() - timedelta(days=40)).isoformat()
    old_file = company_dir / f"corrections_{old_date}.json"
    old_file.write_text(json.dumps(payload), encoding="utf-8")
    removed = bm.cleanup_old_backups(TAX_ID, retention_days=30)
    assert old_file.name in removed
    assert not old_file.exists()

    bm.shutdown()


def test_backup_manager_async_does_not_block(tmp_path):
    bm = BackupManager(backup_root=tmp_path / "backups")
    payload = make_payload([rec("A1", "5121")])

    t0 = time.monotonic()
    future = bm.create_backup(TAX_ID, payload, async_mode=True)
    submit_elapsed = time.monotonic() - t0
    assert submit_elapsed < 0.2, "submitting an async backup should return almost immediately"

    result_path = future.result(timeout=5)
    assert result_path.exists()
    bm.shutdown()


def test_audit_logger_query_and_export(tmp_path):
    audit = AuditLogger(log_dir=tmp_path / "audit")
    audit.log_action("create", "INV-1", None, {"corrected_to": "5121"}, operator="alice", company_tax_id=TAX_ID)
    audit.log_action("update", "INV-1", {"corrected_to": "5121"}, {"corrected_to": "5122"}, operator="bob", company_tax_id=TAX_ID)
    audit.log_action("delete", "INV-2", {"corrected_to": "5199"}, None, operator="bob", company_tax_id=TAX_ID)

    all_logs = audit.query_logs(company_tax_id=TAX_ID)
    assert len(all_logs) == 3

    bob_logs = audit.query_logs(company_tax_id=TAX_ID, operator="bob")
    assert len(bob_logs) == 2

    inv1_logs = audit.query_logs(invoice_id="INV-1")
    assert len(inv1_logs) == 2

    log_path = audit._current_log_path()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert {"timestamp", "action", "invoice_id", "old_value", "new_value", "operator"}.issubset(parsed.keys())

    out_jsonl = audit.export_logs(tmp_path / "export.jsonl", fmt="jsonl", company_tax_id=TAX_ID)
    assert out_jsonl.exists()
    out_json = audit.export_logs(tmp_path / "export.json", fmt="json", company_tax_id=TAX_ID)
    assert json.loads(out_json.read_text(encoding="utf-8"))
    out_csv = audit.export_logs(tmp_path / "export.csv", fmt="csv", company_tax_id=TAX_ID)
    assert out_csv.exists()


def test_performance_under_one_second(tmp_path):
    audit = AuditLogger(log_dir=tmp_path / "audit")
    vc = VersionControl(data_dir=tmp_path / "corrections", max_versions_kept=10, audit_logger=audit)
    payload = make_payload([rec(f"INV-{i}", "5121") for i in range(500)])

    t0 = time.monotonic()
    v1 = vc.save_version(TAX_ID, payload, operator="perf-test")
    save_elapsed = time.monotonic() - t0
    assert save_elapsed < 1.0, f"save_version took {save_elapsed:.3f}s"

    payload2 = make_payload([rec(f"INV-{i}", "5122") for i in range(500)])
    v2 = vc.save_version(TAX_ID, payload2, operator="perf-test")

    t0 = time.monotonic()
    vc.compare_versions(TAX_ID, v1, v2)
    compare_elapsed = time.monotonic() - t0
    assert compare_elapsed < 1.0, f"compare_versions took {compare_elapsed:.3f}s"

    t0 = time.monotonic()
    vc.restore_version(TAX_ID, v1, operator="perf-test", current_payload=payload2)
    restore_elapsed = time.monotonic() - t0
    assert restore_elapsed < 1.0, f"restore_version took {restore_elapsed:.3f}s"


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
