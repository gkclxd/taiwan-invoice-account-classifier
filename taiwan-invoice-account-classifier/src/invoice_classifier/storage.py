# -*- coding: utf-8 -*-
"""
invoice_classifier.storage
=============================
儲存與審計系統（整合自 M5-2 correction_manager.py + M6-2 version_control.py）。

包含：
    1. CorrectionManager — 客戶人工修正紀錄的 CRUD 操作，每家公司一個
       JSON 檔案（company_{tax_id}_corrections.json），寫入前自動備份。
    2. VersionControl — 具連續版本號（v1, v2, ...）的版本控制，支援
       版本比較（diff）與還原（restore）。
    3. BackupManager — 獨立於「每次寫入」的每日排程備份，保留最近 N 天。
    4. AuditLogger — 純文字 JSON Lines 格式的審計軌跡，記錄新增/更新/
       刪除/還原/備份等操作。

執行緒安全：
    CorrectionManager 與 VersionControl 皆為每個公司統編使用獨立的
    threading.RLock，避免多執行緒同時寫入同一檔案。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

BAN_PATTERN = re.compile(r"^\d{10}$")
ACCOUNT_CODE_PATTERN = re.compile(r"^\d{4}$")
DEFAULT_VECTOR_DIM = 384
DEFAULT_MAX_BACKUPS = 10

VERSION_FILE_PATTERN = re.compile(r"^company_(?P<tax_id>\d{10})_corrections_v(?P<version>\d+)\.json$")
BACKUP_FILE_PATTERN = re.compile(r"^corrections_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<seq>\d+))?\.json$")


# ---------------------------------------------------------------------------
# 例外類別
# ---------------------------------------------------------------------------

class CorrectionManagerError(Exception):
    """CorrectionManager 相關錯誤的基底類別"""


class ValidationError(CorrectionManagerError):
    """輸入資料驗證失敗"""


class RecordNotFoundError(CorrectionManagerError):
    """查無指定的修正紀錄"""


class StorageError(CorrectionManagerError):
    """檔案讀寫 / 格式錯誤"""


class ConfirmationRequiredError(CorrectionManagerError):
    """刪除操作缺少明確確認（避免誤刪）"""


class VersionControlError(Exception):
    """VersionControl / BackupManager / AuditLogger 共用的例外基底類別"""


class VersionNotFoundError(VersionControlError):
    """指定的版本不存在"""


class BackupNotFoundError(VersionControlError):
    """指定的備份不存在"""


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _today_str() -> str:
    return date.today().isoformat()


def _validate_ban(value: str, field_name: str = "統一編號") -> str:
    if not isinstance(value, str) or not BAN_PATTERN.match(value):
        raise ValidationError(f"{field_name}必須為 10 碼數字字串，實際收到：{value!r}")
    return value


def _validate_account_code(value: str, field_name: str = "會計科目代碼") -> str:
    if not isinstance(value, str) or not ACCOUNT_CODE_PATTERN.match(value):
        raise ValidationError(
            f"{field_name}必須為 4 碼數字字串（系統預設格式，企業可自訂對照表覆寫），"
            f"實際收到：{value!r}"
        )
    return value


def _validate_unit_interval(value: float, field_name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name}必須為數值，實際收到：{value!r}")
    if not (0.0 <= v <= 1.0):
        raise ValidationError(f"{field_name}必須介於 0-1 之間，實際收到：{value!r}")
    return v


def _validate_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name}不可為空字串")
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    """以 temp file + os.replace 方式原子寫入 JSON，避免寫到一半損毀檔案。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    return json.loads(text)


def _records_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = payload.get("corrections", []) if isinstance(payload, dict) else []
    index: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        rid = rec.get("invoice_id")
        if rid is not None:
            index[str(rid)] = rec
    return index


# ---------------------------------------------------------------------------
# CorrectionRecord（儲存層專用 dataclass，與 data_models.CorrectionRecord 欄位一致）
# ---------------------------------------------------------------------------

@dataclass
class CorrectionRecord:
    """客戶人工修正紀錄（儲存層 dataclass 版本，欄位與 data_models.CorrectionRecord 一致）"""

    invoice_id: str
    timestamp: str
    summary: str
    summary_vector: List[float]
    original_pred: str
    corrected_to: str
    confidence_weight: float
    seller_ban: str
    buyer_ban: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CorrectionRecord":
        try:
            return cls(
                invoice_id=d["invoice_id"],
                timestamp=d["timestamp"],
                summary=d["summary"],
                summary_vector=list(d["summary_vector"]),
                original_pred=d["original_pred"],
                corrected_to=d["corrected_to"],
                confidence_weight=float(d["confidence_weight"]),
                seller_ban=d["seller_ban"],
                buyer_ban=d["buyer_ban"],
            )
        except KeyError as exc:
            raise StorageError(f"修正紀錄缺少必要欄位：{exc}") from exc


# ---------------------------------------------------------------------------
# 向量化（優先使用專案向量器，退回確定性雜湊向量）
# ---------------------------------------------------------------------------

def _fallback_hash_vector(text: str, dim: int = DEFAULT_VECTOR_DIM) -> List[float]:
    """
    確定性雜湊型向量產生器（無法載入 preprocessor.InvoiceVectorizer 時的退回方案）。
    相同輸入 -> 永遠相同輸出（可重現、可測試）。
    """
    text_bytes = text.encode("utf-8")
    needed_bytes = dim * 4
    buf = bytearray()
    counter = 0
    while len(buf) < needed_bytes:
        h = hashlib.sha256(text_bytes + counter.to_bytes(4, "big")).digest()
        buf.extend(h)
        counter += 1
    raw = [int.from_bytes(buf[i:i + 4], "big") for i in range(0, needed_bytes, 4)]
    vec = [(x / 0xFFFFFFFF) * 2.0 - 1.0 for x in raw]
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _load_project_vectorizer():
    """嘗試載入專案既有的 InvoiceVectorizer，失敗則回傳 None。"""
    try:
        from .preprocessor import InvoiceVectorizer
        return InvoiceVectorizer(backend="tfidf_svd")
    except Exception:
        return None


class SummaryVectorizer:
    """摘要向量化包裝器：優先使用專案向量器，否則退回確定性雜湊向量。"""

    def __init__(self, dim: int = DEFAULT_VECTOR_DIM):
        self.dim = dim
        self._impl = _load_project_vectorizer()

    def encode(self, text: str) -> List[float]:
        if self._impl is not None:
            try:
                vec = self._impl.encode(text)
                return [float(x) for x in list(vec)]
            except Exception:
                pass
        return _fallback_hash_vector(text, self.dim)


# ---------------------------------------------------------------------------
# 檔案層：讀寫 + 即時備份
# ---------------------------------------------------------------------------

class CorrectionStore:
    """單一公司修正紀錄檔案的讀寫層，每次寫入前自動備份舊檔案。"""

    def __init__(self, data_dir: str, company_tax_id: str, max_backups: int = DEFAULT_MAX_BACKUPS):
        _validate_ban(company_tax_id, "公司統一編號")
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.company_tax_id = company_tax_id
        self.max_backups = max_backups
        self.file_path = self.data_dir / f"company_{company_tax_id}_corrections.json"

    def _empty_payload(self) -> Dict[str, Any]:
        return {
            "company_tax_id": self.company_tax_id,
            "corrections": [],
            "seller_preferences": {},
            "last_updated": _utc_now_iso(),
        }

    def _backup_current_file(self) -> Optional[Path]:
        if not self.file_path.exists():
            return None
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        backup_path = self.backup_dir / f"company_{self.company_tax_id}_corrections.{ts}.bak.json"
        try:
            shutil.copy2(self.file_path, backup_path)
        except OSError as exc:
            raise StorageError(f"備份舊檔案失敗：{exc}") from exc
        self._prune_backups()
        return backup_path

    def _prune_backups(self) -> None:
        pattern = f"company_{self.company_tax_id}_corrections.*.bak.json"
        backups = sorted(self.backup_dir.glob(pattern), key=lambda p: p.name)
        excess = len(backups) - self.max_backups
        for old in backups[:max(excess, 0)]:
            try:
                old.unlink()
            except OSError:
                pass

    def load(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            return self._empty_payload()
        try:
            raw_text = self.file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"無法讀取檔案 {self.file_path}：{exc}") from exc
        if not raw_text.strip():
            return self._empty_payload()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise StorageError(f"檔案 {self.file_path} 非合法 JSON：{exc}") from exc

        if not isinstance(payload, dict):
            raise StorageError(f"檔案 {self.file_path} 內容格式錯誤：頂層必須為物件(dict)")

        payload.setdefault("company_tax_id", self.company_tax_id)
        payload.setdefault("corrections", [])
        payload.setdefault("seller_preferences", {})
        payload.setdefault("last_updated", _utc_now_iso())

        if not isinstance(payload["corrections"], list):
            raise StorageError(f"檔案 {self.file_path} 內容格式錯誤：corrections 必須為陣列")

        return payload

    def save(self, payload: Dict[str, Any]) -> None:
        self._backup_current_file()
        payload["last_updated"] = _utc_now_iso()
        tmp_path = self.file_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.file_path)
        except OSError as exc:
            raise StorageError(f"寫入檔案 {self.file_path} 失敗：{exc}") from exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# AuditLogger：審計軌跡（合併版，同時支援簡易與完整格式）
# ---------------------------------------------------------------------------

@dataclass
class AuditLogEntry:
    timestamp: str
    action: str
    invoice_id: Optional[str]
    old_value: Any
    new_value: Any
    operator: str
    company_tax_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    """
    將所有操作（新增/更新/刪除/還原/備份等）以 JSON Lines 格式追加寫入審計日誌。
    """

    def __init__(self, log_dir: Union[str, Path] = "./data/audit", rotate_monthly: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_monthly = rotate_monthly
        self._lock = threading.Lock()

    def _current_log_path(self) -> Path:
        if self.rotate_monthly:
            suffix = datetime.now(timezone.utc).strftime("%Y%m")
            return self.log_dir / f"audit_log_{suffix}.jsonl"
        return self.log_dir / "audit_log.jsonl"

    def log_action(
        self,
        action: str,
        invoice_id: Optional[str],
        old_value: Any,
        new_value: Any,
        operator: str = "system",
        company_tax_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            timestamp=timestamp or _utc_now_iso(),
            action=action,
            invoice_id=invoice_id,
            old_value=old_value,
            new_value=new_value,
            operator=operator,
            company_tax_id=company_tax_id,
        )
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with self._lock:
            with self._current_log_path().open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry

    # 相容 M5-2 的簡化呼叫介面：log(AuditEntry-like)
    def log(self, entry: "AuditLogEntry") -> None:
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with self._lock:
            with self._current_log_path().open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _iter_log_files(self) -> List[Path]:
        if self.rotate_monthly:
            return sorted(self.log_dir.glob("audit_log_*.jsonl"))
        path = self._current_log_path()
        return [path] if path.exists() else []

    def query_logs(
        self,
        company_tax_id: Optional[str] = None,
        action: Optional[str] = None,
        invoice_id: Optional[str] = None,
        operator: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for log_path in self._iter_log_files():
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if company_tax_id is not None and entry.get("company_tax_id") != company_tax_id:
                        continue
                    if action is not None and entry.get("action") != action:
                        continue
                    if invoice_id is not None and entry.get("invoice_id") != invoice_id:
                        continue
                    if operator is not None and entry.get("operator") != operator:
                        continue
                    ts = entry.get("timestamp", "")
                    if start_time is not None and ts < start_time:
                        continue
                    if end_time is not None and ts > end_time:
                        continue
                    results.append(entry)
        results.sort(key=lambda e: e.get("timestamp", ""))
        return results

    def export_logs(self, out_path: Union[str, Path], fmt: str = "jsonl", **query_kwargs: Any) -> Path:
        entries = self.query_logs(**query_kwargs)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fmt_normalized = fmt.lower().strip()
        if fmt_normalized == "jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        elif fmt_normalized == "json":
            out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        elif fmt_normalized == "csv":
            fieldnames = ["timestamp", "action", "invoice_id", "old_value", "new_value", "operator", "company_tax_id"]
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for e in entries:
                    row = dict(e)
                    for k in ("old_value", "new_value"):
                        if not isinstance(row.get(k), (str, int, float, type(None))):
                            row[k] = json.dumps(row[k], ensure_ascii=False)
                    writer.writerow(row)
        else:
            raise VersionControlError(f"不支援的匯出格式：{fmt}（僅支援 jsonl/json/csv）")
        return out_path


# ---------------------------------------------------------------------------
# CorrectionManager：對外主要 CRUD 介面
# ---------------------------------------------------------------------------

class CorrectionManager:
    """
    修正紀錄管理系統主類別（CRUD + 備份 + 審計）。

    執行緒安全策略：每個公司統編對應一把 threading.RLock，
    建立/查找公司專屬鎖的過程由全域 `_locks_guard` 鎖保護。
    """

    def __init__(
        self,
        data_dir: str = "./data/corrections",
        max_backups: int = DEFAULT_MAX_BACKUPS,
        vectorizer: Optional[SummaryVectorizer] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.data_dir = data_dir
        self.max_backups = max_backups
        self.vectorizer = vectorizer or SummaryVectorizer()
        self.audit_logger = audit_logger or AuditLogger(data_dir)
        self._locks: Dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, company_tax_id: str) -> threading.RLock:
        with self._locks_guard:
            if company_tax_id not in self._locks:
                self._locks[company_tax_id] = threading.RLock()
            return self._locks[company_tax_id]

    def _store_for(self, company_tax_id: str) -> CorrectionStore:
        return CorrectionStore(self.data_dir, company_tax_id, self.max_backups)

    def _audit(self, company_tax_id: str, operation: str, invoice_id: Optional[str], **detail: Any) -> None:
        self.audit_logger.log_action(
            action=operation, invoice_id=invoice_id, old_value=None, new_value=detail,
            operator="system", company_tax_id=company_tax_id,
        )

    @staticmethod
    def _matches_date_range(timestamp: str, date_from: Optional[str], date_to: Optional[str]) -> bool:
        if not date_from and not date_to:
            return True
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return False
        if date_from:
            try:
                start = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            except ValueError:
                start = datetime.fromisoformat(date_from + "T00:00:00+00:00")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if ts < start:
                return False
        if date_to:
            try:
                end = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            except ValueError:
                end = datetime.fromisoformat(date_to + "T23:59:59+00:00")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            if ts > end:
                return False
        return True

    def add_correction(
        self,
        company_tax_id: str,
        invoice_id: str,
        summary: str,
        original_pred: str,
        corrected_to: str,
        seller_ban: str,
        buyer_ban: str,
        confidence_weight: float = 1.0,
        allow_duplicate: bool = False,
    ) -> CorrectionRecord:
        """新增一筆修正紀錄（timestamp、summary_vector 自動生成）。"""
        _validate_ban(company_tax_id, "公司統一編號")
        _validate_non_empty_str(invoice_id, "invoice_id")
        _validate_non_empty_str(summary, "summary")
        _validate_account_code(original_pred, "母版預測科目代碼")
        _validate_account_code(corrected_to, "修正後科目代碼")
        _validate_ban(seller_ban, "賣方統一編號")
        _validate_ban(buyer_ban, "買方統一編號")
        confidence_weight = _validate_unit_interval(confidence_weight, "confidence_weight")

        lock = self._lock_for(company_tax_id)
        with lock:
            store = self._store_for(company_tax_id)
            payload = store.load()

            if not allow_duplicate:
                for existing in payload["corrections"]:
                    if existing.get("invoice_id") == invoice_id:
                        raise ValidationError(
                            f"invoice_id={invoice_id!r} 已存在修正紀錄，"
                            f"如需新增第二筆請傳入 allow_duplicate=True，"
                            f"或改用 update_correction 更新既有紀錄"
                        )

            record = CorrectionRecord(
                invoice_id=invoice_id,
                timestamp=_utc_now_iso(),
                summary=summary,
                summary_vector=self.vectorizer.encode(summary),
                original_pred=original_pred,
                corrected_to=corrected_to,
                confidence_weight=confidence_weight,
                seller_ban=seller_ban,
                buyer_ban=buyer_ban,
            )
            payload["corrections"].append(record.to_dict())

            seller_prefs = payload.setdefault("seller_preferences", {})
            seller_stat = seller_prefs.setdefault(seller_ban, {})
            seller_stat[corrected_to] = seller_stat.get(corrected_to, 0) + 1

            store.save(payload)
            self._audit(company_tax_id, "create", invoice_id, corrected_to=corrected_to)
            return record

    def query_corrections(
        self,
        company_tax_id: str,
        invoice_id: Optional[str] = None,
        corrected_to: Optional[str] = None,
        seller_ban: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[CorrectionRecord]:
        """依條件查詢修正紀錄（所有條件皆為選填，多條件為 AND 關係）。"""
        _validate_ban(company_tax_id, "公司統一編號")
        lock = self._lock_for(company_tax_id)
        with lock:
            store = self._store_for(company_tax_id)
            payload = store.load()

        results: List[CorrectionRecord] = []
        for raw in payload["corrections"]:
            if invoice_id is not None and raw.get("invoice_id") != invoice_id:
                continue
            if corrected_to is not None and raw.get("corrected_to") != corrected_to:
                continue
            if seller_ban is not None and raw.get("seller_ban") != seller_ban:
                continue
            if not self._matches_date_range(raw.get("timestamp", ""), date_from, date_to):
                continue
            results.append(CorrectionRecord.from_dict(raw))
        return results

    def update_correction(
        self,
        company_tax_id: str,
        invoice_id: str,
        confidence_weight: Optional[float] = None,
        corrected_to: Optional[str] = None,
    ) -> CorrectionRecord:
        """更新既有修正紀錄的 confidence_weight 與/或 corrected_to，自動更新 timestamp。"""
        _validate_ban(company_tax_id, "公司統一編號")
        _validate_non_empty_str(invoice_id, "invoice_id")
        if confidence_weight is None and corrected_to is None:
            raise ValidationError("update_correction 至少需提供 confidence_weight 或 corrected_to 其中之一")
        if confidence_weight is not None:
            confidence_weight = _validate_unit_interval(confidence_weight, "confidence_weight")
        if corrected_to is not None:
            _validate_account_code(corrected_to, "修正後科目代碼")

        lock = self._lock_for(company_tax_id)
        with lock:
            store = self._store_for(company_tax_id)
            payload = store.load()

            candidates = [
                (idx, rec) for idx, rec in enumerate(payload["corrections"])
                if rec.get("invoice_id") == invoice_id
            ]
            if not candidates:
                raise RecordNotFoundError(f"查無 invoice_id={invoice_id!r} 的修正紀錄")

            idx, target = max(candidates, key=lambda pair: pair[1].get("timestamp", ""))

            old_corrected_to = target.get("corrected_to")
            old_seller_ban = target.get("seller_ban")

            if confidence_weight is not None:
                target["confidence_weight"] = confidence_weight
            if corrected_to is not None:
                target["corrected_to"] = corrected_to
            target["timestamp"] = _utc_now_iso()

            payload["corrections"][idx] = target

            if corrected_to is not None and corrected_to != old_corrected_to and old_seller_ban:
                seller_prefs = payload.setdefault("seller_preferences", {})
                seller_stat = seller_prefs.setdefault(old_seller_ban, {})
                if old_corrected_to in seller_stat:
                    seller_stat[old_corrected_to] = max(0, seller_stat[old_corrected_to] - 1)
                seller_stat[corrected_to] = seller_stat.get(corrected_to, 0) + 1

            store.save(payload)
            self._audit(
                company_tax_id, "update", invoice_id,
                confidence_weight=confidence_weight, corrected_to=corrected_to,
            )
            return CorrectionRecord.from_dict(target)

    def delete_correction(self, company_tax_id: str, invoice_id: str, confirm: bool = False) -> int:
        """刪除指定 invoice_id 的修正紀錄；必須明確傳入 confirm=True 才會執行刪除。"""
        _validate_ban(company_tax_id, "公司統一編號")
        _validate_non_empty_str(invoice_id, "invoice_id")
        if not confirm:
            raise ConfirmationRequiredError(
                f"刪除 invoice_id={invoice_id!r} 需明確傳入 confirm=True 以避免誤刪"
            )

        lock = self._lock_for(company_tax_id)
        with lock:
            store = self._store_for(company_tax_id)
            payload = store.load()

            remaining = []
            removed = []
            for rec in payload["corrections"]:
                if rec.get("invoice_id") == invoice_id:
                    removed.append(rec)
                else:
                    remaining.append(rec)

            if not removed:
                raise RecordNotFoundError(f"查無 invoice_id={invoice_id!r} 的修正紀錄，無法刪除")

            payload["corrections"] = remaining

            seller_prefs = payload.setdefault("seller_preferences", {})
            for rec in removed:
                seller_ban = rec.get("seller_ban")
                corrected_to = rec.get("corrected_to")
                if seller_ban and seller_ban in seller_prefs:
                    stat = seller_prefs[seller_ban]
                    if corrected_to in stat:
                        stat[corrected_to] = max(0, stat[corrected_to] - 1)

            store.save(payload)
            self._audit(company_tax_id, "delete", invoice_id, removed_count=len(removed))
            return len(removed)

    def export_corrections(self, company_tax_id: str, output_path: str, fmt: str = "json") -> str:
        """批次匯出修正紀錄，格式支援 'json' 或 'csv'。"""
        _validate_ban(company_tax_id, "公司統一編號")
        fmt_normalized = fmt.lower().strip()
        if fmt_normalized not in ("json", "csv"):
            raise ValidationError(f"不支援的匯出格式：{fmt!r}，僅支援 'json' 或 'csv'")

        lock = self._lock_for(company_tax_id)
        with lock:
            store = self._store_for(company_tax_id)
            payload = store.load()

        records = payload["corrections"]
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "invoice_id", "timestamp", "summary", "summary_vector",
            "original_pred", "corrected_to", "confidence_weight",
            "seller_ban", "buyer_ban",
        ]

        try:
            if fmt_normalized == "json":
                out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                with out_path.open("w", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for rec in records:
                        row = dict(rec)
                        row["summary_vector"] = json.dumps(row.get("summary_vector", []))
                        writer.writerow(row)
        except OSError as exc:
            raise StorageError(f"匯出檔案 {out_path} 失敗：{exc}") from exc

        self._audit(company_tax_id, "export", None, fmt=fmt_normalized, count=len(records), path=str(out_path))
        return str(out_path)


# ---------------------------------------------------------------------------
# VersionControl：版本控制
# ---------------------------------------------------------------------------

class VersionControl:
    """
    追蹤單一公司修正紀錄檔案的版本歷史（連續版本號 v1, v2, ...）。
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "./data/corrections",
        max_versions_kept: int = 10,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.data_dir = Path(data_dir)
        self.versions_dir = self.data_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.max_versions_kept = max_versions_kept
        self.audit_logger = audit_logger
        self._locks_guard = threading.Lock()
        self._locks: Dict[str, threading.RLock] = {}

    def _lock_for(self, company_tax_id: str) -> threading.RLock:
        with self._locks_guard:
            if company_tax_id not in self._locks:
                self._locks[company_tax_id] = threading.RLock()
            return self._locks[company_tax_id]

    def _index_path(self, company_tax_id: str) -> Path:
        return self.versions_dir / f"company_{company_tax_id}_version_index.json"

    def _version_path(self, company_tax_id: str, version: int) -> Path:
        return self.versions_dir / f"company_{company_tax_id}_corrections_v{version}.json"

    def _load_index(self, company_tax_id: str) -> Dict[str, Any]:
        idx_path = self._index_path(company_tax_id)
        if not idx_path.exists():
            return {"company_tax_id": company_tax_id, "latest_version": 0, "versions": []}
        try:
            data = _load_json(idx_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionControlError(f"版本索引檔損毀：{idx_path}：{exc}") from exc
        data.setdefault("company_tax_id", company_tax_id)
        data.setdefault("latest_version", 0)
        data.setdefault("versions", [])
        return data

    def _save_index(self, company_tax_id: str, index_data: Dict[str, Any]) -> None:
        _atomic_write_json(self._index_path(company_tax_id), index_data)

    def _prune_versions(self, company_tax_id: str, index_data: Dict[str, Any]) -> None:
        versions_meta: List[Dict[str, Any]] = index_data["versions"]
        versions_meta.sort(key=lambda m: m["version"])
        excess = len(versions_meta) - self.max_versions_kept
        if excess <= 0:
            return
        to_remove = versions_meta[:excess]
        for meta in to_remove:
            vpath = self._version_path(company_tax_id, meta["version"])
            try:
                if vpath.exists():
                    vpath.unlink()
            except OSError:
                pass
        index_data["versions"] = versions_meta[excess:]

    def _audit(self, company_tax_id: str, action: str, operator: str, old_value: Any, new_value: Any,
               invoice_id: Optional[str] = None) -> None:
        if self.audit_logger is not None:
            self.audit_logger.log_action(
                action=action, invoice_id=invoice_id, old_value=old_value, new_value=new_value,
                operator=operator, company_tax_id=company_tax_id,
            )

    def save_version(self, company_tax_id: str, payload: Dict[str, Any], operator: str = "system",
                      note: str = "") -> int:
        """儲存目前修正紀錄內容為一個新版本，回傳新版本號（從 1 開始遞增）。"""
        t0 = time.monotonic()
        lock = self._lock_for(company_tax_id)
        with lock:
            index_data = self._load_index(company_tax_id)
            new_version = int(index_data["latest_version"]) + 1

            version_path = self._version_path(company_tax_id, new_version)
            _atomic_write_json(version_path, payload)

            record_count = len(payload.get("corrections", [])) if isinstance(payload, dict) else 0
            meta = {
                "version": new_version, "timestamp": _utc_now_iso(), "operator": operator,
                "note": note, "record_count": record_count, "file": version_path.name,
            }
            index_data["versions"].append(meta)
            index_data["latest_version"] = new_version
            self._prune_versions(company_tax_id, index_data)
            self._save_index(company_tax_id, index_data)

        self._audit(
            company_tax_id, "save_version", operator, old_value=None,
            new_value={"version": new_version, "record_count": record_count, "note": note},
        )

        elapsed = time.monotonic() - t0
        if elapsed >= 1.0:
            self._audit(company_tax_id, "perf_warning", "system", old_value=None,
                        new_value={"operation": "save_version", "elapsed_seconds": elapsed})
        return new_version

    def list_versions(self, company_tax_id: str) -> List[Dict[str, Any]]:
        index_data = self._load_index(company_tax_id)
        return sorted(index_data["versions"], key=lambda m: m["version"])

    def get_version(self, company_tax_id: str, version: int) -> Dict[str, Any]:
        vpath = self._version_path(company_tax_id, version)
        if not vpath.exists():
            raise VersionNotFoundError(f"公司 {company_tax_id} 的版本 v{version} 不存在（可能已因保留數量上限被清除）")
        try:
            return _load_json(vpath)
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionControlError(f"讀取版本檔失敗 {vpath}：{exc}") from exc

    def compare_versions(self, company_tax_id: str, version_from: int, version_to: int) -> Dict[str, Any]:
        """比較兩個版本之間修正紀錄的差異（新增/刪除/更新/未變更）。"""
        payload_from = self.get_version(company_tax_id, version_from)
        payload_to = self.get_version(company_tax_id, version_to)

        idx_from = _records_index(payload_from)
        idx_to = _records_index(payload_to)

        added_ids = set(idx_to) - set(idx_from)
        removed_ids = set(idx_from) - set(idx_to)
        common_ids = set(idx_from) & set(idx_to)

        added = [idx_to[i] for i in sorted(added_ids)]
        removed = [idx_from[i] for i in sorted(removed_ids)]

        updated: List[Dict[str, Any]] = []
        unchanged_count = 0
        for rid in sorted(common_ids):
            before, after = idx_from[rid], idx_to[rid]
            if before != after:
                changed_fields = sorted(
                    k for k in set(before.keys()) | set(after.keys()) if before.get(k) != after.get(k)
                )
                updated.append({
                    "invoice_id": rid, "before": before, "after": after, "changed_fields": changed_fields,
                })
            else:
                unchanged_count += 1

        return {
            "company_tax_id": company_tax_id, "version_from": version_from, "version_to": version_to,
            "added": added, "removed": removed, "updated": updated, "unchanged_count": unchanged_count,
            "summary": {
                "added": len(added), "removed": len(removed), "updated": len(updated),
                "unchanged": unchanged_count,
            },
        }

    @staticmethod
    def diff_report_markdown(diff: Dict[str, Any]) -> str:
        lines = [
            f"# 版本差異報告：公司 {diff['company_tax_id']}", "",
            f"比較版本：v{diff['version_from']} → v{diff['version_to']}", "",
            f"- 新增：{diff['summary']['added']} 筆",
            f"- 刪除：{diff['summary']['removed']} 筆",
            f"- 更新：{diff['summary']['updated']} 筆",
            f"- 未變更：{diff['summary']['unchanged']} 筆", "",
        ]
        if diff["added"]:
            lines.append("## 新增紀錄\n")
            for rec in diff["added"]:
                lines.append(f"- `{rec.get('invoice_id')}`：修正為 {rec.get('corrected_to')}")
            lines.append("")
        if diff["removed"]:
            lines.append("## 刪除紀錄\n")
            for rec in diff["removed"]:
                lines.append(f"- `{rec.get('invoice_id')}`：原修正為 {rec.get('corrected_to')}")
            lines.append("")
        if diff["updated"]:
            lines.append("## 更新紀錄\n")
            for item in diff["updated"]:
                lines.append(
                    f"- `{item['invoice_id']}`：欄位變更 {', '.join(item['changed_fields'])}"
                    f"（{item['before'].get('corrected_to')} → {item['after'].get('corrected_to')}）"
                )
            lines.append("")
        return "\n".join(lines)

    def restore_version(self, company_tax_id: str, version: int, operator: str = "system",
                         current_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """還原到指定版本；若提供 current_payload，先自動備份當前版本。"""
        t0 = time.monotonic()
        lock = self._lock_for(company_tax_id)
        with lock:
            target_payload = self.get_version(company_tax_id, version)

            backed_up_as: Optional[int] = None
            if current_payload is not None:
                backed_up_as = self.save_version(
                    company_tax_id, current_payload, operator=operator,
                    note=f"還原前自動備份（即將還原至 v{version}）",
                )

        self._audit(
            company_tax_id, "restore", operator,
            old_value={"backed_up_as_version": backed_up_as},
            new_value={"restored_to_version": version},
        )

        elapsed = time.monotonic() - t0
        if elapsed >= 1.0:
            self._audit(company_tax_id, "perf_warning", "system", old_value=None,
                        new_value={"operation": "restore_version", "elapsed_seconds": elapsed})
        return target_payload


# ---------------------------------------------------------------------------
# BackupManager：每日排程備份
# ---------------------------------------------------------------------------

class BackupManager:
    """負責「每日排程」備份（獨立於 VersionControl 的即時版本快照）。"""

    def __init__(
        self,
        backup_root: Union[str, Path] = "./backups",
        retention_days: int = 30,
        audit_logger: Optional[AuditLogger] = None,
        max_workers: int = 4,
    ):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.audit_logger = audit_logger
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="backup-worker")
        self._scheduler_threads: Dict[str, threading.Thread] = {}
        self._scheduler_stop_events: Dict[str, threading.Event] = {}

    def _company_dir(self, company_tax_id: str) -> Path:
        d = self.backup_root / f"company_{company_tax_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_backup_path(self, company_tax_id: str, backup_date: Optional[str] = None) -> Path:
        backup_date = backup_date or _today_str()
        company_dir = self._company_dir(company_tax_id)
        candidate = company_dir / f"corrections_{backup_date}.json"
        if not candidate.exists():
            return candidate
        seq = 2
        while True:
            candidate = company_dir / f"corrections_{backup_date}_{seq}.json"
            if not candidate.exists():
                return candidate
            seq += 1

    def _do_backup(self, company_tax_id: str, payload: Dict[str, Any], operator: str) -> Path:
        target_path = self._next_backup_path(company_tax_id)
        _atomic_write_json(target_path, payload)
        if self.audit_logger is not None:
            self.audit_logger.log_action(
                action="backup", invoice_id=None, old_value=None,
                new_value={"backup_file": target_path.name}, operator=operator,
                company_tax_id=company_tax_id,
            )
        self.cleanup_old_backups(company_tax_id)
        return target_path

    def create_backup(self, company_tax_id: str, payload: Dict[str, Any], operator: str = "system",
                       async_mode: bool = True) -> Union["Future[Path]", Path]:
        """建立備份；async_mode=True 提交至背景執行緒池，不阻塞呼叫端。"""
        if async_mode:
            return self._executor.submit(self._do_backup, company_tax_id, payload, operator)
        return self._do_backup(company_tax_id, payload, operator)

    def list_backups(self, company_tax_id: str) -> List[Dict[str, Any]]:
        company_dir = self._company_dir(company_tax_id)
        results = []
        for p in sorted(company_dir.glob("corrections_*.json")):
            m = BACKUP_FILE_PATTERN.match(p.name)
            if not m:
                continue
            stat = p.stat()
            results.append({
                "file": p.name, "date": m.group("date"),
                "sequence": int(m.group("seq")) if m.group("seq") else 1,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
        return results

    def restore_backup(self, company_tax_id: str, backup_file: str, operator: str = "system") -> Dict[str, Any]:
        company_dir = self._company_dir(company_tax_id)
        backup_path = company_dir / backup_file
        if not backup_path.exists():
            raise BackupNotFoundError(f"公司 {company_tax_id} 找不到備份檔：{backup_file}")
        try:
            payload = _load_json(backup_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionControlError(f"讀取備份檔失敗 {backup_path}：{exc}") from exc

        if self.audit_logger is not None:
            self.audit_logger.log_action(
                action="restore_backup", invoice_id=None, old_value=None,
                new_value={"restored_from": backup_file}, operator=operator,
                company_tax_id=company_tax_id,
            )
        return payload

    def cleanup_old_backups(self, company_tax_id: str, retention_days: Optional[int] = None) -> List[str]:
        retention_days = retention_days if retention_days is not None else self.retention_days
        cutoff = date.today().toordinal() - retention_days
        removed: List[str] = []
        for meta in self.list_backups(company_tax_id):
            try:
                backup_ordinal = date.fromisoformat(meta["date"]).toordinal()
            except ValueError:
                continue
            if backup_ordinal < cutoff:
                company_dir = self._company_dir(company_tax_id)
                path = company_dir / meta["file"]
                try:
                    path.unlink()
                    removed.append(meta["file"])
                except OSError:
                    pass
        return removed

    def schedule_daily_backup(self, company_tax_id: str, get_payload_fn, run_at: str = "02:00",
                               operator: str = "scheduler") -> None:
        """啟動每日定時背景備份執行緒（daemon thread）。"""
        self.stop_scheduled_backup(company_tax_id)
        stop_event = threading.Event()
        self._scheduler_stop_events[company_tax_id] = stop_event

        try:
            run_hour, run_minute = (int(x) for x in run_at.split(":"))
        except ValueError as exc:
            raise VersionControlError(f"run_at 格式錯誤，應為 'HH:MM'，實際收到：{run_at!r}") from exc

        def _loop():
            while not stop_event.is_set():
                now = datetime.now()
                target = now.replace(hour=run_hour, minute=run_minute, second=0, microsecond=0)
                if target <= now:
                    target = target + _one_day()
                sleep_seconds = max((target - now).total_seconds(), 1.0)
                if stop_event.wait(timeout=sleep_seconds):
                    return
                try:
                    payload = get_payload_fn()
                    self.create_backup(company_tax_id, payload, operator=operator, async_mode=False)
                except Exception:
                    continue

        thread = threading.Thread(target=_loop, name=f"daily-backup-{company_tax_id}", daemon=True)
        self._scheduler_threads[company_tax_id] = thread
        thread.start()

    def stop_scheduled_backup(self, company_tax_id: str) -> None:
        stop_event = self._scheduler_stop_events.pop(company_tax_id, None)
        if stop_event is not None:
            stop_event.set()
        thread = self._scheduler_threads.pop(company_tax_id, None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def shutdown(self, wait: bool = True) -> None:
        for tax_id in list(self._scheduler_stop_events.keys()):
            self.stop_scheduled_backup(tax_id)
        self._executor.shutdown(wait=wait)


def _one_day():
    from datetime import timedelta
    return timedelta(days=1)


__all__ = [
    "CorrectionManager", "CorrectionRecord", "CorrectionStore", "SummaryVectorizer",
    "VersionControl", "BackupManager", "AuditLogger", "AuditLogEntry",
    "CorrectionManagerError", "ValidationError", "RecordNotFoundError", "StorageError",
    "ConfirmationRequiredError", "VersionControlError", "VersionNotFoundError", "BackupNotFoundError",
]
