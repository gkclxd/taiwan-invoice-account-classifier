# -*- coding: utf-8 -*-
"""
invoice_classifier.personalization
====================================
客戶個人化學習層（整合自 M3-1 personalization_prototype.py）。

功能：
    將同一家公司的人工修正紀錄（CorrectionRecord 列表），依修正後科目分組，
    計算「加權平均原型向量」，建立客戶專屬的科目偏好原型字典。

權重組成（三項相乘）：
    1. confidence_weight  — 人工設定的可信度（0-1）
    2. time_decay_weight  — 時間衰減，半衰期預設 90 天（越新權重越高）
    3. count_boost_weight — 修正次數加成，最多以 5 次歸一化

增量更新：
    PrototypeStore 內同時保存「加權向量總和」與「權重總和」兩個中間量
    （_sum_vector, _sum_weight），因此新增修正紀錄時只需針對受影響科目
    重新累加，不需重算全部歷史資料。

擴充點：
    本模組不假設固定的科目代碼集合，任何 4 碼科目代碼皆可作為
    corrected_to 出現；若企業使用自訂會計科目對照表（見
    data_models.CompanyProfile.custom_account_mapping），該轉換應在
    呼叫本模組前後由上層（classifier.py）處理，本模組僅負責向量學習本身。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .data_models import CorrectionRecord


@dataclass
class AccountPrototype:
    """單一會計科目的偏好原型（可增量更新的中間狀態）"""

    account_code: str
    prototype_vector: np.ndarray
    correction_count: int
    avg_weight: float
    last_corrected: str
    _sum_vector: np.ndarray = field(repr=False, default=None)
    _sum_weight: float = field(repr=False, default=0.0)
    _weight_history_sum: float = field(repr=False, default=0.0)

    def to_dict(self) -> dict:
        return {
            "prototype_vector": self.prototype_vector,
            "correction_count": self.correction_count,
            "avg_weight": self.avg_weight,
            "last_corrected": self.last_corrected,
        }


# ----------------------------------------------------------------------------
# 1. 時間衰減函式
# ----------------------------------------------------------------------------

def time_decay_weight(
    corrected_at: str,
    now: Optional[datetime] = None,
    half_life_days: float = 90.0,
) -> float:
    """計算時間衰減權重（0-1），半衰期預設 90 天。

    公式：weight = 0.5 ** (elapsed_days / half_life_days)

    邊界處理：
        - 時間戳解析失敗時回傳保守值 0.5，不拋出例外（避免單筆髒資料中斷整批計算）。
        - 若 corrected_at 晚於 now（時鐘漂移/測試資料），elapsed_days 視為 0，回傳權重 1.0。
    """
    if not corrected_at:
        return 0.5

    try:
        ts = _parse_iso8601(corrected_at)
    except (ValueError, TypeError):
        return 0.5

    ref_now = now if now is not None else datetime.now(timezone.utc)

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)

    elapsed_days = (ref_now - ts).total_seconds() / 86400.0
    if elapsed_days <= 0:
        return 1.0
    if half_life_days <= 0:
        return 1.0

    decay = 0.5 ** (elapsed_days / half_life_days)
    return float(max(0.0, min(1.0, decay)))


def _parse_iso8601(s: str) -> datetime:
    """寬鬆解析 ISO 8601 字串，支援結尾 'Z' 表示 UTC。"""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ----------------------------------------------------------------------------
# 2. 修正次數加成函式（最多以 5 次歸一化）
# ----------------------------------------------------------------------------

def count_boost_weight(count_so_far: int, cap: int = 5) -> float:
    """修正次數越多，該筆紀錄所屬科目的整體加成越高（組內排名式設計）。"""
    if cap <= 0:
        return 1.0
    return min(count_so_far, cap) / float(cap)


# ----------------------------------------------------------------------------
# 3. 核心：計算科目原型字典（批次版本）
# ----------------------------------------------------------------------------

def compute_account_prototypes(
    records: List[CorrectionRecord],
    half_life_days: float = 90.0,
    count_cap: int = 5,
    now: Optional[datetime] = None,
) -> Dict[str, dict]:
    """依修正紀錄計算「科目 → 加權平均原型向量」字典。

    邊界情況：
        - records 為空 → 回傳空字典 {}
        - 單一修正 → 該科目原型向量 = 該筆向量本身
        - 向量長度不一致 → 拋出 ValueError（避免靜默產生錯誤結果）
    """
    if not records:
        return {}

    ref_now = now if now is not None else datetime.now(timezone.utc)

    groups: Dict[str, List[CorrectionRecord]] = {}
    for rec in records:
        groups.setdefault(rec.corrected_to, []).append(rec)

    result: Dict[str, dict] = {}

    for account_code, recs in groups.items():
        recs_sorted = sorted(recs, key=lambda r: _safe_sort_key(r.timestamp))

        vectors = np.array([r.summary_vector for r in recs_sorted], dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError(f"科目 {account_code} 的向量維度不一致或格式錯誤")

        expected_dim = vectors.shape[1]
        for r in recs_sorted:
            if len(r.summary_vector) != expected_dim:
                raise ValueError(
                    f"科目 {account_code} 內存在向量長度不一致："
                    f"預期 {expected_dim}，實際 {len(r.summary_vector)}"
                )

        n = len(recs_sorted)

        conf_weights = np.array(
            [max(0.0, min(1.0, r.confidence_weight)) for r in recs_sorted], dtype=np.float64,
        )
        decay_weights = np.array(
            [time_decay_weight(r.timestamp, now=ref_now, half_life_days=half_life_days)
             for r in recs_sorted], dtype=np.float64,
        )
        boost_weights = np.array(
            [count_boost_weight(k + 1, cap=count_cap) for k in range(n)], dtype=np.float64,
        )

        final_weights = conf_weights * decay_weights * boost_weights

        sum_weight = float(final_weights.sum())
        if sum_weight <= 0:
            prototype_vector = vectors.mean(axis=0)
            avg_weight = 0.0
        else:
            prototype_vector = (vectors * final_weights[:, None]).sum(axis=0) / sum_weight
            avg_weight = float(final_weights.mean())

        last_corrected = max(r.timestamp for r in recs_sorted)

        proto = AccountPrototype(
            account_code=account_code,
            prototype_vector=prototype_vector,
            correction_count=n,
            avg_weight=avg_weight,
            last_corrected=last_corrected,
            _sum_vector=(vectors * final_weights[:, None]).sum(axis=0),
            _sum_weight=sum_weight,
            _weight_history_sum=float(final_weights.sum()),
        )
        result[account_code] = proto.to_dict()
        result[account_code]["_proto_obj"] = proto

    return result


def _safe_sort_key(timestamp: str):
    try:
        return _parse_iso8601(timestamp)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------------
# 4. 增量更新版本：PrototypeStore
# ----------------------------------------------------------------------------

class PrototypeStore:
    """支援增量更新的原型向量儲存器。

    設計取捨：
        時間衰減理論上會讓「舊資料的相對權重」隨時間推移而改變。
        本實作採「近似增量」策略：add_record() 時，用當下時間為新記錄計算
        decay，並直接累加進 sum_vector / sum_weight（不重新調整舊記錄的
        decay）。因半衰期長達 90 天，短期內此近似誤差可忽略；
        若需嚴格重算（例如夜間批次），可呼叫 recompute_all()。
    """

    def __init__(self, half_life_days: float = 90.0, count_cap: int = 5):
        self.half_life_days = half_life_days
        self.count_cap = count_cap
        self._records: List[CorrectionRecord] = []
        self._prototypes: Dict[str, AccountPrototype] = {}

    def bulk_load(self, records: List[CorrectionRecord], now: Optional[datetime] = None) -> None:
        """初始化載入歷史修正紀錄，建立完整原型。"""
        self._records = list(records)
        raw = compute_account_prototypes(
            records, half_life_days=self.half_life_days, count_cap=self.count_cap, now=now
        )
        self._prototypes = {code: d["_proto_obj"] for code, d in raw.items()}

    def add_record(self, record: CorrectionRecord, now: Optional[datetime] = None) -> None:
        """增量新增一筆修正紀錄，O(1) 更新對應科目的原型向量。

        注意：假設修正紀錄依真實時間先後依序送入。若有回溯性補登
        （例如批次匯入一批舊資料），應改用 bulk_load() 或 recompute_all()。
        """
        self._records.append(record)
        ref_now = now if now is not None else datetime.now(timezone.utc)

        code = record.corrected_to
        vec = np.array(record.summary_vector, dtype=np.float64)

        existing = self._prototypes.get(code)
        prior_count = existing.correction_count if existing else 0
        new_count = prior_count + 1

        conf_w = max(0.0, min(1.0, record.confidence_weight))
        decay_w = time_decay_weight(record.timestamp, now=ref_now, half_life_days=self.half_life_days)
        boost_w = count_boost_weight(new_count, cap=self.count_cap)
        w = conf_w * decay_w * boost_w

        if existing is None:
            sum_vector = vec * w
            sum_weight = w
        else:
            if existing.prototype_vector.shape[0] != vec.shape[0]:
                raise ValueError(
                    f"科目 {code} 向量維度不一致：預期 {existing.prototype_vector.shape[0]}，"
                    f"實際 {vec.shape[0]}"
                )
            sum_vector = existing._sum_vector + vec * w
            sum_weight = existing._sum_weight + w

        prototype_vector = sum_vector / sum_weight if sum_weight > 0 else vec
        last_corrected = record.timestamp
        if existing and existing.last_corrected > last_corrected:
            last_corrected = existing.last_corrected

        avg_weight = sum_weight / new_count

        self._prototypes[code] = AccountPrototype(
            account_code=code,
            prototype_vector=prototype_vector,
            correction_count=new_count,
            avg_weight=avg_weight,
            last_corrected=last_corrected,
            _sum_vector=sum_vector,
            _sum_weight=sum_weight,
        )

    def recompute_all(self, now: Optional[datetime] = None) -> None:
        """用完整歷史記錄嚴格重算所有原型（消除增量近似誤差）。建議定期批次執行。"""
        self.bulk_load(self._records, now=now)

    def get_prototypes(self) -> Dict[str, dict]:
        """輸出對外格式字典（不含內部狀態欄位）。"""
        return {code: proto.to_dict() for code, proto in self._prototypes.items()}

    def get_prototype_vector(self, account_code: str) -> Optional[np.ndarray]:
        proto = self._prototypes.get(account_code)
        return proto.prototype_vector if proto else None


# ----------------------------------------------------------------------------
# 5. 賣方偏好統計（供 fusion_engine 之 seller_score 使用）
# ----------------------------------------------------------------------------

def compute_seller_preferences(records: List[CorrectionRecord]) -> Dict[str, Dict[str, float]]:
    """
    依修正紀錄統計「賣方統編 -> 科目偏好比例」。

    回傳格式：{seller_ban: {account_code: ratio}}，ratio 為該賣方所有
    修正紀錄中對應到該科目的比例（0-1，同一賣方所有科目比例加總為 1）。
    """
    counts: Dict[str, Dict[str, int]] = {}
    for rec in records:
        seller_stat = counts.setdefault(rec.seller_ban, {})
        seller_stat[rec.corrected_to] = seller_stat.get(rec.corrected_to, 0) + 1

    preferences: Dict[str, Dict[str, float]] = {}
    for seller_ban, stat in counts.items():
        total = sum(stat.values())
        if total <= 0:
            continue
        preferences[seller_ban] = {code: count / total for code, count in stat.items()}
    return preferences


__all__ = [
    "AccountPrototype",
    "time_decay_weight",
    "count_boost_weight",
    "compute_account_prototypes",
    "PrototypeStore",
    "compute_seller_preferences",
]
