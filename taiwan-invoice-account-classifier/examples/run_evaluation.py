# -*- coding: utf-8 -*-
"""
examples/run_evaluation.py

⚠️ 重要聲明：本腳本所有指標均基於程式產生之合成評估資料
（examples/eval_synthetic_invoices.csv），非真實發票、非真實企業交易紀錄，
僅供展示系統運作方式，不代表實際生產環境準確率。

用途：
    讀取 examples/eval_synthetic_invoices.csv（220 筆合成標籤發票），
    以與 examples/run_classification_demo.py 相同的「確定性假向量編碼器」
    建立示範用 MasterModel，逐筆呼叫 InvoiceClassifier.predict()，
    實測（非估算）：
        - Top-1 準確率
        - Macro-F1
        - Top-3 準確率（依 fusion_raw 中 top_k_candidates 判斷）
        - 平均 / P95 預測時間（毫秒，使用 time.perf_counter() 對每筆呼叫實測）

執行方式：
    python examples/run_evaluation.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from invoice_classifier.data_models import InvoiceData  # noqa: E402
from invoice_classifier.master_model import MasterModel  # noqa: E402
from invoice_classifier.classifier import InvoiceClassifier  # noqa: E402

ACCOUNT_CODES = [
    "5121", "5122", "1268", "5213", "5214", "5215", "5216",
    "5217", "5218", "5219", "5221", "5222", "5226", "5253",
]
ACCOUNT_NAMES = {
    "5121": "進貨", "5122": "進貨費用", "1268": "進項稅額", "5213": "交際費",
    "5214": "交通費", "5215": "差旅費", "5216": "辦公費", "5217": "郵電費",
    "5218": "水電費", "5219": "廣告費", "5221": "運費", "5222": "保險費",
    "5226": "職工福利", "5253": "折舊費用",
}
_KEYWORD_PROTOTYPES = {
    "5121": ["進貨", "貨物", "商品", "原物料", "原料"],
    "5122": ["進貨費用", "報關", "通關"],
    "1268": ["稅", "營業稅", "VAT", "稅額"],
    "5213": ["交際", "應酬", "宴客", "送禮", "客戶"],
    "5214": ["交通", "計程車", "油資"],
    "5215": ["差旅", "出差", "住宿", "機票"],
    "5216": ["辦公", "文具", "耗材"],
    "5217": ["郵電", "通訊", "電話"],
    "5218": ["水電", "電費", "水費"],
    "5219": ["廣告", "行銷", "企劃"],
    "5221": ["運費", "運送"],
    "5222": ["保險費", "保險"],
    "5226": ["員工", "旅遊", "聚餐", "尾牙", "福利", "教育訓練"],
    "5253": ["折舊", "設備"],
}
_DIM = 64


def _fake_embed(text: str) -> np.ndarray:
    vec = np.zeros(_DIM, dtype=float)
    text = text or ""
    chars = list(text)
    grams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)] or (chars or [" "])
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8", errors="ignore")).hexdigest(), 16) % _DIM
        vec[h] += 1.0
    return vec


class _DemoEmbedder:
    def encode(self, texts):
        return np.array([_fake_embed(t) for t in texts])


def _build_demo_master_model() -> MasterModel:
    account_vectors: Dict[str, np.ndarray] = {}
    for code in ACCOUNT_CODES:
        kws = _KEYWORD_PROTOTYPES.get(code, [ACCOUNT_NAMES.get(code, code)])
        vecs = [_fake_embed(kw) for kw in kws]
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        account_vectors[code] = avg / norm if norm > 1e-12 else avg
    return MasterModel(
        account_codes=ACCOUNT_CODES,
        account_vectors=account_vectors,
        account_names=ACCOUNT_NAMES,
        embedder=_DemoEmbedder(),
    )


def load_eval_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue  # label 未出現於 true 或 pred，略過（避免除以零造成誤導）
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return statistics.mean(f1s) if f1s else 0.0


def main() -> None:
    print("=" * 78)
    print("⚠️  以下所有指標均基於程式產生之合成評估資料，非真實發票資料，")
    print("    僅供展示系統運作方式，不代表實際生產環境準確率。")
    print("=" * 78)

    csv_path = Path(__file__).resolve().parent / "eval_synthetic_invoices.csv"
    rows = load_eval_rows(csv_path)
    print(f"讀取合成評估集 {len(rows)} 筆：{csv_path}\n")

    master_model = _build_demo_master_model()
    classifier = InvoiceClassifier(master_model=master_model)

    y_true: List[str] = []
    y_pred: List[str] = []
    top3_hits = 0
    latencies_ms: List[float] = []

    for row in rows:
        invoice = InvoiceData(
            invoice_id=row["invoice_id"],
            buyer_ban=row["buyer_ban"],
            seller_ban=row["seller_ban"],
            summary=row["summary"],
            amount=float(row["amount"]) if row.get("amount") else None,
            trade_condition=row["trade_condition"] or None,
            invoice_date=row["invoice_date"],
        )
        gt = row["ground_truth_account"]

        t0 = time.perf_counter()
        result = classifier.predict(invoice, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        pred_code = result.prediction.account_code
        y_true.append(gt)
        y_pred.append(pred_code)

        top3_codes = [c["account_code"] for c in result.fusion_raw["explanation"]["top_k_candidates"]]
        if gt in top3_codes:
            top3_hits += 1

    n = len(rows)
    top1_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    top3_acc = top3_hits / n
    labels = sorted(set(y_true) | set(y_pred))
    mf1 = macro_f1(y_true, y_pred, labels)

    latencies_ms_sorted = sorted(latencies_ms)
    avg_ms = statistics.mean(latencies_ms)
    p95_idx = min(len(latencies_ms_sorted) - 1, int(round(0.95 * (len(latencies_ms_sorted) - 1))))
    p95_ms = latencies_ms_sorted[p95_idx]

    print(f"樣本數（N）              : {n}")
    print(f"Top-1 準確率（Accuracy）  : {top1_acc:.4f} ({sum(1 for t,p in zip(y_true,y_pred) if t==p)}/{n})")
    print(f"Macro-F1                 : {mf1:.4f}")
    print(f"Top-3 準確率              : {top3_acc:.4f} ({top3_hits}/{n})")
    print(f"平均預測時間（ms）        : {avg_ms:.4f}")
    print(f"P95 預測時間（ms）        : {p95_ms:.4f}")
    print(f"最小/最大預測時間（ms）   : {min(latencies_ms):.4f} / {max(latencies_ms):.4f}")

    # 依科目分類的混淆情形，供報告附錄使用
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for t, p in zip(y_true, y_pred):
        if t == p:
            per_label[t]["tp"] += 1
        else:
            per_label[t]["fn"] += 1
            per_label[p]["fp"] += 1

    result_json = {
        "disclaimer": "以下所有指標均基於程式產生之合成評估資料，非真實發票資料，僅供展示系統運作方式，不代表實際生產環境準確率。",
        "data_source": "synthetic (examples/eval_synthetic_invoices.csv, generated by examples/generate_eval_set.py)",
        "n_samples": n,
        "top1_accuracy": round(top1_acc, 4),
        "macro_f1": round(mf1, 4),
        "top3_accuracy": round(top3_acc, 4),
        "latency_ms": {
            "mean": round(avg_ms, 4),
            "p95": round(p95_ms, 4),
            "min": round(min(latencies_ms), 4),
            "max": round(max(latencies_ms), 4),
        },
        "per_label": {k: dict(v) for k, v in per_label.items()},
    }
    out_json = Path(__file__).resolve().parent.parent / "reports" / "eval_results.json"
    out_json.write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整結果已寫入：{out_json}")


if __name__ == "__main__":
    main()
