# -*- coding: utf-8 -*-
"""
端對端示範腳本：examples/run_classification_demo.py

用途：
    讀取 examples/sample_invoices.csv（35 筆完全虛構之範例發票），
    透過 InvoiceClassifier 完整流程（前處理 -> 母版模型編碼 -> 個人化
    -> 融合引擎 -> 稅務合規檢查 -> 說明產生器）逐筆預測會計科目，
    並印出人類可讀的預測結果與說明摘要。

重要聲明：
    本示範腳本與整體系統為「稅務輔助與風險提示工具」，預測結果與風險
    標示僅供參考，非稅務或法律意見；高風險案例請依系統標示轉交
    企業會計／稅務顧問進行人工覆核，不應直接作為申報依據。

執行方式：
    python examples/run_classification_demo.py

注意：
    本示範腳本使用「確定性假向量編碼器」（hashlib.md5-based，非正式
    訓練後的語意模型）以便在不需下載模型檔或連網的情況下即可執行，
    目的僅在展示完整資料流與輸出格式；正式使用前請依 README 指示
    執行 training/ 下的腳本，以 src/invoice_classifier/master_model.py
    的 train_from_csv() 訓練並產出正式 master_model.pkl。
"""

from __future__ import annotations

import csv
import hashlib
import sys
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
    """確定性假向量產生器（僅供示範腳本使用，非正式訓練後模型）。"""
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


def load_sample_invoices(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    print("=" * 78)
    print("台灣發票會計科目分類機器人 —— 端對端示範")
    print("=" * 78)
    print(
        "本工具為「稅務輔助與風險提示工具」，預測結果與風險標示僅供參考，\n"
        "非稅務或法律意見；請勿以此自動判定使用者有無逃漏稅意圖。\n"
        "高風險（high/critical）案例請依標示轉交企業會計／稅務顧問人工覆核。\n"
    )

    csv_path = Path(__file__).resolve().parent / "sample_invoices.csv"
    rows = load_sample_invoices(csv_path)
    print(f"讀取範例發票 {len(rows)} 筆（完全虛構資料，僅供示範）：{csv_path}\n")

    master_model = _build_demo_master_model()
    classifier = InvoiceClassifier(master_model=master_model)

    manual_review_count = 0
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

        result = classifier.predict(invoice)
        d = result.to_dict()
        pred = d["prediction"]
        expl = d["explanation"]

        risk_level = expl["risk_level"]
        if risk_level in ("high", "critical"):
            manual_review_count += 1

        print("-" * 78)
        print(f"[{row['invoice_id']}] {row['seller_name']} — {row['summary']}（{row['amount']} 元）")
        print(f"  預測科目：{pred['account_code']}（{pred['account_name']}）"
              f"　置信度：{pred['confidence']:.2f}")
        print(f"  風險等級：{risk_level}"
              + ("　⚠ 建議轉人工覆核" if risk_level in ("high", "critical") else ""))
        if expl.get("compliance_notes"):
            for note in expl["compliance_notes"]:
                print(f"    - {note}")
        summary_text = expl.get("human_readable_summary") or ""
        if summary_text:
            print(f"  說明摘要：{summary_text.splitlines()[0]}")

    print("-" * 78)
    print(f"\n示範結束。共 {len(rows)} 筆，其中 {manual_review_count} 筆被標示為建議轉人工覆核。")
    print(
        "提醒：本示範使用簡化的確定性假向量編碼器，僅展示資料流與輸出格式，\n"
        "不代表正式訓練後模型的分類品質。正式使用前請參閱 README 指示，\n"
        "執行 training/ 下的腳本以訓練並產出正式 master_model.pkl（不隨本專案提供）。"
    )


if __name__ == "__main__":
    main()
