# -*- coding: utf-8 -*-
"""
產生帶標籤的合成評估集（僅供展示系統運作方式，非真實發票資料）。

輸出：examples/eval_synthetic_invoices.csv，欄位包含 invoice_id, buyer_ban,
buyer_name, seller_ban, seller_name, summary, amount, trade_condition,
invoice_date, ground_truth_account（依關鍵字樣板規則分派之合成標籤，
僅用於本專案自我評估，非真實發票、非真實企業交易紀錄）。
"""
import csv
import random

random.seed(20260901)

TEMPLATES = {
    "5121": ["進貨原物料採購", "商品進貨一批", "原料進貨結算"],
    "5122": ["進貨運輸整理費用", "進貨相關雜項費用"],
    "1268": ["營業稅5%進項稅額", "加值型營業稅VAT稅額"],
    "5213": ["客戶交際應酬餐敘", "業務往來送禮致贈", "客戶宴客接待費用"],
    "5214": ["市區計程車交通費", "油資加油費用報支"],
    "5215": ["出差住宿及機票費用", "國外差旅交通住宿費"],
    "5216": ["辦公文具耗材採購", "辦公用品採購一批"],
    "5217": ["郵電及通訊費用", "電話網路通訊費"],
    "5218": ["水電費用繳納", "電費水費合併繳納"],
    "5219": ["廣告行銷企劃費用", "廣告刊登行銷費"],
    "5221": ["貨物運送運費", "國內運送運費支出"],
    "5222": ["貨物保險費支出", "運輸保險費用"],
    "5226": ["員工尾牙聚餐福利", "員工旅遊教育訓練", "員工聚餐福利費用"],
    "5253": ["設備折舊提列費用", "機器設備折舊攤提"],
}

SELLERS = [
    ("測試商行", "A"), ("示範企業社", "B"), ("展示有限公司", "C"),
    ("樣本貿易股份有限公司", "D"), ("假設科技有限公司", "E"),
    ("演練工程行", "F"), ("模擬國際貿易", "G"), ("測試物流有限公司", "H"),
    ("範例餐飲股份有限公司", "I"), ("樣板顧問有限公司", "J"),
]

N = 220


def random_ban(seed_idx: int) -> str:
    return f"{9000000000 + seed_idx}"[:10]


def main():
    rows = []
    codes = list(TEMPLATES.keys())
    for i in range(1, N + 1):
        code = codes[(i - 1) % len(codes)]
        summary = random.choice(TEMPLATES[code])
        seller_name, tag = random.choice(SELLERS)
        amount = round(random.uniform(500, 300000), 0)
        trade_condition = random.choice(["", "", "", "CIF", "FOB"])
        month = random.randint(1, 8)
        day = random.randint(1, 28)
        rows.append({
            "invoice_id": f"EVAL-{i:04d}",
            "buyer_ban": "1234567890",
            "buyer_name": "評估用範例股份有限公司",
            "seller_ban": random_ban(i),
            "seller_name": f"{seller_name}{tag}",
            "summary": summary,
            "amount": amount,
            "trade_condition": trade_condition,
            "invoice_date": f"2026-{month:02d}-{day:02d}",
            "ground_truth_account": code,
        })

    random.shuffle(rows)
    out_path = "examples/eval_synthetic_invoices.csv"
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"produced {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
