# -*- coding: utf-8 -*-
"""
test_risk_scorer.py — 發票稅務風險評分測試案例（整合自 M4-2 test_risk_scorer.py）

改寫重點：
    - import 路徑改為 invoice_classifier.tax_compliance.assess_invoice_risk
    - 統編改用 10 碼數字（配合 data_models 驗證規則，原案例為示範性 8 碼）

涵蓋 low / medium / high / critical 四種風險等級，
以及五大風險維度（科目、關鍵字、貿易條件、金額異常、憑證）之單獨與組合觸發情境。
"""

from __future__ import annotations

import time

import pytest

from invoice_classifier.tax_compliance import assess_invoice_risk

SELLER = "1234567890"
BUYER = "0987654321"

CASES = []


def case(name, summary, expected_level, amount=10000.0, trade_condition=None, account_code=None):
    CASES.append({
        "name": name, "summary": summary, "amount": amount,
        "trade_condition": trade_condition, "account_code": account_code,
        "expected_level": expected_level,
    })


# --- LOW 風險（0.0-0.3）-----------------------------------------------------
case("低風險_一般進貨", "採購原物料一批", "low", amount=48000.0, account_code="5121")
case("低風險_辦公用品", "購買文具用品一批", "low", amount=2500.0, account_code="5216")
case("低風險_FOB含運費單據", "採購商品，含運費單據", "low", amount=30000.0,
     trade_condition="FOB", account_code="5121")

# --- MEDIUM 風險（0.3-0.5）---------------------------------------------------
case("中風險_交際費科目", "客戶餐敘費用", "medium", amount=4500.0, account_code="5213")
case("中風險_差旅費科目", "出差住宿費用", "medium", amount=12000.0, account_code="5215")
case("低風險_關鍵字交際單一命中", "業務交際費用支出", "low", amount=5000.0)
case("低風險_FOB無運費單據", "採購商品一批", "low", amount=20000.0, trade_condition="FOB")
case("中風險_交際加員工雙關鍵字", "員工交際應酬費用", "medium", amount=5000.0)

# --- HIGH 風險（0.5-0.7）-----------------------------------------------------
case("高風險_運費科目", "海外貨物運送運費", "high", amount=9000.0, account_code="5221")
case("高風險_保險費科目", "貨物運輸保險費", "high", amount=4500.0, account_code="5222")
case("中風險_CIF含運費關鍵字", "CIF條件進口貨物運費", "medium", amount=15000.0, trade_condition="CIF")
case("高風險_收據憑證加交際關鍵字", "交際應酬收據", "high", amount=3000.0)

# --- CRITICAL 風險（0.7-1.0）-------------------------------------------------
case("極高風險_進項稅額未分離", "營業稅5%進項稅額", "critical", amount=5000.0, account_code="1268")
case("高風險_金額為零", "商品進貨", "high", amount=0.0, account_code="5121")
case("高風險_金額為負數", "折讓沖銷", "high", amount=-1000.0, account_code="5121")
case("極高風險_金額為零加交際關鍵字", "交際應酬費用", "critical", amount=0.0, account_code="5213")
case("極高風險_自用轎車進項稅額", "自用乘人小汽車進項稅額扣抵", "critical", amount=800000.0, account_code="1268")
case("極高風險_多重關鍵字疊加", "員工自用交際應酬收據估單", "critical", amount=6000.0)
case("極高風險_CIF運費加估單", "CIF運費保險費估單證明單", "critical", amount=20000.0, trade_condition="CIF")

# --- 邊界 / 特殊情境 ---------------------------------------------------------
case("邊界_金額異常超同業兩倍", "一般進貨", "medium", amount=150000.0, account_code="5121")
case("邊界_金額缺失不強行扣分", "採購耗材", "low", amount=None, account_code="5216")
case("邊界_無任何觸發", "一般費用", "low", amount=100.0)


@pytest.mark.parametrize("c", CASES, ids=[c["name"] for c in CASES])
def test_risk_level(c):
    invoice = {
        "summary": c["summary"], "seller_ban": SELLER, "buyer_ban": BUYER,
        "amount": c["amount"], "trade_condition": c["trade_condition"],
    }
    start = time.perf_counter()
    result = assess_invoice_risk(invoice, account_code=c["account_code"])
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1, f"{c['name']} 評估耗時超過 0.1 秒：{elapsed:.4f}s"
    assert result["risk_level"] == c["expected_level"], (
        f"{c['name']} 失敗：score={result['risk_score']:.3f}, "
        f"level={result['risk_level']}（預期 {c['expected_level']}）"
    )
    if result["risk_level"] in ("high", "critical"):
        assert result["manual_review_recommended"] is True


def test_manual_review_flag_false_for_low_risk():
    result = assess_invoice_risk({"summary": "一般費用", "amount": 100.0}, account_code=None)
    assert result["risk_level"] == "low"
    assert result["manual_review_recommended"] is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
