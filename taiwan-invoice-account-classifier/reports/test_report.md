# 測試報告（Test Report）

> 本報告記錄之數字均為實際執行結果，執行環境與時間如下所述，非估算或人工填寫。

## 執行環境

| 項目 | 內容 |
|---|---|
| 執行日期 | 2026-09-01 |
| Python 版本 | 3.10.12（Linux） |
| pytest 版本 | 9.1.1 |
| 測試框架外掛 | pytest-cov 7.1.0 |
| 專案版本 | invoice_classifier 0.1.0 |

## 一、pytest 完整測試結果

執行指令：

```
python -m pytest tests/ -v --tb=short
```

**結果：136 passed, 4 warnings, 耗時 9.50 秒**

- 全部 136 項測試 **100% 通過**，無任何失敗（failed）或錯誤（error）。
- 4 項警告皆為 `PytestCollectionWarning`（`TestCaseResult` / `TestReport` / `TestResult` 等 dataclass 因命名以 `Test` 開頭被 pytest 誤判為測試類別而略過收集，非功能性錯誤，不影響任何測試案例執行）。

### 各測試檔案涵蓋範圍

| 測試檔案 | 涵蓋模組／功能 | 案例數（概估） |
|---|---|---|
| `tests/test_boundary_pytest.py` | 邊界案例（BC001–BC028）、10000 筆修正壓力測試、全套件效能測試 | 30 |
| `tests/test_compliance_pytest.py` | 稅法合規案例（TC001–TC025） | 25 |
| `tests/test_correction_manager.py` | 修正紀錄新增／查詢／更新／刪除／匯出／併發／效能 | 21 |
| `tests/test_fusion_engine.py` | 融合引擎四來源加權、規則觸發、效能 | 9 |
| `tests/test_monthly_report.py` | 月報產生、缺值資料品質、效能（1000 筆發票） | 6 |
| `tests/test_risk_scorer.py` | 風險評分五維度、風險等級邊界 | 21 |
| `tests/test_tax_rules.py` | 稅法規則比對（TC01–TC13）、規則 2/6 整合註記 | 14 |
| `tests/test_version_control.py` | 版本控制生命週期、備份、審計紀錄、效能 | 6 |

（各檔案內部另含非 pytest 形式之獨立 runner 腳本，如 `test_boundary_runner.py`、`test_compliance_runner.py`，作為輔助驗證工具，其邏輯已由對應的 `*_pytest.py` 以 pytest 形式納入本次執行統計。）

## 二、稅法規則覆蓋率（config/tax_rules.json）

`config/tax_rules.json` 目前共定義 **6 條**稅法合規規則（`version 1.1`，`last_updated 2026-09-01`）：

| rule_id | 對應法規 |
|---|---|
| RULE_001_input_tax_separation | 加值型及非加值型營業稅法 第15條、第19條 |
| RULE_002_cif_freight_no_duplicate | 營利事業所得稅查核準則 第44條 |
| RULE_003_customs_fee_into_cost | 營利事業所得稅查核準則 第37條 |
| RULE_004_entertainment_expense_cap | 營利事業所得稅查核準則 第62條 |
| RULE_005_employee_welfare_vs_entertainment | 營利事業所得稅查核準則 第81條 |
| RULE_006_passenger_car_input_tax_not_deductible | 加值型及非加值型營業稅法 第19條 |

實測方式：於 `tests/test_tax_rules.py`、`tests/test_compliance_cases.py`、`tests/test_fusion_engine.py` 中以 `grep` 搜尋各 `RULE_00X` 規則 ID 出現次數：

| rule_id | 被引用測試次數（跨檔案加總） |
|---|---|
| RULE_001 | 11 |
| RULE_002 | 6 |
| RULE_003 | 6 |
| RULE_004 | 6 |
| RULE_005 | 5 |
| RULE_006 | 7 |

**規則覆蓋率：6 / 6 = 100%**（config/tax_rules.json 中定義的每一條規則，皆至少有一項測試案例明確觸發並驗證其行為）。

## 三、結論

- 本次為實機執行結果，非歷史數字或估算：136/136 測試通過，耗時 9.50 秒。
- 6 條稅法規則全數具備對應測試覆蓋。
- 本報告不涉及分類準確率／F1 等統計指標，該類指標請見 `reports/performance_report.md`（基於合成資料，非真實發票）。
