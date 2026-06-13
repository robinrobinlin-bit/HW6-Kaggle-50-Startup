# HW6 Kaggle 50 Startups Profit Prediction

![HW6 Report Preview](image.png)

## 專案簡介

本專案使用 Kaggle 50 Startups 資料集，依照 CRISP-DM 流程建立企業獲利預測模型，並透過 Multiple Linear Regression 與特徵選擇方法分析企業獲利的主要驅動因子。

本專案不只建立預測模型，也整理出可解釋的商業洞察，協助企業了解研發支出、行銷支出、行政支出與地區因素對獲利能力的影響。

---

## 專案目標

1. 建立 50 Startups 企業獲利預測模型。
2. 使用 Multiple Linear Regression 預測 Profit。
3. 比較不同特徵組合的模型表現。
4. 使用 5 種特徵選擇方法找出重要特徵。
5. 依照 CRISP-DM 流程完成完整資料科學專案。
6. 輸出模型、圖表、報告與技術白皮書。

---

## CRISP-DM 專案流程

| 階段 | 說明 |
|---|---|
| Business Understanding | 定義企業獲利預測問題 |
| Data Understanding | 檢查資料欄位、型態、缺失值與相關性 |
| Data Preparation | One-Hot Encoding、切分訓練集與測試集 |
| Modeling | 建立多元線性迴歸模型 |
| Evaluation | 使用 RMSE、R-squared 評估模型 |
| Deployment | 輸出模型、圖表與報告 |

---

## 資料集欄位說明

| 欄位名稱 | 中文說明 | 角色 |
|---|---|---|
| R&D Spend | 研發支出 | 輸入特徵 |
| Administration | 行政支出 | 輸入特徵 |
| Marketing Spend | 行銷支出 | 輸入特徵 |
| State | 公司所在州別 | 類別特徵 |
| Profit | 公司獲利 | 預測目標 |

---

## 使用技術

| 類別 | 工具 |
|---|---|
| 程式語言 | Python |
| 資料處理 | pandas、numpy |
| 機器學習 | scikit-learn |
| 視覺化 | matplotlib、seaborn |
| 模型儲存 | joblib |
| 專案流程 | CRISP-DM |

---

## 模型方法

本專案使用：

```text
Multiple Linear Regression 多元線性迴歸