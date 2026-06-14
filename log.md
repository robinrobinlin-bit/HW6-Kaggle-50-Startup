# Project Implementation Log: 50 Startups Profit Prediction (CRISP-DM Workflow)

This document provides a highly detailed, step-by-step technical log of the development work completed for **HW6: 50 Startups Profit Prediction**. The project is structured around the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to build a robust, reproducible, and deployable Machine Learning solution.

---

## 1. Project Directory Structure & Created Files

The project has been organized into a modular structure supporting clean separation of concerns, automated pipeline execution, interactive deployment, and professional reporting.

```text
hw6_50_startups_profit_prediction/
├── data/
│   └── 50_Startups.csv                # Raw dataset (50 rows, 5 features)
├── docs/
│   └── crisp_dm_workflow.drawio       # Editable XML file of the workflow diagram
├── outputs/
│   ├── figures/                       # Cohesive dark "Tech Blue" theme diagnostic plots
│   │   ├── actual_vs_predicted.png    # Fitted values vs Actual values (with y=x reference)
│   │   ├── correlation_heatmap.png    # Pearson correlation matrix (including state flags)
│   │   ├── feature_selection_comparison.png  # Heatmap of feature rankings across 5 methods
│   │   ├── feature_selection_performance.png # Dual-axis plot of RMSE vs R-squared
│   │   ├── r2_by_features.png         # Train R2 vs Adjusted R2 comparisons
│   │   ├── residual_plot.png          # Residual diagnostics verifying homoscedasticity
│   │   └── rmse_by_features.png       # Test RMSE across different feature subsets
│   ├── models/                        # Serialized artifacts for deployment
│   │   ├── feature_columns.pkl        # List of features used in the best model
│   │   └── startup_profit_model_v2.pkl# Joblib serialized best LinearRegression model
│   └── reports/                       # Auto-generated reports and compiled deliverables
│       ├── data_understanding_report.txt     # Descriptive stats, missing values, shape
│       ├── feature_selection_ranking.csv     # Tabular ranks of features
│       ├── feature_selection_summary.txt     # Comprehensive text summary on feature selection
│       ├── model_comparison.csv       # Metrics (RMSE, MAE, R2, Adj-R2) for 5 combinations
│       ├── model_comparison.txt       # Formatted model comparison table
│       └── startup_profit_prediction_handout.pdf # Compiled ReportLab multi-page A4 PDF booklet
├── src/                               # Source code modules
│   ├── data_preprocessing.py          # Data ingestion, analysis report, OHE, train-test split
│   ├── evaluation.py                  # Model selector, coefficient extractor, and serializer
│   ├── feature_selection.py           # Evaluates SFS, RFE, SelectKBest, Lasso, RF
│   ├── generate_pdf.py                # ReportLab script generating multi-page illustrated PDF
│   ├── main.py                        # Central pipeline orchestrator
│   ├── model_training.py              # Evaluates 5 preset models on the test set
│   └── visualization.py               # Generates 7 dark-themed matplotlib/seaborn figures
├── HW6.md                             # Detailed project summary report
├── README.md                          # Quickstart guide, formulas, installation instructions
├── requirements.txt                   # Dependency list
├── run_project.py                     # Entry point to execute the pipeline
├── streamlit_app.py                   # Streamlit web-based dashboard and live predictor
└── image.png                          # Hand-drawn styled A4 CRISP-DM overview cheat sheet
```

---

## 2. Step-by-Step Implementation Details

### Step 1: Requirements Definition & Dependency Setup
*   Defined the python environment requirements in [requirements.txt](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/requirements.txt):
    *   `pandas` & `numpy` (data structure operations)
    *   `scikit-learn` (machine learning algorithms, feature selection wrappers, metrics)
    *   `matplotlib` & `seaborn` (visualization engine)
    *   `joblib` (model persistence)
    *   `streamlit` (dashboard and deployment GUI framework)
    *   `reportlab` (dynamic A4 PDF compiling library used in `src/generate_pdf.py` for print-ready hand-outs)

---

### Step 2: Data Understanding (CRISP-DM Phase 2)
*   **Source File**: [src/data_preprocessing.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/data_preprocessing.py) (`load_and_check_raw_data`)
*   **Inputs**: `data/50_Startups.csv`
*   **Logic**:
    1. Loads the raw CSV into a pandas DataFrame.
    2. Performs data inspection: checks number of rows (50) and columns (5).
    3. Checks data types: `R&D Spend` (float64), `Administration` (float64), `Marketing Spend` (float64), `State` (object/categorical), and `Profit` (float64).
    4. Computes descriptive statistics (min, max, mean, standard deviation, percentiles) for all columns.
    5. Performs data hygiene checks: searches for missing values (`df.isnull().sum()`) and duplicate rows (`df.duplicated().sum()`), confirming 0 missing values and 0 duplicates.
    6. Extracts the numeric Pearson correlation matrix.
    7. Outputs all metrics into a formal text file: `outputs/reports/data_understanding_report.txt`.

---

### Step 3: Data Preparation (CRISP-DM Phase 3)
*   **Source File**: [src/data_preprocessing.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/data_preprocessing.py) (`preprocess_data`, `split_data`)
*   **Logic**:
    1. **Dummy Variable Trap Prevention**: The categorical column `State` (contains New York, Florida, California) was processed via One-Hot Encoding (`pd.get_dummies`).
       * *Rationale*: Putting all three dummies along with an intercept into a linear model causes perfect multicollinearity (since $State_{NY} + State_{FL} + State_{CA} = 1$). To avoid this trap, **`California` was dropped as the baseline**. If `New York = 0` and `Florida = 0`, it implicitly represents `California`.
    2. **Feature Matrix Alignment**: Features were ordered into a structured matrix $X$ with the columns:
       `['R&D Spend', 'Marketing Spend', 'New York', 'Florida', 'Administration']`
       and target vector $y$: `Profit`.
    3. **Train-Test Partitioning**: Split the dataset into an **80% training set (40 samples)** and **20% testing set (10 samples)** using `sklearn.model_selection.train_test_split`.
       * Specified a seed of `random_state=42` to guarantee exact replication of results across runs.

---

### Step 4: Modeling (CRISP-DM Phase 4)
*   **Source File**: [src/model_training.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/model_training.py) (`train_and_compare_models`)
*   **Logic**:
    1. Defined 5 predetermined feature combinations representing growing model complexity:
       * **Model 1**: `['R&D Spend']` (1 Feature)
       * **Model 2**: `['R&D Spend', 'Marketing Spend']` (2 Features)
       * **Model 3**: `['R&D Spend', 'Marketing Spend', 'New York']` (3 Features)
       * **Model 4**: `['R&D Spend', 'Marketing Spend', 'New York', 'Florida']` (4 Features)
       * **Model 5**: `['R&D Spend', 'Marketing Spend', 'New York', 'Florida', 'Administration']` (5 Features)
    2. Trained individual `sklearn.linear_model.LinearRegression` estimators on the training partition.
    3. Generated predictions on the test partition.
    4. Evaluated performance metrics:
       * **MSE (Mean Squared Error)**
       * **RMSE (Root Mean Squared Error)**: $\sqrt{MSE}$
       * **MAE (Mean Absolute Error)**
       * **R-squared ($R^2$)**: $1 - \frac{SS_{res}}{SS_{tot}}$
       * **Adjusted R-squared**: $1 - (1 - R^2) \frac{n - 1}{n - p - 1}$ (where $n = 10$ test samples, and $p$ is the feature count).
    5. Logged all comparisons to `outputs/reports/model_comparison.csv` and `outputs/reports/model_comparison.txt`.

---

### Step 5: Evaluation & Feature Selection Cross-Analysis (CRISP-DM Phase 5)
*   **Source File**: [src/feature_selection.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/feature_selection.py) (`perform_feature_selection`, `get_ranks`)
*   **Logic**:
    Evaluated five different algorithmic techniques to rank features from rank 1 (highest importance) to rank 5 (lowest importance):
    1. **Sequential Forward Selection (SFS)**:
       * Implemented a manual greedy forward selector. It starts with an empty feature set, tests each candidate feature individually, and selects the one that yields the highest training set $R^2$. It repeats this process to select the next feature, recording the entry sequence to derive rankings.
    2. **Recursive Feature Elimination (RFE)**:
       * Implemented using scikit-learn's `RFE` wrapped around a `LinearRegression` model, iteratively eliminating the feature with the smallest coefficient weight until one remains.
    3. **SelectKBest**:
       * Ranked variables using the univariate $F$-regression score (`f_regression`), which scores linear correlations between each feature and the target.
    4. **Lasso (L1) Regression**:
       * Standardized input features using `StandardScaler` to ensure scale-invariance. Trained a `Lasso` estimator with $\alpha = 100.0$. Features were ranked based on the magnitude of their absolute coefficients ($|\beta_j|$).
    5. **Random Forest Feature Importance**:
       * Trained a `RandomForestRegressor` with $100$ estimators and `random_state=42`, ranking features by mean decrease in impurity.
    6. **Consolidation**:
       * Computed the **Average Rank** across all 5 methods.
       * Saved results to `outputs/reports/feature_selection_ranking.csv`.
       * Wrote a technical analysis explaining that adding features beyond `R&D Spend` and `Marketing Spend` yields model overfitting (proven by deteriorating Adjusted $R^2$ and rising test RMSE), saving it to `outputs/reports/feature_selection_summary.txt`.

---

### Step 6: Best Model Selection & Persistence
*   **Source File**: [src/evaluation.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/evaluation.py) (`evaluate_and_save_best_model`)
*   **Logic**:
    1. Identified **Model 2: 2 Features (R&D Spend + Marketing Spend)** as the optimal, parsimonious model balancing simplicity and performance.
       * *Test Metrics*: RMSE of `8206.3288`, $R^2$ of `0.9168` (Training $R^2$ is `0.9519`).
    2. Extracted the trained parameters:
       * **Intercept ($\beta_0$)**: `50286.8118`
       * **R&D Spend coefficient ($\beta_1$)**: `0.8056`
       * **Marketing Spend coefficient ($\beta_2$)**: `0.0272`
       * **Linear Formula**:
         $$\text{Profit} = 50286.8118 + 0.8056 \times (\text{R\&D Spend}) + 0.0272 \times (\text{Marketing Spend})$$
    3. Serialized the trained model object to `outputs/models/startup_profit_model_v2.pkl` and the list of matching features to `outputs/models/feature_columns.pkl` using `joblib.dump`.

---

### Step 7: Data Visualization
*   **Source File**: [src/visualization.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/visualization.py) (`generate_plots`, `apply_tech_blue_theme`)
*   **Logic**:
    1. Designed a custom matplotlib style function (`apply_tech_blue_theme`) to override defaults, applying a high-end dark theme:
       * Deep Navy background (`#0D1B2A`), Dark Slate axes panels (`#1B263B`), Off-white fonts (`#E0E1DD`), Slate-blue grids (`#415A77`), and neon accent colors (Neon Cyan `#00F5D4`, Light Blue `#00B4D8`, Magenta `#FF007F`).
    2. Rendered and saved **7 distinct plots** to `outputs/figures/`:
       * `rmse_by_features.png`: Line plot tracing RMSE values over feature sizes, highlighting the optimal feature count in magenta.
       * `r2_by_features.png`: Double line plot comparing train $R^2$ and Adjusted $R^2$ values to demonstrate where overfitting occurs.
       * `feature_selection_performance.png`: Dual-axis plot plotting both RMSE (left y-axis, cyan) and $R^2$ (right y-axis, blue) in a single visual.
       * `feature_selection_comparison.png`: Heatmap of ranks, visually aligning feature rankings from all 5 selection methods.
       * `actual_vs_predicted.png`: Scatter plot mapping actual values vs model predictions on the test set, overlaying a dashed line representing the ideal fit ($y = x$).
       * `residual_plot.png`: Scatter plot graphing residuals ($y_{actual} - y_{pred}$) against predicted values to verify linear regression assumptions.
       * `correlation_heatmap.png`: Matrix heatmap plotting correlation coefficients across all inputs, including dummy variables for states.

---

### Step 8: PDF Handout Compilation
*   **Source File**: [src/generate_pdf.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/src/generate_pdf.py) (`build_pdf_handout`, `register_chinese_font`)
*   **Logic**:
    1. Utilized Python's `reportlab` library to build a multi-page A4 PDF booklet.
    2. Implemented font detection (`register_chinese_font`) to register native Windows Chinese TrueType Fonts (`msjh.ttc` or `msyh.ttc`) to support Chinese rendering (falling back to standard Helvetica if fonts are missing).
    3. Programmed structured layout flowables:
       * **Page 1 (Cover)**: Title page introducing the CRISP-DM project overview, embedded with the correlation matrix.
       * **Page 2 (Model Comparisons)**: Paragraphs detailing linear regression results, a formatted `Table` summarizing metrics (MSE, RMSE, MAE, R2, Adj-R2) for all 5 combinations, and the RMSE trend plot.
       * **Page 3 (Feature Selection)**: Section summarizing the 5 ranking methods, complete with a structured rankings table and the ranking heatmap.
       * **Page 4 (Formulas & Diagnostics)**: Formula block highlighted in a code style layout, regression coefficient interpretation, and vertical alignment of prediction scatter and residual diagnostic plots.
       * **Page 5 (Infographic)**: Full-page embedding of the hand-drawn A4 cheat sheet (`image.png`) summarizing the entire CRISP-DM workflow.
    4. Compiled and saved the PDF to `outputs/reports/startup_profit_prediction_handout.pdf`.

---

### Step 9: Interactive Dashboard Development (Deployment)
*   **Source File**: [streamlit_app.py](file:///c:/Users/user/Desktop/HW6/hw6_50_startups_profit_prediction/streamlit_app.py)
*   **Logic**:
    1. Configured layout parameters using `st.set_page_config` (wide layout, custom icon).
    2. Injected custom CSS styles via `st.markdown` to create stylized, modern glassmorphic dashboard container cards:
       * `.predict-box`: A custom neon border (`#00F5D4`) card with a glowing box-shadow for displaying predictions.
       * `.formula-box`: A styled terminal box displaying the model equation.
       * `.metric-card`: Cards for showing detailed process stages.
    3. Programmed a **3-tab Sidebar Navigation System**:
       * **🔮 Interactive Predictor**:
         * Implemented dual slider controls for inputting `R&D Spend` and `Marketing Spend`.
         * Excluded inputs (Administration Spend, State) are displayed but disabled, with helper tooltips explaining they were excluded by feature selection to avoid overfitting.
         * Loads the serialized model dynamically via `joblib.load` to compute predictions instantly.
         * Renders the estimated profit value, the dynamic mathematical formula, and written business insights.
       * **📊 Data Exploratory & Insights**:
         * Displays tabular listings of the dataset, descriptive statistics, and correlation matrix.
         * Shows 4 major diagnostic plots in a double-column configuration.
       * **🎓 CRISP-DM Methodology**:
         * Displays educational cards describing the work done in each of the 6 CRISP-DM stages.

---

## 3. Core Insights & Analytical Summary

1.  **Dominance of R&D Spend**:
    *   `R&D Spend` emerged as the most critical driver of company profit, ranking #1 across all 5 feature selection methods.
    *   The model coefficient for `R&D Spend` is `0.8056`, indicating that for every dollar spent on research and development, company profit increases by approximately 81 cents.
2.  **Marketing Spend Influence**:
    *   `Marketing Spend` was identified as the second most important feature, ranking #2 in SFS, SelectKBest, Lasso, and Random Forest.
    *   Its coefficient is `0.0272` (2.7 cents profit return per dollar spent), showing a positive but significantly smaller effect compared to R&D.
3.  **Variable Exclusion (Occam's Razor)**:
    *   `Administration` and `State` factors (New York, Florida, California) were shown to have no statistically significant impact on improving prediction accuracy.
    *   Excluding these variables simplified the model, significantly reduced the risk of overfitting, and improved test-set generalization (Model 2 with 2 features yields an Adjusted $R^2$ of `0.8931` and RMSE of `8206.33`, whereas Model 5 with all features degrades the Adjusted $R^2$ to `0.7721` and increases the RMSE to `9055.96`).
