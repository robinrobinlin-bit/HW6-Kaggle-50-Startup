import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configurations
st.set_page_config(
    page_title="50 Startups Profit Predictor Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (targeted styling only, avoiding global tag overrides)
st.markdown("""
<style>
    /* Styling custom prediction container card */
    .predict-box {
        background-color: #1a2c3d;
        padding: 30px;
        border-radius: 15px;
        border: 2px solid #00F5D4;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 15px rgba(0, 245, 212, 0.15);
    }
    .predict-value {
        color: #00F5D4 !important;
        font-size: 2.8rem !important;
        font-weight: bold;
        margin-top: 10px;
    }
    /* Formula card styling */
    .formula-box {
        background-color: #1a2c3d;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #415A77;
        font-family: 'Courier New', Courier, monospace;
        color: #90E0EF !important;
        margin-bottom: 25px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: bold;
    }
    /* CRISP-DM cards styling */
    .metric-card {
        background-color: #1a2c3d;
        padding: 18px;
        border-radius: 8px;
        border-left: 5px solid #00B4D8;
        border-top: 1px solid #415A77;
        border-right: 1px solid #415A77;
        border-bottom: 1px solid #415A77;
        margin-bottom: 15px;
    }
    .metric-title {
        color: #00F5D4 !important;
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 1.1rem;
    }
    .metric-desc {
        color: #f0f0f0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
model_path = os.path.join("outputs", "models", "startup_profit_model_v2.pkl")
columns_path = os.path.join("outputs", "models", "feature_columns.pkl")
data_path = os.path.join("data", "50_Startups.csv")

# Load model and columns
model_loaded = False
if os.path.exists(model_path) and os.path.exists(columns_path):
    try:
        model = joblib.load(model_path)
        feature_cols = joblib.load(columns_path)
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
else:
    st.sidebar.warning("⚠️ Model files not found. Please run the training script first!")

# Load dataset
@st.cache_data
def load_dataset(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df_raw = load_dataset(data_path)

# Sidebar navigation
st.sidebar.title("🚀 Navigation & Settings")
page = st.sidebar.radio("Go to:", ["🔮 Interactive Predictor", "📊 Data Exploratory & Insights", "🎓 CRISP-DM Methodology"])

if page == "🔮 Interactive Predictor":
    st.title("🔮 Startup Profit Predictor Dashboard")
    st.markdown("Enter the spending variables of the startup to predict its net profit.")
    
    if not model_loaded:
        st.error("Model files are missing. Please execute the pipeline (`python run_project.py`) first to train and save the model.")
    else:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("🛠️ Input Variables")
            # Get defaults from dataset if available
            default_rd = float(df_raw['R&D Spend'].mean()) if df_raw is not None else 73721.0
            default_mkt = float(df_raw['Marketing Spend'].mean()) if df_raw is not None else 211025.0
            
            rd_input = st.slider("🔬 R&D Spend (研發支出) ($)", 0.0, 200000.0, default_rd, step=500.0)
            mkt_input = st.slider("📣 Marketing Spend (行銷支出) ($)", 0.0, 500000.0, default_mkt, step=1000.0)
            
            # Optional display of other variables (explaining they are excluded)
            st.markdown("---")
            st.markdown("##### ℹ️ Variables Excluded by Feature Selection")
            st.slider("💼 Administration Spend (行政支出) ($) - *Excluded*", 0.0, 200000.0, 120000.0, disabled=True)
            st.selectbox("📍 State (地區) - *Excluded*", ["New York", "Florida", "California"], disabled=True)
            
        with col2:
            # Predict
            input_df = pd.DataFrame([[rd_input, mkt_input]], columns=feature_cols)
            prediction = model.predict(input_df)[0]
            
            st.subheader("🎯 Prediction Output")
            st.markdown(f"""
            <div class="predict-box">
                <div style="font-size: 1.2rem; color: #90E0EF;">Estimated Startup Profit (預估獲利)</div>
                <div class="predict-value">${prediction:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display model equation
            st.subheader("📝 Model Linear Formula")
            intercept = model.intercept_
            coef_rd, coef_mkt = model.coef_[0], model.coef_[1]
            st.markdown(f"""
            <div class="formula-box">
                Profit = {intercept:,.4f} + {coef_rd:.4f} * (R&D Spend) + {coef_mkt:.4f} * (Marketing Spend)
            </div>
            """, unsafe_allow_html=True)
            
            # Insights
            st.subheader("💡 Business Insights (商業洞察)")
            st.markdown(f"""
            - **R&D Spend (研發支出) 係數為 `{coef_rd:.4f}`**：代表研發投入每增加 1 元，獲利預期將增加約 `{coef_rd:.2f}` 元。這是公司獲利最強勁的驅動因子。
            - **Marketing Spend (行銷支出) 係數為 `{coef_mkt:.4f}`**：代表行銷花費每增加 1 元，獲利預期將增加約 `{coef_mkt:.2f}` 元。雖然影響為正，但邊際效應遠小於研發。
            - **行政與地區支出被剔除**：在特徵選擇中，`Administration`與`State`變數均對提升獲利預測無顯著貢獻，簡化模型不僅提升了泛化能力，也避免了過擬合。
            """)

elif page == "📊 Data Exploratory & Insights":
    st.title("📊 Data Analytics & Model Diagnostic Plots")
    st.markdown("Explore data statistics and check validation diagnostic charts generated during CRISP-DM.")
    
    if df_raw is not None:
        st.subheader("📋 Dataset Overview (Raw 50 Startups)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🔢 Descriptive Statistics")
            st.dataframe(df_raw.describe(), use_container_width=True)
        with c2:
            st.markdown("##### 📌 Correlation Matrix")
            st.dataframe(df_raw.select_dtypes(include=[np.number]).corr(), use_container_width=True)
            
    st.markdown("---")
    st.subheader("📈 Technical Plots Showcase")
    
    fig_folder = os.path.join("outputs", "figures")
    if os.path.exists(fig_folder):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. Actual vs Predicted Profit Fit")
            actual_pred_img = os.path.join(fig_folder, "actual_vs_predicted.png")
            if os.path.exists(actual_pred_img):
                st.image(actual_pred_img, use_container_width=True)
                
            st.markdown("#### 2. Residual Analysis Plot")
            residual_img = os.path.join(fig_folder, "residual_plot.png")
            if os.path.exists(residual_img):
                st.image(residual_img, use_container_width=True)
        
        with col2:
            st.markdown("#### 3. Feature Selection Performance Curve")
            perf_img = os.path.join(fig_folder, "feature_selection_performance.png")
            if os.path.exists(perf_img):
                st.image(perf_img, use_container_width=True)
                
            st.markdown("#### 4. Correlation Heatmap")
            corr_img = os.path.join(fig_folder, "correlation_heatmap.png")
            if os.path.exists(corr_img):
                st.image(corr_img, use_container_width=True)
    else:
        st.warning("Figures not found under `outputs/figures/`. Please run the pipeline script first to generate them.")

elif page == "🎓 CRISP-DM Methodology":
    st.title("🎓 CRISP-DM Process Framework")
    
    steps = [
        ("1. Business Understanding (商業理解)", "定義企業獲利預測的核心商業課題。確立我們必須找出資源（研發、行銷、行政）的最佳配置比例以極大化 Profit。"),
        ("2. Data Understanding (資料理解)", "讀取並探索 50 筆 Startup 數據。完成缺失值、重複值檢驗與變數關聯度分析，輸出成 `data_understanding_report.txt`。"),
        ("3. Data Preparation (資料準備)", "對 State 變數進行 One-Hot Encoding。為避免 Dummy Variable Trap (共線性)，剔除 California 作為 baseline，僅保留 New York 與 Florida。切分 20% 測試集，設定 `random_state=42`。"),
        ("4. Modeling (模型建置)", "建立 sklearn 多元線性迴歸，並比較從 1 到 5 個特徵的多種特徵子集表現，追蹤 RMSE 與 R-squared 變化。"),
        ("5. Evaluation (模型評估)", "透過 5 種特徵選擇方法（SFS、RFE、SelectKBest、Lasso、RF）交叉分析，證明 `R&D Spend + Marketing Spend` 是最佳的特徵組合，以最簡模型達成 RMSE ~ 8206 與 R2 ~ 0.9168。"),
        ("6. Deployment (部署儲存)", "使用 `joblib` 將最佳預測模型與特徵欄位順序儲存為 `.pkl` 檔案，並開發此 Streamlit 網頁系統供業務端使用。")
    ]
    
    for title, desc in steps:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
