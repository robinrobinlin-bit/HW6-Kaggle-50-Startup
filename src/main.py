import os
import sys

# Add src to python path if not already there
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_check_raw_data, preprocess_data, split_data
from model_training import train_and_compare_models
from feature_selection import perform_feature_selection
from evaluation import evaluate_and_save_best_model
from visualization import generate_plots
from generate_pdf import build_pdf_handout


def run_pipeline():
    """
    Orchestrates the entire CRISP-DM pipeline for the 50 Startups profit prediction project.
    """
    print("Starting CRISP-DM Machine Learning Pipeline...\n")
    
    # 1. Data Understanding (讀取與分析資料)
    data_path = os.path.join("data", "50_Startups.csv")
    try:
        df_raw = load_and_check_raw_data(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please place the 50_Startups.csv file in the data/ folder and try again.")
        sys.exit(1)
        
    # 2. Data Preparation (資料準備與 One-Hot 編碼)
    print("Preprocessing data (One-Hot Encoding with California as baseline)...")
    X, y = preprocess_data(df_raw)
    
    print("Splitting data (test_size=0.2, random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 3. Modeling (多元線性迴歸模型訓練與比較)
    print("Training models on 5 feature combinations...")
    comparison_df, trained_models = train_and_compare_models(X_train, X_test, y_train, y_test)
    
    # 4. Feature Selection (五種特徵選擇方法分析)
    print("Performing feature selection using SFS, RFE, SelectKBest, Lasso, and Random Forest...")
    ranking_df = perform_feature_selection(X_train, y_train, comparison_df)
    
    # 5. Evaluation & Model Saving (模型評估與最佳模型儲存)
    print("Evaluating test set performance and saving the best model...")
    best_info = evaluate_and_save_best_model(trained_models, X_train, y_train)
    
    # 6. Visualization (視覺化圖表繪製)
    print("Generating 7 technical white-paper charts...")
    best_name = best_info["Name"]
    best_predictions = trained_models[best_name]["predictions"]
    generate_plots(comparison_df, ranking_df, best_info, best_predictions, X_test, y_test, df_raw)
    
    # 7. PDF Handout Compilation (圖文並茂講義 PDF 產生)
    print("Generating illustrated PDF handout booklet...")
    build_pdf_handout()
    
    # Format features string for console print
    best_features_str = " + ".join(best_info["Features"])
    
    # 8. Output summary terminal block
    print("\n" + "="*60)
    print("HW6 50 Startups Profit Prediction Completed Successfully")
    print("="*60)
    print("\nBest Model:")
    print(f"Features: {best_features_str}")
    print(f"RMSE: {best_info['RMSE']:.4f}")
    print(f"R-squared: {best_info['R-squared']:.4f}")
    print("\nOutput files saved to:")
    print("  outputs/figures/")
    print("  outputs/models/")
    print("  outputs/reports/")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_pipeline()
