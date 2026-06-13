import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple linear regression models on 5 pre-defined feature combinations,
    calculates metrics on the test set, and saves results to csv and txt files.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Testing target.
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        DataFrame containing metrics for all 5 models.
    trained_models : dict
        Dictionary of trained model instances.
    """
    # Define feature combinations
    feature_combinations = {
        "1 Feature (R&D)": ['R&D Spend'],
        "2 Features (R&D + Marketing)": ['R&D Spend', 'Marketing Spend'],
        "3 Features (R&D + Marketing + NY)": ['R&D Spend', 'Marketing Spend', 'New York'],
        "4 Features (R&D + Marketing + NY + FL)": ['R&D Spend', 'Marketing Spend', 'New York', 'Florida'],
        "5 Features (R&D + Marketing + NY + FL + Admin)": ['R&D Spend', 'Marketing Spend', 'New York', 'Florida', 'Administration']
    }
    
    results = []
    trained_models = {}
    
    n = len(y_test)
    
    for name, features in feature_combinations.items():
        # Subset features
        X_tr = X_train[features]
        X_te = X_test[features]
        p = len(features)
        
        # Train model
        model = LinearRegression()
        model.fit(X_tr, y_train)
        
        # Predict
        y_pred = model.predict(X_te)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Adjusted R-squared
        if n - p - 1 > 0:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = np.nan
            
        # Record results
        results.append({
            "Model Name": name,
            "Features": ", ".join(features),
            "Num Features": p,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            "Adjusted R-squared": adj_r2
        })
        
        # Save model instance
        trained_models[name] = {
            "model": model,
            "features": features,
            "rmse": rmse,
            "r2": r2,
            "predictions": y_pred
        }
        
    comparison_df = pd.DataFrame(results)
    
    # Ensure outputs directory exists
    os.makedirs(os.path.join("outputs", "reports"), exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join("outputs", "reports", "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    # Save Text Report
    txt_path = os.path.join("outputs", "reports", "model_comparison.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("                       Model Comparison Report (CRISP-DM Modeling)\n")
        f.write("="*90 + "\n\n")
        f.write(comparison_df.to_string(index=False) + "\n\n")
        f.write("="*90 + "\n")
        
    print(f"Model comparisons completed. Saved to:\n  - {csv_path}\n  - {txt_path}")
    return comparison_df, trained_models
