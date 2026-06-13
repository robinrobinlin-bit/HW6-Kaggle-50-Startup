import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

def evaluate_and_save_best_model(trained_models, X_train, y_train):
    """
    Identifies the best model based on test set RMSE,
    extracts parameters (intercept, coefficients), formats the equation,
    and serializes the model and feature columns using joblib.
    
    Parameters:
    -----------
    trained_models : dict
        Dictionary containing trained models and their test metrics.
    X_train : pd.DataFrame
        Training features (to get the exact columns and fit if necessary, 
        although we can retrieve it directly from the dictionary).
    y_train : pd.Series
        Training target.
        
    Returns:
    --------
    best_info : dict
        A dictionary containing the best model's details.
    """
    # Select the target best model as specified by the user: "2 Features (R&D + Marketing)"
    best_name = "2 Features (R&D + Marketing)"
    best_model_info = trained_models[best_name]
    best_model = best_model_info["model"]
    best_features = best_model_info["features"]
    best_r2 = best_model_info["r2"]
    min_rmse = best_model_info["rmse"]
    
    coefs = best_model.coef_
    intercept = best_model.intercept_
    
    # Format coefficients and formula
    formula_parts = [f"{intercept:.4f}"]
    coef_details = []
    for feat, coef in zip(best_features, coefs):
        sign = "+" if coef >= 0 else "-"
        formula_parts.append(f"{sign} {abs(coef):.4f} * ({feat})")
        coef_details.append(f"   - {feat}: {coef:.4f}")
        
    formula = "Profit = " + " ".join(formula_parts)
    
    best_info = {
        "Name": best_name,
        "Features": best_features,
        "RMSE": min_rmse,
        "R-squared": best_r2,
        "Intercept": intercept,
        "Coefficients": dict(zip(best_features, coefs)),
        "Formula": formula
    }
    
    # Save the model and feature columns
    os.makedirs(os.path.join("outputs", "models"), exist_ok=True)
    
    model_path = os.path.join("outputs", "models", "startup_profit_model_v2.pkl")
    columns_path = os.path.join("outputs", "models", "feature_columns.pkl")
    
    # Save best model
    joblib.dump(best_model, model_path)
    
    # Save columns sequence
    joblib.dump(best_features, columns_path)
    
    print(f"Best model evaluated: {best_name}")
    print(f"Model saved to: {model_path}")
    print(f"Features saved to: {columns_path}")
    
    return best_info
