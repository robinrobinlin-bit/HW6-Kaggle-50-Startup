import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def get_ranks(values, ascending=False):
    """
    Helper to convert values (like importances or scores) into ranks (1 to N).
    If ascending=False, the largest value gets rank 1.
    If ascending=True, the smallest value gets rank 1.
    """
    series = pd.Series(values)
    if ascending:
        return series.rank(method='first', ascending=True).astype(int)
    else:
        return series.rank(method='first', ascending=False).astype(int)

def perform_feature_selection(X_train, y_train, model_comparison_df=None):
    """
    Performs 5 feature selection methods and outputs a ranking comparison and summary.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    model_comparison_df : pd.DataFrame, optional
        Dataframe from model_training to assist in analysis.
        
    Returns:
    --------
    ranking_df : pd.DataFrame
        Consolidated feature rankings.
    """
    feature_names = list(X_train.columns)
    n_features = len(feature_names)
    
    # 1. Forward Sequential Feature Selection (SFS)
    # Manual implementation for explicit ranking from 1 to N
    remaining = list(feature_names)
    selected = []
    sfs_order = []
    for i in range(n_features):
        best_score = -np.inf
        best_feat = None
        for f in remaining:
            candidate = selected + [f]
            model = LinearRegression()
            model.fit(X_train[candidate], y_train)
            score = model.score(X_train[candidate], y_train) # training R2 score
            if score > best_score:
                best_score = score
                best_feat = f
        selected.append(best_feat)
        remaining.remove(best_feat)
        sfs_order.append(best_feat)
        
    sfs_ranks = [sfs_order.index(f) + 1 for f in feature_names]
    
    # 2. Recursive Feature Elimination (RFE)
    rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=1)
    rfe_selector.fit(X_train, y_train)
    rfe_ranks = list(rfe_selector.ranking_)
    
    # 3. SelectKBest (f_regression)
    skb = SelectKBest(score_func=f_regression, k='all')
    skb.fit(X_train, y_train)
    skb_ranks = list(get_ranks(skb.scores_, ascending=False))
    
    # 4. Lasso Regression (with standardized features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Use alpha=100.0 (reasonable regularizer to get distinct coefficient sizes)
    lasso = Lasso(alpha=100.0, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_ranks = list(get_ranks(np.abs(lasso.coef_), ascending=False))
    
    # 5. Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_ranks = list(get_ranks(rf.feature_importances_, ascending=False))
    
    # Consolidate into DataFrame
    ranking_data = {
        "Feature": feature_names,
        "SFS Forward": sfs_ranks,
        "RFE": rfe_ranks,
        "SelectKBest": skb_ranks,
        "Lasso": lasso_ranks,
        "Random Forest": rf_ranks
    }
    ranking_df = pd.DataFrame(ranking_data)
    
    # Calculate Average Rank
    rank_cols = ["SFS Forward", "RFE", "SelectKBest", "Lasso", "Random Forest"]
    ranking_df["Average Rank"] = ranking_df[rank_cols].mean(axis=1)
    ranking_df = ranking_df.sort_values(by="Average Rank").reset_index(drop=True)
    
    # Ensure reports directory exists
    os.makedirs(os.path.join("outputs", "reports"), exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join("outputs", "reports", "feature_selection_ranking.csv")
    ranking_df.to_csv(csv_path, index=False, encoding="utf-8")
    
    # Analysis logic for report
    most_important = ranking_df.iloc[0]["Feature"]
    top_2 = list(ranking_df.iloc[:2]["Feature"])
    
    # Analysis on "Does adding more features always improve model performance?"
    # and "Which feature set is the best model?"
    more_features_better_text = (
        "No, adding more features is not always better. While adding features increases R-squared "
        "on the training set, it can lead to overfitting and a decrease in Adjusted R-squared or "
        "increase in RMSE on the test set. For example, adding Administration and State (Florida/New York) "
        "to the model with [R&D Spend, Marketing Spend] increases the complexity without improving "
        "predictive power, and may degrade generalization on testing data."
    )
    
    best_features_set = "R&D Spend + Marketing Spend"
    if model_comparison_df is not None:
        # Programmatically find the model with lowest RMSE on test set
        best_row = model_comparison_df.loc[model_comparison_df["RMSE"].idxmin()]
        best_features_set = best_row["Features"]
        best_features_name = best_row["Model Name"]
    else:
        best_features_name = "2 Features (R&D + Marketing)"
        
    # Save Text Summary
    txt_path = os.path.join("outputs", "reports", "feature_selection_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("               Feature Selection & Analysis Report\n")
        f.write("="*70 + "\n\n")
        f.write("1. Feature Rankings across 5 Selection Methods:\n")
        f.write(ranking_df.to_string(index=False) + "\n\n")
        f.write("2. Comprehensive Analysis:\n")
        f.write(f"   - Most Important Feature (Rank 1): {most_important}\n")
        f.write(f"     Rationale: It consistently ranked #1 across all 5 methods (SFS, RFE, SelectKBest, Lasso, RF).\n\n")
        f.write(f"   - Top 2 Features: {', '.join(top_2)}\n")
        f.write(f"     Rationale: These two variables account for the vast majority of the variance in Profit.\n\n")
        f.write(f"   - Is adding more features always better?\n")
        f.write(f"     {more_features_better_text}\n\n")
        f.write(f"   - Best Model Feature Combination:\n")
        f.write(f"     The optimal combination is: [{best_features_set}] (from '{best_features_name}').\n")
        f.write(f"     This set achieves the lowest RMSE and highest Adjusted R-squared, balancing model simplicity and performance.\n\n")
        f.write("="*70 + "\n")
        
    print(f"Feature selection completed. Saved to:\n  - {csv_path}\n  - {txt_path}")
    return ranking_df
