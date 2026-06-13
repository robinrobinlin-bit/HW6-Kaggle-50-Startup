import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Apply professional "Tech Blue" dark theme styles to Matplotlib
def apply_tech_blue_theme():
    """
    Applies a sleek, sci-fi tech blue theme to matplotlib plots.
    Features:
    - Deep navy background (#0D1B2A)
    - Darker slate axes (#1B263B)
    - Off-white/silver labels (#E0E1DD)
    - Slate-blue grids (#415A77)
    - Accent colors: Neon Cyan (#00F5D4), Tech Blue (#00B4D8), Deep Sky Blue (#0077B6)
    """
    sns.set_theme(style="dark")
    
    plt.rcParams['figure.facecolor'] = '#0D1B2A'
    plt.rcParams['axes.facecolor'] = '#1B263B'
    plt.rcParams['text.color'] = '#E0E1DD'
    plt.rcParams['axes.labelcolor'] = '#E0E1DD'
    plt.rcParams['xtick.color'] = '#E0E1DD'
    plt.rcParams['ytick.color'] = '#E0E1DD'
    plt.rcParams['axes.edgecolor'] = '#415A77'
    plt.rcParams['grid.color'] = '#415A77'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['legend.facecolor'] = '#1B263B'
    plt.rcParams['legend.edgecolor'] = '#415A77'

def generate_plots(comparison_df, ranking_df, best_info, best_predictions, X_test, y_test, df_raw):
    """
    Generates all 7 required plots and saves them to outputs/figures/.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Dataframe from model training.
    ranking_df : pd.DataFrame
        Dataframe from feature selection.
    best_info : dict
        Details of the best model.
    best_predictions : np.ndarray
        Predictions of the best model on test set.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test actual profit.
    df_raw : pd.DataFrame
        Raw dataset (for correlation matrix).
    """
    apply_tech_blue_theme()
    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    
    # 1. RMSE by Number of Features
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(comparison_df["Num Features"], comparison_df["RMSE"], marker='o', color='#00F5D4', linewidth=2, markersize=8)
    ax.set_title("RMSE by Number of Features", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("RMSE ($)", fontsize=11)
    ax.set_xticks(comparison_df["Num Features"])
    # Highlight the best number of features
    best_num = len(best_info["Features"])
    best_rmse = best_info["RMSE"]
    ax.plot(best_num, best_rmse, marker='o', color='#FF007F', markersize=10, markeredgecolor='white', label='Best Model')
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/rmse_by_features.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # 2. R-squared by Number of Features
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(comparison_df["Num Features"], comparison_df["R-squared"], marker='s', color='#00B4D8', linewidth=2, label="R-squared")
    ax.plot(comparison_df["Num Features"], comparison_df["Adjusted R-squared"], marker='^', color='#90E0EF', linewidth=2, linestyle='--', label="Adjusted R-squared")
    ax.set_title("R-squared & Adjusted R-squared by Feature Count", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(comparison_df["Num Features"])
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("outputs/figures/r2_by_features.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # 3. Feature Selection Performance Curve
    # Shows RMSE and R2 on a dual-axis plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = '#00F5D4'
    ax1.set_xlabel('Number of Features', fontsize=11)
    ax1.set_ylabel('RMSE ($)', color=color, fontsize=11)
    line1 = ax1.plot(comparison_df["Num Features"], comparison_df["RMSE"], color=color, marker='o', linewidth=2, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(comparison_df["Num Features"])
    
    ax2 = ax1.twinx()  
    color = '#00B4D8'
    ax2.set_ylabel('R-squared', color=color, fontsize=11)
    line2 = ax2.plot(comparison_df["Num Features"], comparison_df["R-squared"], color=color, marker='s', linewidth=2, linestyle='-.', label='R-squared')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False) # avoid double gridlines
    
    # added legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title("Feature Selection Performance Curve", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_selection_performance.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # 4. Feature Importance Comparison (Heatmap of Ranks)
    # Pivot or format ranking_df
    # X_train features are ranked from 1 (best) to 5 (worst)
    rank_matrix = ranking_df.set_index("Feature")
    rank_cols = ["SFS Forward", "RFE", "SelectKBest", "Lasso", "Random Forest"]
    plot_data = rank_matrix[rank_cols]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # We use a custom color map: lighter means better rank (lower number)
    sns.heatmap(plot_data, annot=True, cmap="YlGnBu_r", cbar=True, linewidths=1, linecolor='#0D1B2A', ax=ax,
                cbar_kws={'label': 'Rank (1 is Best)'}, annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Feature Ranking Comparison Across Methods", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Selection Method", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/figures/feature_selection_comparison.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # 5. Actual vs Predicted Profit (Best Model)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, best_predictions, color='#00F5D4', edgecolors='#1B263B', s=80, alpha=0.8, label='Predicted')
    # Draw reference line y=x
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, color='#FF007F', linestyle='--', alpha=0.8, linewidth=2, label='Ideal Fit (y = x)')
    ax.set_title(f"Actual vs Predicted Profit ({best_info['Name']})", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Actual Profit ($)", fontsize=11)
    ax.set_ylabel("Predicted Profit ($)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/actual_vs_predicted.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # 6. Residual Plot (Best Model)
    residuals = y_test - best_predictions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(best_predictions, residuals, color='#00B4D8', edgecolors='#1B263B', s=80, alpha=0.8)
    ax.axhline(y=0, color='#FF007F', linestyle='--', linewidth=2)
    ax.set_title(f"Residual Plot ({best_info['Name']})", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Profit ($)", fontsize=11)
    ax.set_ylabel("Residuals ($)", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/figures/residual_plot.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()

    # 7. Correlation Heatmap
    # One-hot encode the df_raw state for a complete correlation check
    state_dummies = pd.get_dummies(df_raw['State'], dtype=int)
    # Avoid dummy variable trap (California as baseline)
    if 'California' in state_dummies.columns:
        state_dummies = state_dummies.drop(columns=['California'])
    
    corr_df = pd.concat([df_raw.drop(columns=['State']), state_dummies], axis=1)
    # Rearrange columns so Profit is last
    cols = [c for c in corr_df.columns if c != 'Profit'] + ['Profit']
    corr_df = corr_df[cols]
    
    corr_matrix = corr_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".3f", linewidths=1, linecolor='#0D1B2A', ax=ax,
                annot_kws={"size": 11})
    ax.set_title("Correlation Heatmap (Including Categorical States)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/correlation_heatmap.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    
    print("All 7 plots generated and saved successfully to outputs/figures/")
