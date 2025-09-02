"""
Quick Start Script for AMEX Round 2
===================================
This script provides a simple baseline to get you started quickly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def quick_start():
    print("AMEX Round 2 - Quick Start Baseline")
    print("=" * 40)
    
    # Load main data
    print("Loading data...")
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Check target distribution
    target_counts = train_data['y'].value_counts()
    print(f"\nTarget distribution:")
    print(f"No clicks (0): {target_counts[0]:,} ({target_counts[0]/len(train_data)*100:.1f}%)")
    print(f"Clicks (1): {target_counts[1]:,} ({target_counts[1]/len(train_data)*100:.1f}%)")
    
    # Prepare features (convert to numeric and handle missing values)
    feature_cols = [col for col in train_data.columns if col.startswith('f')]
    
    print(f"\nUsing {len(feature_cols)} features")
    
    # Convert features to numeric
    X = train_data[feature_cols].copy()
    X_test = test_data[feature_cols].copy()
    
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(0)
    X_test = X_test.fillna(0)
    
    # Convert target to numeric
    y = pd.to_numeric(train_data['y'], errors='coerce')
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    # Train simple LightGBM model
    print("\nTraining LightGBM model...")
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'is_unbalance': True,  # Handle class imbalance
        'random_state': 42
    }
    
    train_data_lgb = lgb.Dataset(X_train, label=y_train)
    val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
    
    model = lgb.train(
        lgb_params,
        train_data_lgb,
        valid_sets=[val_data_lgb],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    # Evaluate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    print(f"\nValidation AUC: {auc:.4f}")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_imp.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']}: {row['importance']:,.0f}")
    
    # Generate test predictions
    print("\nGenerating test predictions...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save submission
    submission_file = 'quick_submission.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"Submission saved to: {submission_file}")
    print(f"Prediction stats:")
    print(f"  Mean: {test_pred.mean():.4f}")
    print(f"  Min: {test_pred.min():.4f}")
    print(f"  Max: {test_pred.max():.4f}")
    print(f"  Std: {test_pred.std():.4f}")
    
    print(f"\nBaseline AUC: {auc:.4f}")
    print("Next steps:")
    print("1. Run the full analysis script: python round2_analysis.py")
    print("2. Add feature engineering")
    print("3. Try different models")
    print("4. Handle class imbalance better")

if __name__ == "__main__":
    quick_start()
