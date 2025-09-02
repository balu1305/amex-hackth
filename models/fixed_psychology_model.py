"""
AMEX Round 2 - Quick Employee Psychology Model (Fixed)
======================================================
Handles binary string targets and focuses on core employee behavior
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

def fix_binary_target(y_column):
    """Convert binary string targets to 0/1"""
    print("🔧 Fixing binary string targets...")
    
    # Handle string binary format
    if y_column.dtype == 'object':
        # Count 1s in the binary string for each row
        y_fixed = y_column.apply(lambda x: str(x).count('1') > 0 if pd.notna(x) else 0).astype(int)
    else:
        y_fixed = pd.to_numeric(y_column, errors='coerce').fillna(0).astype(int)
    
    print(f"Target fixed: {y_fixed.value_counts().to_dict()}")
    return y_fixed

def create_simple_psychology_features(train_data, test_data):
    """Create minimal psychology features efficiently"""
    print("🧠 Creating Simple Employee Psychology Features...")
    
    # Select top features only (memory efficient)
    important_features = ['f350', 'f366', 'f364', 'f203', 'f77']
    available_features = [f for f in important_features if f in train_data.columns]
    
    print(f"Using {len(available_features)} core features")
    
    # Convert to numeric
    for col in available_features:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').astype(np.float32)
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').astype(np.float32)
    
    # Simple psychology features
    if 'f366' in available_features:
        # High engagement (top 30%)
        threshold = train_data['f366'].quantile(0.7)
        train_data['high_engagement'] = (train_data['f366'] > threshold).astype(np.int8)
        test_data['high_engagement'] = (test_data['f366'] > threshold).astype(np.int8)
        available_features.append('high_engagement')
    
    if 'f350' in available_features:
        # Conservative behavior (below median)
        median_val = train_data['f350'].median()
        train_data['conservative'] = (train_data['f350'] <= median_val).astype(np.int8)
        test_data['conservative'] = (test_data['f350'] <= median_val).astype(np.int8)
        available_features.append('conservative')
    
    return train_data, test_data, available_features

def main():
    """Simple robust execution"""
    print("🏢 AMEX Round 2 - Fixed Employee Psychology Model")
    print("=" * 55)
    
    # Load data
    print("📂 Loading data...")
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train: {train_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Fix target variable (binary strings to 0/1)
    y = fix_binary_target(train_data['y'])
    click_rate = y.mean()
    print(f"Click Rate: {click_rate:.3%}")
    
    # Create simple features
    train_data, test_data, feature_cols = create_simple_psychology_features(train_data, test_data)
    
    # Handle missing values
    print("🔧 Handling missing values...")
    for col in feature_cols:
        if col in train_data.columns and col in test_data.columns:
            median_val = train_data[col].median()
            train_data[col] = train_data[col].fillna(median_val)
            test_data[col] = test_data[col].fillna(median_val)
    
    # Prepare data
    X = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    print(f"Features: {X.shape}")
    print(f"Test: {X_test.shape}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Ultra-conservative LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 10,         # Very small
        'learning_rate': 0.01,    # Slow learning
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 1000, # Large leaves
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'max_depth': 3,           # Very shallow
        'scale_pos_weight': 20,   # Handle imbalance (removed is_unbalance)
        'verbosity': -1,
        'random_state': 42
    }
    
    print("🚀 Training conservative model...")
    
    # Train model
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Evaluate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"\n🎯 Validation AUC: {val_auc:.4f}")
    
    # Generate predictions
    print("📊 Generating test predictions...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save submission
    submission_file = 'r2_submission_fileTVIJAYABALAJI_fixed.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Fixed Employee Psychology Model Complete!")
    print(f"📁 Submission: {submission_file}")
    print(f"🎯 Validation AUC: {val_auc:.4f}")
    
    # Prediction stats
    print(f"\n📊 Prediction Statistics:")
    print(f"   Mean: {test_pred.mean():.4f}")
    print(f"   Std: {test_pred.std():.4f}")
    print(f"   Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top Features:")
    for i, (_, row) in enumerate(importance.iterrows()):
        print(f"{i+1}. {row['feature']}: {row['importance']:,.0f}")
    
    print(f"\n🏢 Key Employee Psychology Insights:")
    print(f"   ✅ Fixed binary string target issue")
    print(f"   ✅ Conservative model to prevent overfitting")
    print(f"   ✅ Employee engagement patterns")
    print(f"   ✅ Risk-averse behavior modeling")
    print(f"   ✅ Strong class imbalance handling")
    
    if val_auc < 0.7:
        print(f"\n💡 Conservative validation score - should generalize better to test data")
    else:
        print(f"\n🎉 Good validation score with robust approach!")

if __name__ == "__main__":
    main()
