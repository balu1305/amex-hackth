"""
AMEX Round 2 - Streamlined Employee Psychology Model
===================================================
Memory-efficient model focused on core employee behavior patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

def create_minimal_psychology_features(train_data, test_data):
    """Create only essential employee psychology features"""
    print("🧠 Creating Essential Employee Psychology Features...")
    
    # Select only the most important base features (from our previous analysis)
    important_features = [
        'f350', 'f366', 'f364', 'f203', 'f77', 'f223', 'f30', 'f38', 'f204', 'f361',
        'f51', 'f363', 'f224', 'f58', 'f139', 'f39', 'f76', 'f68', 'f85', 'f349'
    ]
    
    # Keep only features that exist
    available_features = [f for f in important_features if f in train_data.columns]
    print(f"Using {len(available_features)} key features")
    
    # Convert to numeric
    for col in available_features:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').astype(np.float32)
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').astype(np.float32)
    
    # === CORE EMPLOYEE PSYCHOLOGY ===
    
    # 1. Weekend Pattern (f349 appears to be day of week)
    if 'f349' in available_features:
        train_data['weekend_behavior'] = (train_data['f349'] >= 5).astype(np.int8)
        test_data['weekend_behavior'] = (test_data['f349'] >= 5).astype(np.int8)
        print("✅ Weekend browsing pattern")
    
    # 2. High Value Customer (based on f30, f38, f39 which seem spending-related)
    if all(col in available_features for col in ['f30', 'f38', 'f39']):
        total_spend = train_data[['f30', 'f38', 'f39']].sum(axis=1)
        high_value_threshold = total_spend.quantile(0.8)
        train_data['high_value_customer'] = (total_spend > high_value_threshold).astype(np.int8)
        
        test_total_spend = test_data[['f30', 'f38', 'f39']].sum(axis=1)
        test_data['high_value_customer'] = (test_total_spend > high_value_threshold).astype(np.int8)
        print("✅ High value customer pattern")
    
    # 3. Conservative Behavior (based on most important feature f350)
    if 'f350' in available_features:
        median_f350 = train_data['f350'].median()
        train_data['conservative_user'] = (train_data['f350'] <= median_f350).astype(np.int8)
        test_data['conservative_user'] = (test_data['f350'] <= median_f350).astype(np.int8)
        print("✅ Conservative behavior pattern")
    
    # 4. Engagement Level (based on f366 which has high importance)
    if 'f366' in available_features:
        high_engagement_threshold = train_data['f366'].quantile(0.7)
        train_data['high_engagement'] = (train_data['f366'] > high_engagement_threshold).astype(np.int8)
        test_data['high_engagement'] = (test_data['f366'] > high_engagement_threshold).astype(np.int8)
        print("✅ Engagement level pattern")
    
    # Final feature set (base + psychology)
    final_features = available_features + ['weekend_behavior', 'high_value_customer', 'conservative_user', 'high_engagement']
    final_features = [f for f in final_features if f in train_data.columns]
    
    print(f"Final features: {len(final_features)}")
    return train_data, test_data, final_features

def get_ultra_conservative_params():
    """Ultra-conservative parameters to prevent overfitting"""
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        
        # Extreme overfitting prevention
        'num_leaves': 15,            # Very small trees
        'learning_rate': 0.003,      # Very slow learning
        'feature_fraction': 0.4,     # Use only 40% features
        'bagging_fraction': 0.5,     # Use only 50% data
        'bagging_freq': 2,
        'min_data_in_leaf': 500,     # Large leaves
        
        # Strong regularization
        'lambda_l1': 5.0,
        'lambda_l2': 5.0,
        'max_depth': 4,              # Very shallow
        
        # Class imbalance
        'is_unbalance': True,
        'scale_pos_weight': 25,      # Strong minority weight
        
        # Additional safety
        'extra_trees': True,
        'subsample_for_bin': 50000,  # Smaller subsample
        'verbosity': -1,
        'random_state': 42,
        'force_col_wise': True
    }

def main():
    """Streamlined execution for memory efficiency"""
    print("🏢 AMEX Round 2 - Streamlined Employee Psychology Model")
    print("=" * 65)
    print("🎯 Focus: Core employee patterns + strong overfitting prevention")
    
    # Load data
    print("\\n📂 Loading data...")
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train: {train_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Target analysis
    click_rate = train_data['y'].mean()
    print(f"\\n📊 Click Rate: {click_rate:.3%} (Very Low - Employee behavior)")
    
    # Create streamlined features
    train_data, test_data, feature_cols = create_minimal_psychology_features(train_data, test_data)
    
    # Handle missing values efficiently
    print("\\n🔧 Efficient missing value handling...")
    for col in feature_cols:
        if col in train_data.columns and col in test_data.columns:
            median_val = train_data[col].median()
            train_data[col] = train_data[col].fillna(median_val)
            test_data[col] = test_data[col].fillna(median_val)
    
    # Prepare data without copying large dataframes
    print("\\n📊 Preparing final datasets...")
    X = train_data[feature_cols]
    y = train_data['y']
    X_test = test_data[feature_cols]
    
    print(f"Feature matrix: {X.shape}")
    print(f"Test matrix: {X_test.shape}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\\nTrain: {len(X_train):,} samples")
    print(f"Validation: {len(X_val):,} samples")
    
    # Get ultra-conservative parameters
    params = get_ultra_conservative_params()
    print("\\n⚙️ Using ultra-conservative parameters")
    
    # Train model with patient early stopping
    print("\\n🚀 Training patient model...")
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=15000,  # Many rounds but will stop early
        callbacks=[
            lgb.early_stopping(stopping_rounds=1000),  # Very patient
            lgb.log_evaluation(period=500)
        ]
    )
    
    # Evaluate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"\\n🎯 Validation AUC: {val_auc:.4f}")
    print(f"Best Iteration: {model.best_iteration}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔍 Top Employee Psychology Features:")
    for i, (_, row) in enumerate(importance.head(8).iterrows()):
        print(f"{i+1}. {row['feature']}: {row['importance']:,.0f}")
    
    # Generate test predictions
    print(f"\\n📊 Generating predictions...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save submission
    submission_file = 'r2_submission_fileTVIJAYABALAJI_stream.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\\n✅ Streamlined Employee Model Complete!")
    print(f"📁 Submission: {submission_file}")
    print(f"🎯 Validation AUC: {val_auc:.4f}")
    
    # Prediction analysis
    print(f"\\n📊 Prediction Analysis:")
    print(f"   Mean: {test_pred.mean():.4f}")
    print(f"   Std: {test_pred.std():.4f}")
    print(f"   Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    print(f"   Pred > 0.5: {(test_pred > 0.5).sum():,} samples ({(test_pred > 0.5).mean():.1%})")
    
    # Model insights
    print(f"\\n🏢 Employee Psychology Model Insights:")
    print(f"   ✅ Ultra-conservative to prevent overfitting")
    print(f"   ✅ Weekend browsing behavior")
    print(f"   ✅ High-value customer identification")
    print(f"   ✅ Conservative user patterns")
    print(f"   ✅ Engagement level categorization")
    print(f"   ✅ Patient early stopping (1000 rounds)")
    print(f"   ✅ Strong class imbalance handling")
    print(f"   ✅ Minimal feature set for generalization")
    
    # Performance expectation
    if val_auc < 0.65:
        print(f"\\n💡 Conservative approach - prioritizing generalization over validation score")
        print(f"   This should perform better on test data than high-validation models")
    elif val_auc > 0.8:
        print(f"\\n⚠️  High validation score - watch for overfitting on leaderboard")
    else:
        print(f"\\n🎯 Balanced validation score - good generalization potential")
    
    # Save importance for analysis
    importance.to_csv('streamlined_importance.csv', index=False)

if __name__ == "__main__":
    main()
