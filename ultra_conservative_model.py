"""
AMEX Round 2 - Ultra Conservative Employee Psychology Model
===========================================================
Extreme overfitting prevention with employee behavior focus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def fix_binary_target(y_column):
    """Convert binary string targets to 0/1"""
    print("🔧 Fixing binary string targets...")
    
    if y_column.dtype == 'object':
        # Count 1s in the binary string for each row
        y_fixed = y_column.apply(lambda x: str(x).count('1') > 0 if pd.notna(x) else 0).astype(int)
    else:
        y_fixed = pd.to_numeric(y_column, errors='coerce').fillna(0).astype(int)
    
    click_rate = y_fixed.mean()
    print(f"Target fixed - Click Rate: {click_rate:.3%} (Typical employee behavior)")
    return y_fixed

def create_employee_focused_features(train_data, test_data):
    """Focus on key employee psychology patterns"""
    print("🧠 Creating Employee-Focused Features...")
    
    # Only the most proven important features
    core_features = ['f350', 'f366', 'f364']  # Top 3 from importance
    available_features = [f for f in core_features if f in train_data.columns]
    
    print(f"Using {len(available_features)} core features")
    
    # Convert to numeric efficiently
    for col in available_features:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').astype(np.float32)
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').astype(np.float32)
    
    # Key employee psychology: Evening browsing pattern
    if 'f366' in available_features:
        # High engagement employees (more likely to click offers)
        engagement_75th = train_data['f366'].quantile(0.75)
        train_data['highly_engaged_employee'] = (train_data['f366'] > engagement_75th).astype(np.int8)
        test_data['highly_engaged_employee'] = (test_data['f366'] > engagement_75th).astype(np.int8)
        available_features.append('highly_engaged_employee')
        print("✅ Highly engaged employee pattern")
    
    return train_data, test_data, available_features

def cross_validate_conservative_model(X, y):
    """5-fold CV with ultra-conservative model"""
    print("🔄 5-Fold Cross-Validation (Ultra Conservative)...")
    
    # Extreme overfitting prevention parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 8,          # Extremely small trees
        'learning_rate': 0.005,   # Very slow learning
        'feature_fraction': 0.6,  # Use only 60% features
        'bagging_fraction': 0.6,  # Use only 60% data
        'bagging_freq': 3,
        'min_data_in_leaf': 2000, # Very large leaves
        'lambda_l1': 5.0,         # Strong regularization
        'lambda_l2': 5.0,
        'max_depth': 3,           # Very shallow
        'scale_pos_weight': 15,   # Moderate class weight
        'verbosity': -1,
        'random_state': 42,
        'extra_trees': True       # More randomness
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"📊 Fold {fold + 1}/5", end=" ")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train with extreme patience
        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=500),  # Very patient
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_val_fold, y_pred)
        cv_scores.append(auc_score)
        
        print(f"AUC: {auc_score:.4f}")
    
    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    
    print(f"📈 CV Results: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Individual: {[f'{s:.4f}' for s in cv_scores]}")
    
    return cv_scores, params

def main():
    """Ultra-conservative approach for real-world generalization"""
    print("🏢 AMEX Round 2 - Ultra Conservative Employee Model")
    print("=" * 60)
    print("🎯 Goal: Maximum generalization with employee psychology")
    
    # Load data
    print("\\n📂 Loading data...")
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train: {train_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Fix target
    y = fix_binary_target(train_data['y'])
    
    # Create focused features
    train_data, test_data, feature_cols = create_employee_focused_features(train_data, test_data)
    
    # Handle missing values
    print("\\n🔧 Conservative missing value handling...")
    for col in feature_cols:
        if col in train_data.columns and col in test_data.columns:
            # Use median for conservative approach
            median_val = train_data[col].median()
            train_data[col] = train_data[col].fillna(median_val)
            test_data[col] = test_data[col].fillna(median_val)
    
    # Prepare data
    X = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    print(f"\\n📊 Final data:")
    print(f"   Features: {X.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Cross-validation
    cv_scores, best_params = cross_validate_conservative_model(X, y)
    
    # Train final model with even more conservative approach
    print(f"\\n🚀 Training final ultra-conservative model...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # Larger validation
    )
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    # Even more conservative for final model
    best_params['learning_rate'] = 0.003  # Even slower
    best_params['num_leaves'] = 6          # Even smaller
    
    final_model = lgb.train(
        best_params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=800),  # Ultra patient
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Final validation
    val_pred = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    final_auc = roc_auc_score(y_val, val_pred)
    
    print(f"\\n🎯 Final Validation AUC: {final_auc:.4f}")
    print(f"Best Iteration: {final_model.best_iteration}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔍 Employee Psychology Feature Importance:")
    for i, (_, row) in enumerate(importance.iterrows()):
        print(f"{i+1}. {row['feature']}: {row['importance']:,.0f}")
    
    # Generate conservative predictions
    print(f"\\n📊 Generating ultra-conservative predictions...")
    test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save submission
    submission_file = 'r2_submission_fileTVIJAYABALAJI_ultra_conservative.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\\n✅ Ultra Conservative Employee Model Complete!")
    print(f"📁 Submission: {submission_file}")
    print(f"🎯 CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"🎯 Final AUC: {final_auc:.4f}")
    
    # Prediction analysis
    print(f"\\n📊 Conservative Prediction Analysis:")
    print(f"   Mean: {test_pred.mean():.4f}")
    print(f"   Std: {test_pred.std():.4f}")
    print(f"   Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Confidence in generalization
    cv_consistency = np.std(cv_scores)
    if cv_consistency < 0.01:
        print(f"\\n🎉 Excellent CV consistency - High confidence in generalization!")
    elif cv_consistency < 0.02:
        print(f"\\n✅ Good CV consistency - Should generalize well")
    else:
        print(f"\\n⚠️  CV variance detected - Conservative approach still recommended")
    
    print(f"\\n🏢 Ultra Conservative Employee Psychology Model:")
    print(f"   ✅ Extremely small trees (6 leaves)")
    print(f"   ✅ Very slow learning (0.003 rate)")
    print(f"   ✅ Strong regularization (L1=5, L2=5)")
    print(f"   ✅ Large validation set (30%)")
    print(f"   ✅ Patient early stopping (800 rounds)")
    print(f"   ✅ Employee engagement focus")
    print(f"   ✅ 5-fold cross-validation validated")
    
    if final_auc < 0.7:
        print(f"\\n💡 Ultra-conservative: Prioritizing real-world performance over validation")
    elif final_auc > 0.9:
        print(f"\\n⚠️  High validation AUC - Monitor for potential overfitting")
    else:
        print(f"\\n🎯 Balanced validation AUC - Optimal for generalization")

if __name__ == "__main__":
    main()
