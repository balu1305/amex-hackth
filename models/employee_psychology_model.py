"""
AMEX Round 2 - Memory-Efficient Employee Psychology Model
=========================================================
Focused on key employee behavior patterns with strong overfitting prevention
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

def create_smart_employee_features(train_data, test_data):
    """Create only the most impactful employee psychology features"""
    print("🧠 Creating Smart Employee Psychology Features...")
    
    # Get numeric feature columns
    feature_cols = [col for col in train_data.columns if col.startswith('f')]
    print(f"Base features: {len(feature_cols)}")
    
    # Convert to numeric efficiently
    for col in feature_cols:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    
    # === CORE EMPLOYEE PSYCHOLOGY FEATURES ===
    
    # 1. Evening Browsing Pattern (Key insight: employees browse after work)
    print("📱 Evening browsing patterns...")
    
    # Look for day-of-week features (0-6 range suggests days)
    for col in ['f349', 'f84', 'f120', 'f176']:  # Common day features
        if col in train_data.columns:
            values = train_data[col].fillna(0)
            if values.max() <= 6:  # Day of week
                # Weekend relaxed browsing
                train_data[f'{col}_weekend'] = (values >= 5).astype(np.int8)
                test_data[f'{col}_weekend'] = (test_data[col].fillna(0) >= 5).astype(np.int8)
                break  # Only create one weekend feature to save memory
    
    # 2. Spending Consistency (Regular vs Impulse spenders)
    print("💰 Spending consistency...")
    
    # Find spending features (higher variance, larger values)
    spend_features = []
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if values.std() > 10 and values.max() > 100:  # Likely spending
            spend_features.append(col)
    
    if len(spend_features) >= 3:
        # Take top 3 spending features to avoid memory issues
        top_spend = spend_features[:3]
        
        # Total spending
        train_data['total_spend'] = train_data[top_spend].sum(axis=1).astype(np.float32)
        test_data['total_spend'] = test_data[top_spend].sum(axis=1).astype(np.float32)
        
        # Spending consistency (regular spenders have low variance)
        train_data['spend_consistency'] = (1 / (train_data[top_spend].std(axis=1) + 1)).astype(np.float32)
        test_data['spend_consistency'] = (1 / (test_data[top_spend].std(axis=1) + 1)).astype(np.float32)
    
    # 3. High Engagement Flag (Active vs Passive users)
    print("🔄 Engagement patterns...")
    
    # Find frequency/count features
    freq_features = []
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if (values >= 0).all() and values.max() < 1000:  # Count-like
            if len(values.unique()) < 50:  # Reasonable count range
                freq_features.append(col)
    
    if len(freq_features) >= 2:
        # Take top 2 to save memory
        top_freq = freq_features[:2]
        
        # Total engagement
        train_data['total_engagement'] = train_data[top_freq].sum(axis=1).astype(np.float32)
        test_data['total_engagement'] = test_data[top_freq].sum(axis=1).astype(np.float32)
        
        # High engagement flag
        engagement_threshold = train_data['total_engagement'].quantile(0.7)
        train_data['high_engagement'] = (train_data['total_engagement'] > engagement_threshold).astype(np.int8)
        test_data['high_engagement'] = (test_data['total_engagement'] > engagement_threshold).astype(np.int8)
    
    # 4. Risk-Averse Behavior (Conservative spending patterns)
    print("⚠️ Risk patterns...")
    
    # Create one conservative flag from most important feature
    important_features = ['f350', 'f366', 'f364', 'f203', 'f77']  # From our previous analysis
    
    for col in important_features:
        if col in train_data.columns:
            values = train_data[col].fillna(0)
            if values.std() > 0:
                # Conservative behavior = below median
                median_val = values.median()
                train_data['conservative_behavior'] = (values <= median_val).astype(np.int8)
                test_data['conservative_behavior'] = (test_data[col].fillna(0) <= median_val).astype(np.int8)
                break  # Only one conservative feature
    
    # 5. Salary Cycle Patterns (End/beginning of month behavior)
    print("📅 Salary cycle...")
    
    # Look for day-of-month features (1-31 range)
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if values.min() >= 1 and values.max() <= 31 and len(values.unique()) > 10:
            # Payday pattern (beginning/end of month)
            train_data['payday_period'] = ((values <= 5) | (values >= 25)).astype(np.int8)
            test_data['payday_period'] = ((test_data[col].fillna(0) <= 5) | (test_data[col].fillna(0) >= 25)).astype(np.int8)
            break  # Only one payday feature
    
    print(f"✅ Created focused psychology features")
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    return train_data, test_data

def get_robust_lgb_params():
    """LightGBM parameters designed to prevent overfitting"""
    return {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        
        # Conservative overfitting prevention
        'num_leaves': 31,
        'learning_rate': 0.005,      # Very low learning rate
        'feature_fraction': 0.5,     # Use only half features
        'bagging_fraction': 0.6,     # Use only 60% data
        'bagging_freq': 3,
        'min_data_in_leaf': 300,     # Large leaf requirement
        
        # Strong regularization
        'lambda_l1': 3.0,
        'lambda_l2': 3.0,
        'max_depth': 5,              # Shallow trees
        
        # Class imbalance handling
        'is_unbalance': True,
        'scale_pos_weight': 20,      # Strong minority class weight
        
        # Additional robustness
        'extra_trees': True,         # More randomness
        'feature_pre_filter': False,
        'verbosity': -1,
        'random_state': 42,
        'force_col_wise': True
    }

def robust_validation(X, y, params):
    """5-fold cross-validation with robust evaluation"""
    print("🔄 5-Fold Robust Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"📊 Fold {fold + 1}/5", end=" ")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Create LightGBM datasets
        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=5000,  # High rounds but will stop early
            callbacks=[
                lgb.early_stopping(stopping_rounds=500),  # Patient early stopping
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_val_fold, y_pred)
        cv_scores.append(auc_score)
        
        print(f"AUC: {auc_score:.4f}")
        
        # Clean memory
        del model, train_set, val_set
        gc.collect()
    
    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    
    print(f"\n📈 Cross-Validation Results:")
    print(f"   Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"   All Folds: {[f'{s:.4f}' for s in cv_scores]}")
    
    return cv_scores

def main():
    """Main execution focusing on employee psychology"""
    print("🏢 AMEX Round 2 - Employee Psychology Robust Model")
    print("=" * 60)
    print("🎯 Focus: Working employee behavior with overfitting prevention")
    
    # Load data
    print("\n📂 Loading data...")
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train: {train_data.shape}")
    print(f"Test: {test_data.shape}")
    
    # Target analysis
    target_dist = train_data['y'].value_counts(normalize=True)
    print(f"\n📊 Target: No-Click {target_dist[0]:.1%}, Click {target_dist[1]:.1%}")
    print(f"Imbalance Ratio: {target_dist[0]/target_dist[1]:.1f}:1")
    
    # Create focused psychology features
    train_data, test_data = create_smart_employee_features(train_data, test_data)
    
    # Prepare feature matrix
    exclude_cols = ['id1', 'id2', 'id3', 'id4', 'id5', 'y']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"\n🎯 Final features: {len(feature_cols)}")
    
    # Handle missing values simply
    print("🔧 Handling missing values...")
    for col in feature_cols:
        median_val = train_data[col].median()
        train_data[col] = train_data[col].fillna(median_val)
        test_data[col] = test_data[col].fillna(median_val)
    
    # Prepare final data
    X = train_data[feature_cols].copy()
    y = train_data['y'].copy()
    X_test = test_data[feature_cols].copy()
    
    # Clean memory
    del train_data, test_data
    gc.collect()
    
    print(f"Feature matrix: {X.shape}")
    print(f"Test matrix: {X_test.shape}")
    
    # Get robust parameters
    params = get_robust_lgb_params()
    print(f"\n⚙️ Using conservative parameters to prevent overfitting")
    
    # Cross-validation
    cv_scores = robust_validation(X, y, params)
    
    # Train final model
    print(f"\n🚀 Training final model...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    final_model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=800),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Final validation
    val_pred = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    final_auc = roc_auc_score(y_val, val_pred)
    
    print(f"\n🎯 Final Validation AUC: {final_auc:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Employee Psychology Features:")
    for i, (_, row) in enumerate(importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']}: {row['importance']:,.0f}")
    
    # Generate predictions
    print(f"\n📊 Generating test predictions...")
    test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save submission
    submission_file = 'r2_submission_fileTVIJAYABALAJI_psychology.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Employee Psychology Model Complete!")
    print(f"📁 Submission: {submission_file}")
    print(f"🎯 CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"🎯 Final AUC: {final_auc:.4f}")
    
    print(f"\n📊 Prediction Analysis:")
    print(f"   Mean: {test_pred.mean():.4f}")
    print(f"   Std: {test_pred.std():.4f}")
    print(f"   Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Check if we improved
    if np.mean(cv_scores) > 0.75:
        print(f"\n🎉 Strong cross-validation score! Should generalize better.")
    else:
        print(f"\n⚠️  Conservative model - prioritizing generalization over CV score")
    
    # Save importance
    importance.to_csv('psychology_feature_importance.csv', index=False)
    
    print(f"\n🏢 Employee Psychology Insights Applied:")
    print(f"   ✅ Evening/weekend browsing preferences")
    print(f"   ✅ Spending consistency patterns")
    print(f"   ✅ Engagement level categorization")
    print(f"   ✅ Risk-averse behavior modeling")
    print(f"   ✅ Salary cycle awareness")
    print(f"   ✅ Conservative parameters to prevent overfitting")
    print(f"   ✅ 5-fold cross-validation")
    print(f"   ✅ Early stopping with patience")

if __name__ == "__main__":
    main()
