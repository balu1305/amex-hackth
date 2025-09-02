"""
AMEX Round 2 - Employee Psychology Based Robust Model
=====================================================
Built for real-world working employee behavior patterns with strong overfitting prevention
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import gc
import warnings
import re
warnings.filterwarnings('ignore')

def analyze_data_patterns(train_data):
    """Deep analysis of customer patterns"""
    print("🧠 Analyzing Employee Psychology Patterns...")
    print("=" * 60)
    
    # Target distribution
    target_dist = train_data['y'].value_counts(normalize=True)
    print(f"📊 Target Distribution:")
    print(f"   No Click: {target_dist[0]:.1%}")
    print(f"   Click: {target_dist[1]:.1%}")
    print(f"   Class Imbalance Ratio: {target_dist[0]/target_dist[1]:.1f}:1")
    
    # Find timing-related features
    print(f"\n⏰ Searching for Time Patterns...")
    time_features = []
    
    # Look for features that might represent time/day patterns
    for col in train_data.columns:
        if col.startswith('f'):
            try:
                values = pd.to_numeric(train_data[col], errors='coerce')
                if not values.isna().all():
                    # Check if values look like time (0-23 for hours, 0-6 for days)
                    if values.min() >= 0 and values.max() <= 23 and len(values.unique()) <= 24:
                        print(f"   🕐 {col}: Possible hour feature (0-{values.max():.0f})")
                        time_features.append((col, 'hour'))
                    elif values.min() >= 0 and values.max() <= 6 and len(values.unique()) <= 7:
                        print(f"   📅 {col}: Possible day feature (0-{values.max():.0f})")
                        time_features.append((col, 'day'))
            except:
                pass
    
    return time_features

def create_employee_psychology_features(train_data, test_data):
    """Create features based on working employee psychology"""
    print("\n🧠 Creating Employee Psychology Features...")
    
    # Get numeric features only
    feature_cols = [col for col in train_data.columns if col.startswith('f')]
    
    print(f"Processing {len(feature_cols)} base features...")
    
    # Convert features to numeric
    for col in feature_cols:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    
    # === EMPLOYEE BEHAVIOR PATTERNS ===
    
    # 1. Evening/Night Browsing (Working employees browse offers after work)
    print("📱 Creating evening/night browsing patterns...")
    
    # Find time-related features and create evening patterns
    for col in feature_cols:
        if col in ['f349', 'f350', 'f351']:  # Common time feature names
            values = train_data[col].fillna(0)
            if values.max() <= 23:  # Likely hour feature
                # Evening hours (6 PM - 11 PM) when employees check phones
                train_data[f'{col}_evening'] = ((values >= 18) & (values <= 23)).astype(int)
                test_data[f'{col}_evening'] = ((test_data[col].fillna(0) >= 18) & (test_data[col].fillna(0) <= 23)).astype(int)
                
                # Late night (9 PM+) when employees have time to think
                train_data[f'{col}_late_night'] = (values >= 21).astype(int)
                test_data[f'{col}_late_night'] = (test_data[col].fillna(0) >= 21).astype(int)
                
                # Weekend patterns (employees more relaxed)
            elif values.max() <= 6:  # Likely day of week
                train_data[f'{col}_weekend'] = (values >= 5).astype(int)  # Fri, Sat, Sun
                test_data[f'{col}_weekend'] = (test_data[col].fillna(0) >= 5).astype(int)
    
    # 2. Spending Habit Consistency (Regular vs Irregular spenders)
    print("💰 Creating spending consistency patterns...")
    
    # Find spending-related features
    spend_features = []
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if values.std() > 0 and values.max() > 100:  # Likely spending amounts
            spend_features.append(col)
    
    if len(spend_features) >= 3:
        spend_cols = spend_features[:5]  # Take top 5 spending features
        
        # Total spending pattern
        train_data['total_spend'] = train_data[spend_cols].sum(axis=1)
        test_data['total_spend'] = test_data[spend_cols].sum(axis=1)
        
        # Spending consistency (low std = regular spender)
        train_data['spend_consistency'] = 1 / (train_data[spend_cols].std(axis=1) + 1)
        test_data['spend_consistency'] = 1 / (test_data[spend_cols].std(axis=1) + 1)
        
        # High value customer flag
        spend_80th = train_data['total_spend'].quantile(0.8)
        train_data['high_value_customer'] = (train_data['total_spend'] > spend_80th).astype(int)
        test_data['high_value_customer'] = (test_data['total_spend'] > spend_80th).astype(int)
    
    # 3. Engagement Frequency (How often customer interacts)
    print("🔄 Creating engagement frequency patterns...")
    
    # Find frequency/count features
    freq_features = []
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if values.dtype in ['int64', 'float64'] and values.max() < 1000 and values.min() >= 0:
            # Check if it looks like a count (integer-like values)
            if (values % 1 == 0).sum() / len(values) > 0.8:
                freq_features.append(col)
    
    if len(freq_features) >= 3:
        freq_cols = freq_features[:5]
        
        # Total engagement
        train_data['total_engagement'] = train_data[freq_cols].sum(axis=1)
        test_data['total_engagement'] = test_data[freq_cols].sum(axis=1)
        
        # Regular user flag (consistent engagement)
        train_data['regular_user'] = (train_data['total_engagement'] > train_data['total_engagement'].median()).astype(int)
        test_data['regular_user'] = (test_data['total_engagement'] > train_data['total_engagement'].median()).astype(int)
    
    # 4. Offer Relevance (Based on past behavior)
    print("🎯 Creating offer relevance patterns...")
    
    # Category-based features (if available)
    category_features = []
    for col in feature_cols:
        values = train_data[col].fillna(0)
        # Look for categorical patterns (limited unique values)
        if len(values.unique()) < 50 and len(values.unique()) > 2:
            category_features.append(col)
    
    if len(category_features) >= 2:
        # Category diversity (employees who try different things)
        cat_cols = category_features[:3]
        for col in cat_cols:
            # Normalize categories to 0-1 scale
            col_max = max(train_data[col].max(), test_data[col].max())
            if col_max > 0:
                train_data[f'{col}_normalized'] = train_data[col] / col_max
                test_data[f'{col}_normalized'] = test_data[col] / col_max
    
    # 5. Risk-Averse Behavior (Employees are generally more careful)
    print("⚠️ Creating risk-averse behavior patterns...")
    
    # Create conservative spending flags
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if values.std() > 0:
            # Flag for conservative behavior (below median spending/activity)
            median_val = values.median()
            train_data[f'{col}_conservative'] = (values <= median_val).astype(int)
            test_data[f'{col}_conservative'] = (test_data[col].fillna(0) <= median_val).astype(int)
    
    # 6. Seasonal/Monthly Patterns (Employees have salary cycles)
    print("📅 Creating salary cycle patterns...")
    
    # If we have date-related features, create month patterns
    for col in feature_cols:
        values = train_data[col].fillna(0)
        if values.max() <= 31 and values.min() >= 1:  # Possible day of month
            # Payday patterns (end of month/beginning)
            train_data[f'{col}_payday'] = ((values <= 5) | (values >= 25)).astype(int)
            test_data[f'{col}_payday'] = ((test_data[col].fillna(0) <= 5) | (test_data[col].fillna(0) >= 25)).astype(int)
    
    print(f"✅ Created psychology-based features")
    print(f"Final train shape: {train_data.shape}")
    print(f"Final test shape: {test_data.shape}")
    
    return train_data, test_data

def create_robust_model_with_overfitting_prevention():
    """Create robust model with strong overfitting prevention"""
    
    # Advanced LightGBM parameters to prevent overfitting
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        
        # Prevent overfitting - Conservative settings
        'num_leaves': 31,           # Much smaller (was 128)
        'learning_rate': 0.01,      # Lower learning rate
        'feature_fraction': 0.6,    # Use only 60% features each tree
        'bagging_fraction': 0.7,    # Use only 70% data each tree
        'bagging_freq': 5,          # Frequent bagging
        'min_data_in_leaf': 200,    # Larger leaf size (was 100)
        
        # Regularization - Stronger
        'lambda_l1': 2.0,           # Stronger L1 (was 1.0)
        'lambda_l2': 2.0,           # Stronger L2 (was 1.0)
        'max_depth': 6,             # Shallower trees (was 10)
        
        # Early stopping
        'early_stopping_round': 200,
        
        # Class imbalance
        'is_unbalance': True,
        'scale_pos_weight': 20,     # Strong weight for minority class
        
        'verbosity': -1,
        'random_state': 42,
        'force_col_wise': True,
        'extra_trees': True,        # More randomness
    }
    
    return params

def robust_cross_validation(X, y, params, n_splits=5):
    """Robust cross-validation with multiple metrics"""
    print(f"\n🔄 Robust {n_splits}-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    feature_importance = pd.DataFrame()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n📊 Fold {fold + 1}/{n_splits}")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Create datasets
        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)
        
        # Train model
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_val_fold, y_pred)
        cv_scores.append(auc_score)
        
        print(f"   AUC: {auc_score:.4f}")
        
        # Collect feature importance
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(),
            'fold': fold + 1
        })
        feature_importance = pd.concat([feature_importance, fold_importance])
        
        # Clean memory
        del model, train_set, val_set
        gc.collect()
    
    print(f"\n📈 Cross-Validation Results:")
    print(f"   Mean AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"   Individual Folds: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Feature importance summary
    importance_summary = feature_importance.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    importance_summary = importance_summary.sort_values('mean', ascending=False)
    
    return cv_scores, importance_summary

def main():
    """Main execution with employee psychology focus"""
    print("🏢 AMEX Round 2 - Employee Psychology Robust Model")
    print("=" * 60)
    
    # Load data
    print("📂 Loading data...")
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Analyze patterns
    time_features = analyze_data_patterns(train_data)
    
    # Create psychology-based features
    train_data, test_data = create_employee_psychology_features(train_data, test_data)
    
    # Prepare features
    exclude_cols = ['id1', 'id2', 'id3', 'id4', 'id5', 'y']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"\n🎯 Using {len(feature_cols)} features for robust model")
    
    # Handle missing values robustly
    print("🔧 Robust missing value handling...")
    for col in feature_cols:
        # Use mode for categorical-like features, median for continuous
        if train_data[col].nunique() < 20:
            fill_value = train_data[col].mode().iloc[0] if len(train_data[col].mode()) > 0 else 0
        else:
            fill_value = train_data[col].median()
        
        train_data[col] = train_data[col].fillna(fill_value)
        test_data[col] = test_data[col].fillna(fill_value)
    
    # Prepare final datasets
    X = train_data[feature_cols].copy()
    y = train_data['y'].copy()
    X_test = test_data[feature_cols].copy()
    
    # Clean memory
    del train_data, test_data
    gc.collect()
    
    print(f"Final feature matrix: {X.shape}")
    print(f"Final test matrix: {X_test.shape}")
    
    # Get robust model parameters
    params = create_robust_model_with_overfitting_prevention()
    
    # Robust cross-validation
    cv_scores, feature_importance = robust_cross_validation(X, y, params, n_splits=5)
    
    # Train final model on all data
    print(f"\n🚀 Training final robust model...")
    
    # Split for final validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    final_model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Final validation
    val_pred = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    final_auc = roc_auc_score(y_val, val_pred)
    
    print(f"\n🎯 Final Validation AUC: {final_auc:.4f}")
    
    # Feature importance analysis
    print(f"\n🔍 Top Psychology Features:")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']}: {row['mean']:,.0f} (±{row['std']:.0f})")
    
    # Generate test predictions
    print(f"\n📊 Generating robust test predictions...")
    test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save submission
    team_name = "TVIJAYABALAJI"
    submission_file = f'r2_submission_file{team_name}_robust.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"\n✅ Robust Employee Psychology Model Complete!")
    print(f"📁 Submission saved: {submission_file}")
    print(f"🎯 Cross-Val AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"🎯 Final AUC: {final_auc:.4f}")
    
    print(f"\n📊 Prediction Statistics:")
    print(f"   Mean: {test_pred.mean():.4f}")
    print(f"   Std: {test_pred.std():.4f}")
    print(f"   Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_robust.csv', index=False)
    
    print(f"\n🏢 Employee Psychology Insights Applied:")
    print(f"   ✅ Evening/night browsing patterns")
    print(f"   ✅ Spending consistency (regular vs irregular)")
    print(f"   ✅ Risk-averse behavior modeling")
    print(f"   ✅ Salary cycle patterns")
    print(f"   ✅ Strong overfitting prevention")
    print(f"   ✅ 5-fold cross-validation")

if __name__ == "__main__":
    main()
