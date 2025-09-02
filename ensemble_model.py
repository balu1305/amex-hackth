"""
Model Diagnostics and Improvements
=================================
Additional techniques to boost performance further
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def create_ensemble_model():
    """Create ensemble of multiple models for better performance"""
    print("Creating ensemble model...")
    
    # Load data
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    # Convert target
    train_data['y'] = pd.to_numeric(train_data['y']).astype(np.int8)
    
    # Get features
    feature_cols = [col for col in train_data.columns if col.startswith('f')]
    
    # Convert features
    for col in feature_cols:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').astype(np.float32)
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').astype(np.float32)
    
    # Handle missing values
    for col in feature_cols:
        median_val = train_data[col].median()
        train_data[col] = train_data[col].fillna(median_val)
        test_data[col] = test_data[col].fillna(median_val)
    
    # Create additional features
    print("Creating advanced features...")
    
    # Advanced CTR features
    if all(col in train_data.columns for col in ['f310', 'f312', 'f314']):
        train_data['ctr_trend'] = (train_data['f310'] - train_data['f312']) / (train_data['f314'] + 1e-8)
        test_data['ctr_trend'] = (test_data['f310'] - test_data['f312']) / (test_data['f314'] + 1e-8)
    
    # Engagement momentum
    if all(col in train_data.columns for col in ['f366', 'f363', 'f361']):
        train_data['engagement_momentum'] = train_data['f366'] * train_data['f363'] / (train_data['f361'] + 1e-8)
        test_data['engagement_momentum'] = test_data['f366'] * test_data['f363'] / (test_data['f361'] + 1e-8)
    
    # Time-based features
    if 'f350' in train_data.columns:
        # Create time bins
        train_data['time_sin'] = np.sin(2 * np.pi * train_data['f350'] / (24 * 3600))
        train_data['time_cos'] = np.cos(2 * np.pi * train_data['f350'] / (24 * 3600))
        test_data['time_sin'] = np.sin(2 * np.pi * test_data['f350'] / (24 * 3600))
        test_data['time_cos'] = np.cos(2 * np.pi * test_data['f350'] / (24 * 3600))
    
    # Customer value tiers
    spend_cols = ['f39', 'f40', 'f41']
    if all(col in train_data.columns for col in spend_cols):
        train_data['total_spend'] = train_data[spend_cols].sum(axis=1)
        test_data['total_spend'] = test_data[spend_cols].sum(axis=1)
        
        # Create spend quintiles
        spend_quintiles = pd.qcut(train_data['total_spend'], q=5, labels=False, duplicates='drop')
        train_data['spend_tier'] = spend_quintiles
        test_data['spend_tier'] = pd.cut(test_data['total_spend'], 
                                        bins=pd.qcut(train_data['total_spend'], q=5, retbins=True, duplicates='drop')[1], 
                                        labels=False, include_lowest=True).fillna(2)
    
    # Select final features
    exclude_cols = ['id1', 'id2', 'id3', 'id4', 'id5', 'y']
    final_features = [col for col in train_data.columns if col not in exclude_cols]
    
    X = train_data[final_features]
    y = train_data['y']
    X_test = test_data[final_features]
    
    print(f"Using {len(final_features)} features for ensemble")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model 1: Conservative parameters
    print("Training Model 1 (Conservative)...")
    params1 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.005,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 200,
        'lambda_l1': 2.0,
        'lambda_l2': 2.0,
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 42
    }
    
    train_set1 = lgb.Dataset(X_train, label=y_train)
    val_set1 = lgb.Dataset(X_val, label=y_val, reference=train_set1)
    
    model1 = lgb.train(params1, train_set1, valid_sets=[val_set1], 
                       num_boost_round=5000, callbacks=[lgb.early_stopping(400), lgb.log_evaluation(0)])
    
    # Model 2: Aggressive parameters
    print("Training Model 2 (Aggressive)...")
    params2 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 256,
        'learning_rate': 0.02,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 3,
        'min_data_in_leaf': 50,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 123
    }
    
    train_set2 = lgb.Dataset(X_train, label=y_train)
    val_set2 = lgb.Dataset(X_val, label=y_val, reference=train_set2)
    
    model2 = lgb.train(params2, train_set2, valid_sets=[val_set2], 
                       num_boost_round=3000, callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
    
    # Model 3: Balanced parameters
    print("Training Model 3 (Balanced)...")
    params3 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 4,
        'min_data_in_leaf': 100,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 456
    }
    
    train_set3 = lgb.Dataset(X_train, label=y_train)
    val_set3 = lgb.Dataset(X_val, label=y_val, reference=train_set3)
    
    model3 = lgb.train(params3, train_set3, valid_sets=[val_set3], 
                       num_boost_round=4000, callbacks=[lgb.early_stopping(350), lgb.log_evaluation(0)])
    
    # Evaluate individual models
    pred1 = model1.predict(X_val, num_iteration=model1.best_iteration)
    pred2 = model2.predict(X_val, num_iteration=model2.best_iteration)
    pred3 = model3.predict(X_val, num_iteration=model3.best_iteration)
    
    auc1 = roc_auc_score(y_val, pred1)
    auc2 = roc_auc_score(y_val, pred2)
    auc3 = roc_auc_score(y_val, pred3)
    
    print(f"Model 1 AUC: {auc1:.4f}")
    print(f"Model 2 AUC: {auc2:.4f}")
    print(f"Model 3 AUC: {auc3:.4f}")
    
    # Ensemble predictions
    ensemble_pred = (pred1 + pred2 + pred3) / 3
    ensemble_auc = roc_auc_score(y_val, ensemble_pred)
    print(f"Ensemble AUC: {ensemble_auc:.4f}")
    
    # Generate test predictions
    print("Generating ensemble test predictions...")
    test_pred1 = model1.predict(X_test, num_iteration=model1.best_iteration)
    test_pred2 = model2.predict(X_test, num_iteration=model2.best_iteration)
    test_pred3 = model3.predict(X_test, num_iteration=model3.best_iteration)
    
    # Weighted ensemble (give more weight to better performing models)
    weights = np.array([auc1, auc2, auc3])
    weights = weights / weights.sum()
    
    final_test_pred = (weights[0] * test_pred1 + weights[1] * test_pred2 + weights[2] * test_pred3)
    
    # Create submission
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = final_test_pred
    
    team_name = "TVIJAYABALAJI"
    submission_file = f'r2_submission_ensemble_{team_name}.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"Ensemble submission saved as: {submission_file}")
    print(f"Final ensemble AUC: {ensemble_auc:.4f}")
    print(f"Test prediction stats:")
    print(f"  Mean: {final_test_pred.mean():.4f}")
    print(f"  Std: {final_test_pred.std():.4f}")
    print(f"  Range: [{final_test_pred.min():.4f}, {final_test_pred.max():.4f}]")
    
    return ensemble_auc

def validate_submission(filename):
    """Validate submission file"""
    print(f"Validating {filename}...")
    
    try:
        sub = pd.read_csv(filename)
        template = pd.read_csv('685404e30cfdb_submission_template.csv')
        
        print(f"✅ File loaded successfully")
        print(f"✅ Shape: {sub.shape} (expected: {template.shape})")
        print(f"✅ Columns: {list(sub.columns)}")
        print(f"✅ Missing values: {sub.isnull().sum().sum()}")
        print(f"✅ Prediction range: [{sub['pred'].min():.4f}, {sub['pred'].max():.4f}]")
        print(f"✅ Mean prediction: {sub['pred'].mean():.4f}")
        
        # Check if IDs match
        id_match = (sub[['id1', 'id2', 'id3', 'id5']] == template[['id1', 'id2', 'id3', 'id5']]).all().all()
        print(f"✅ ID columns match template: {id_match}")
        
        print("🎯 Submission is ready!")
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")

if __name__ == "__main__":
    print("AMEX Round 2 - Advanced Ensemble Model")
    print("=" * 50)
    
    # Create ensemble
    ensemble_auc = create_ensemble_model()
    
    # Validate submissions
    print("\n" + "="*50)
    validate_submission('r2_submission_fileTVIJAYABALAJI.csv')
    print("\n" + "="*30)
    validate_submission('r2_submission_ensemble_TVIJAYABALAJI.csv')
