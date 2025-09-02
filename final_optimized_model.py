"""
AMEX Round 2 - Memory-Efficient Optimized Model
===============================================
Fixed version that handles memory efficiently and avoids common pitfalls
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

def optimize_dtypes(df):
    """Optimize data types to save memory"""
    for col in df.columns:
        if df[col].dtype == 'object':
            continue
        
        col_min = df[col].min()
        col_max = df[col].max()
        
        if str(df[col].dtype)[:3] == 'int':
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float32)
    
    return df

def load_and_preprocess():
    """Memory-efficient data loading"""
    print("Loading data with memory optimization...")
    
    # Load main data
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Convert target properly
    train_data['y'] = pd.to_numeric(train_data['y']).astype(np.int8)
    
    # Check target
    print(f"Target distribution: {train_data['y'].value_counts().to_dict()}")
    
    return train_data, test_data

def create_features(train_data, test_data):
    """Efficient feature creation"""
    print("Creating features efficiently...")
    
    # Get feature columns
    feature_cols = [col for col in train_data.columns if col.startswith('f')]
    print(f"Processing {len(feature_cols)} features...")
    
    # Convert features to numeric in chunks
    for i, col in enumerate(feature_cols):
        if i % 50 == 0:
            print(f"Processing feature {i+1}/{len(feature_cols)}")
        
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce').astype(np.float32)
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce').astype(np.float32)
    
    # Create simple interaction features
    print("Creating interaction features...")
    
    # CTR ratios (avoiding memory issues)
    if 'f310' in train_data.columns and 'f314' in train_data.columns:
        train_data['ctr_ratio'] = (train_data['f310'] / (train_data['f314'] + 1e-8)).astype(np.float32)
        test_data['ctr_ratio'] = (test_data['f310'] / (test_data['f314'] + 1e-8)).astype(np.float32)
    
    # Click-impression ratio
    if 'f319' in train_data.columns and 'f324' in train_data.columns:
        train_data['click_imp_ratio'] = (train_data['f319'] / (train_data['f324'] + 1e-8)).astype(np.float32)
        test_data['click_imp_ratio'] = (test_data['f319'] / (test_data['f324'] + 1e-8)).astype(np.float32)
    
    # Time features
    if 'f349' in train_data.columns:
        train_data['is_weekend'] = train_data['f349'].isin([6, 7]).astype(np.int8)
        test_data['is_weekend'] = test_data['f349'].isin([6, 7]).astype(np.int8)
    
    if 'f350' in train_data.columns:
        train_data['hour_bin'] = (train_data['f350'] / 3600).astype(np.int8) 
        test_data['hour_bin'] = (test_data['f350'] / 3600).astype(np.int8)
    
    # Spending features
    spend_cols = ['f39', 'f40', 'f41']
    if all(col in train_data.columns for col in spend_cols):
        train_data['total_spend'] = (train_data[spend_cols].sum(axis=1)).astype(np.float32)
        test_data['total_spend'] = (test_data[spend_cols].sum(axis=1)).astype(np.float32)
    
    # Handle missing values efficiently
    print("Handling missing values...")
    
    # Get all numeric columns
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['id2', 'y']]
    
    for col in numeric_cols:
        # Fill with median
        median_val = train_data[col].median()
        train_data[col] = train_data[col].fillna(median_val)
        test_data[col] = test_data[col].fillna(median_val)
    
    print(f"Final train shape: {train_data.shape}")
    print(f"Final test shape: {test_data.shape}")
    
    return train_data, test_data

def train_lgb_model(X_train, y_train, X_val, y_val):
    """Train optimized LightGBM"""
    print("Training LightGBM with optimized parameters...")
    
    # Better parameters for this dataset
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 128,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'max_depth': 10,
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 42,
        'force_col_wise': True  # For better memory usage
    }
    
    # Create datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    # Train
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(period=200)
        ]
    )
    
    return model

def main():
    """Main execution"""
    print("AMEX Round 2 - Memory-Efficient Solution")
    print("=" * 50)
    
    # Load data
    train_data, test_data = load_and_preprocess()
    
    # Feature engineering
    train_data, test_data = create_features(train_data, test_data)
    
    # Select features (exclude identifiers)
    exclude_cols = ['id1', 'id2', 'id3', 'id4', 'id5', 'y']
    feature_cols = [col for col in train_data.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features")
    
    # Prepare data
    X = train_data[feature_cols].copy()
    y = train_data['y'].copy()
    X_test = test_data[feature_cols].copy()
    
    # Clear memory
    del train_data, test_data
    gc.collect()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Test matrix shape: {X_test.shape}")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    # Train model
    model = train_lgb_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc_score = roc_auc_score(y_val, val_pred)
    
    print(f"\nValidation AUC: {auc_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']}: {row['importance']:,.0f}")
    
    # Generate test predictions
    print("\nGenerating test predictions...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Load submission template
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_pred
    
    # Save with proper naming
    team_name = "TVIJAYABALAJI"
    submission_file = f'r2_submission_file{team_name}.csv'
    submission.to_csv(submission_file, index=False)
    
    print(f"Submission saved as: {submission_file}")
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {test_pred.mean():.4f}")
    print(f"  Std: {test_pred.std():.4f}")
    print(f"  Min: {test_pred.min():.4f}")
    print(f"  Max: {test_pred.max():.4f}")
    print(f"  Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
    
    # Validation of submission
    print(f"\nSubmission Validation:")
    print(f"  Shape: {submission.shape}")
    print(f"  Columns: {list(submission.columns)}")
    print(f"  Missing values: {submission.isnull().sum().sum()}")
    
    print(f"\nFinal AUC Score: {auc_score:.4f}")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_final.csv', index=False)
    print("Feature importance saved to: feature_importance_final.csv")
    
    if auc_score > 0.7:
        print("✅ Model performance looks good!")
    else:
        print("⚠️  Model performance is low. Consider:")
        print("   - More feature engineering")
        print("   - Different hyperparameters") 
        print("   - Ensemble methods")

if __name__ == "__main__":
    main()
