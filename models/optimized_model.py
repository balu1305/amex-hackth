"""
AMEX Round 2 - Optimized Model for Better Performance
====================================================
This script addresses common issues that cause low AUC scores:
1. Better feature engineering
2. Proper cross-validation
3. Optimized hyperparameters
4. Identifier variable handling
5. Better target encoding
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load data with proper preprocessing"""
    print("Loading and preprocessing data...")
    
    # Load main datasets
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Convert target to numeric properly
    train_data['y'] = pd.to_numeric(train_data['y'], errors='coerce')
    
    # Check target distribution
    target_dist = train_data['y'].value_counts()
    print(f"Target distribution: {target_dist[0]} (0), {target_dist[1]} (1)")
    print(f"Positive rate: {target_dist[1]/(target_dist[0]+target_dist[1])*100:.2f}%")
    
    return train_data, test_data

def feature_engineering(train_data, test_data):
    """Advanced feature engineering"""
    print("Starting feature engineering...")
    
    # Combine datasets for consistent preprocessing
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    test_data['y'] = -1  # Placeholder
    
    combined = pd.concat([train_data, test_data], ignore_index=True)
    
    # Convert all f-columns to numeric
    feature_cols = [col for col in combined.columns if col.startswith('f')]
    print(f"Converting {len(feature_cols)} features to numeric...")
    
    for col in feature_cols:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
    
    # Load additional data for feature engineering
    try:
        print("Loading additional datasets...")
        add_event = pd.read_parquet('add_event.parquet')
        add_trans = pd.read_parquet('add_trans.parquet')
        offer_metadata = pd.read_parquet('offer_metadata.parquet')
        
        # Convert additional data to numeric
        for col in add_event.columns:
            if col != 'id2':
                add_event[col] = pd.to_numeric(add_event[col], errors='coerce')
        
        for col in add_trans.columns:
            if col != 'id2':
                add_trans[col] = pd.to_numeric(add_trans[col], errors='coerce')
        
        # Create aggregated features from additional data
        print("Creating features from additional event data...")
        event_agg = add_event.groupby('id2').agg({
            'id6': ['count', 'nunique'],
            'id4': ['min', 'max', 'mean', 'std']
        }).round(4)
        event_agg.columns = ['event_count', 'unique_events', 'first_event', 'last_event', 'avg_event_time', 'event_time_std']
        event_agg['event_time_span'] = event_agg['last_event'] - event_agg['first_event']
        event_agg = event_agg.reset_index()
        
        print("Creating features from additional transaction data...")
        trans_agg = add_trans.groupby('id2').agg({
            'f367': ['sum', 'mean', 'count', 'std'],
            'f368': ['sum', 'mean', 'std'],
            'f369': ['sum', 'mean', 'std'],
            'f370': ['sum', 'mean', 'std']
        }).round(4)
        trans_agg.columns = ['total_f367', 'avg_f367', 'count_f367', 'std_f367',
                            'sum_f368', 'avg_f368', 'std_f368',
                            'sum_f369', 'avg_f369', 'std_f369', 
                            'sum_f370', 'avg_f370', 'std_f370']
        trans_agg = trans_agg.reset_index()
        
        # Merge additional features
        combined = combined.merge(event_agg, on='id2', how='left')
        combined = combined.merge(trans_agg, on='id2', how='left')
        combined = combined.merge(offer_metadata, on='id3', how='left')
        
        print(f"Shape after merging additional data: {combined.shape}")
        
    except Exception as e:
        print(f"Warning: Could not load additional data: {e}")
        print("Continuing with main features only...")
    
    # Create interaction features
    print("Creating interaction features...")
    
    # CTR-based features (avoiding division by zero)
    if 'f310' in combined.columns and 'f314' in combined.columns:
        combined['ctr_momentum'] = combined['f310'] / (combined['f314'] + 1e-8)
    
    if 'f319' in combined.columns and 'f324' in combined.columns:
        combined['click_impression_ratio'] = combined['f319'] / (combined['f324'] + 1e-8)
    
    # Customer value features
    if all(col in combined.columns for col in ['f39', 'f40', 'f41', 'f185']):
        combined['total_spend'] = combined['f39'] + combined['f40'] + combined['f41']
        combined['spend_per_transaction'] = combined['total_spend'] / (combined['f185'] + 1)
    
    # Time-based features
    if 'f349' in combined.columns:
        combined['is_weekend'] = combined['f349'].isin([6, 7]).astype(int)
        combined['is_friday'] = (combined['f349'] == 5).astype(int)
    
    if 'f350' in combined.columns:
        combined['hour_of_day'] = (combined['f350'] / 3600).astype(int)
        combined['time_bin'] = pd.cut(combined['f350'], bins=4, labels=[0,1,2,3]).astype(float)
    
    # Engagement features
    if 'f366' in combined.columns and 'f363' in combined.columns:
        combined['engagement_score'] = combined['f366'] * combined['f363']
    
    # Handle missing values intelligently
    print("Handling missing values...")
    
    # For numerical features, use median
    numerical_cols = combined.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if combined[col].isnull().sum() > 0:
            median_val = combined[col].median()
            combined[col].fillna(median_val, inplace=True)
            # Create missingness indicator for important features
            if col in ['f43', 'f47', 'f366', 'f363'] and combined[col].isnull().sum() > 1000:
                combined[f'{col}_was_missing'] = combined[col].isnull().astype(int)
    
    # For categorical features, use mode or 'unknown'
    categorical_cols = combined.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['id1', 'id2', 'id3'] and combined[col].isnull().sum() > 0:
            mode_val = combined[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
            combined[col].fillna(fill_val, inplace=True)
    
    # Encode categorical variables (excluding identifiers)
    for col in categorical_cols:
        if col not in ['id1', 'id2', 'id3', 'id4', 'id5']:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
    
    # Split back to train and test
    train_processed = combined[combined['is_train'] == 1].copy()
    test_processed = combined[combined['is_train'] == 0].copy()
    
    train_processed = train_processed.drop(['is_train'], axis=1)
    test_processed = test_processed.drop(['is_train', 'y'], axis=1)
    
    print(f"Final train shape: {train_processed.shape}")
    print(f"Final test shape: {test_processed.shape}")
    
    return train_processed, test_processed

def select_features(train_data):
    """Select features excluding identifiers"""
    # Exclude identifier columns as per rules
    identifier_cols = ['id1', 'id2', 'id3', 'id4', 'id5', 'y']
    feature_cols = [col for col in train_data.columns if col not in identifier_cols]
    
    print(f"Selected {len(feature_cols)} features (excluding identifiers)")
    return feature_cols

def train_optimized_model(X_train, y_train, X_val, y_val):
    """Train LightGBM with optimized parameters"""
    print("Training optimized LightGBM model...")
    
    # Optimized parameters for better performance
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.02,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 1.0,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'max_depth': 8,
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 42
    }
    
    # Create datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    # Train with early stopping
    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model

def cross_validate_model(X, y, feature_cols):
    """Perform cross-validation for robust evaluation"""
    print("Performing 5-fold cross-validation...")
    
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/5...")
        
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = train_optimized_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        
        # Predict and evaluate
        val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val_fold, val_pred)
        cv_scores.append(fold_auc)
        
        print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"CV Results: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    return mean_cv_score, std_cv_score

def main():
    """Main execution function"""
    print("AMEX Round 2 - Optimized Solution")
    print("=" * 50)
    
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data()
    
    # Feature engineering
    train_processed, test_processed = feature_engineering(train_data, test_data)
    
    # Select features (excluding identifiers)
    feature_cols = select_features(train_processed)
    
    # Prepare data
    X = train_processed[feature_cols].copy()
    y = train_processed['y'].copy()
    X_test = test_processed[feature_cols].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Cross-validation
    cv_mean, cv_std = cross_validate_model(X, y, feature_cols)
    
    # Train final model on full data
    print("Training final model on full dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    final_model = train_optimized_model(X_train, y_train, X_val, y_val)
    
    # Final validation
    val_pred = final_model.predict(X_val, num_iteration=final_model.best_iteration)
    final_auc = roc_auc_score(y_val, val_pred)
    print(f"Final validation AUC: {final_auc:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("Top 15 most important features:")
    for i, row in feature_importance.head(15).iterrows():
        print(f"{row.name+1:2d}. {row['feature']}: {row['importance']:,.0f}")
    
    # Generate test predictions
    print("Generating test predictions...")
    test_predictions = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    
    # Create submission
    submission_template = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission_template['pred'] = test_predictions
    
    # Save with proper naming convention
    team_name = "TVIJAYABALAJI"  # Replace with your team name
    submission_filename = f'r2_submission_file{team_name}.csv'
    submission_template.to_csv(submission_filename, index=False)
    
    print(f"Submission saved as: {submission_filename}")
    print(f"Prediction statistics:")
    print(f"  Mean: {test_predictions.mean():.4f}")
    print(f"  Std: {test_predictions.std():.4f}")
    print(f"  Min: {test_predictions.min():.4f}")
    print(f"  Max: {test_predictions.max():.4f}")
    print(f"  Unique values: {len(np.unique(test_predictions))}")
    
    print(f"\nModel Performance Summary:")
    print(f"Cross-validation AUC: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Final validation AUC: {final_auc:.4f}")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_optimized.csv', index=False)
    print("Feature importance saved to: feature_importance_optimized.csv")

if __name__ == "__main__":
    main()
