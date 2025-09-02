"""
AMEX Campus Challenge Round 2: Offer Click Prediction
=====================================================

Problem Statement:
- Predict probability of a customer clicking on an offer given that they have seen it
- Binary classification problem (click = 1, no click = 0)
- Goal: Improve customer engagement with offers

Data Overview:
- Train data: 770,164 samples with 372 features
- Test data: 369,301 samples 
- Target variable 'y': Highly imbalanced (733,113 no clicks vs 37,051 clicks - 4.8% positive rate)
- Features include customer behavior, offer characteristics, transaction history, etc.

Key Features Categories:
1. Customer Interest Scores (f1-f12): Interest in various categories
2. Web Behavior (f13-f21): Page visits, interactions
3. Channel Usage (f22-f27): MYCA login channels
4. Offer Engagement (f28-f35): Past impressions/clicks
5. Customer Service (f36-f38): Call history, emails
6. Spending Patterns (f39-f41): Category-wise spending
7. Loyalty Program (f42-f58): Membership details, miles
8. Online Activity (f59-f93): Time spent on pages
9. Offer Performance (f94-f366): CTR, impressions by category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all data files"""
    print("Loading data files...")
    
    # Main datasets
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    # Additional data
    add_event = pd.read_parquet('add_event.parquet')
    add_trans = pd.read_parquet('add_trans.parquet') 
    offer_metadata = pd.read_parquet('offer_metadata.parquet')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Additional event data: {add_event.shape}")
    print(f"Additional transaction data: {add_trans.shape}")
    print(f"Offer metadata: {offer_metadata.shape}")
    
    return train_data, test_data, add_event, add_trans, offer_metadata

def basic_eda(train_data):
    """Perform basic exploratory data analysis"""
    print("\n=== BASIC EDA ===")
    
    # Target distribution
    target_dist = train_data['y'].value_counts()
    print(f"\nTarget distribution:")
    print(f"No Click (0): {target_dist[0]:,} ({target_dist[0]/len(train_data)*100:.2f}%)")
    print(f"Click (1): {target_dist[1]:,} ({target_dist[1]/len(train_data)*100:.2f}%)")
    
    # Missing values
    missing_counts = train_data.isnull().sum()
    missing_features = missing_counts[missing_counts > 0].sort_values(ascending=False)
    print(f"\nFeatures with missing values: {len(missing_features)}")
    if len(missing_features) > 0:
        print("Top 10 features with most missing values:")
        print(missing_features.head(10))
    
    # Data types
    print(f"\nData types distribution:")
    print(train_data.dtypes.value_counts())
    
    return target_dist, missing_features

def feature_engineering(train_data, test_data, add_event, add_trans, offer_metadata):
    """Perform feature engineering"""
    print("\n=== FEATURE ENGINEERING ===")
    
    # Combine train and test for consistent preprocessing
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    test_data['y'] = -1  # Placeholder for test
    
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Merge with offer metadata
    print("Merging with offer metadata...")
    combined_data = combined_data.merge(offer_metadata, on='id3', how='left')
    
    # Feature engineering on additional event data
    print("Engineering features from additional event data...")
    event_features = add_event.groupby('id2').agg({
        'id6': ['count', 'nunique'],  # Event count and unique events
        'id4': ['min', 'max']  # Time range of events
    }).round(4)
    event_features.columns = ['event_count', 'unique_events', 'first_event_time', 'last_event_time']
    event_features['event_time_span'] = event_features['last_event_time'] - event_features['first_event_time']
    event_features = event_features.reset_index()
    
    # Feature engineering on additional transaction data  
    print("Engineering features from additional transaction data...")
    trans_features = add_trans.groupby('id2').agg({
        'f367': ['sum', 'mean', 'count'],  # Transaction amount stats
        'f368': ['sum', 'mean'],  # Other transaction features
        'f369': ['sum', 'mean'],
        'f370': ['sum', 'mean']
    }).round(4)
    trans_features.columns = ['total_amount', 'avg_amount', 'trans_count', 
                             'f368_sum', 'f368_mean', 'f369_sum', 'f369_mean', 
                             'f370_sum', 'f370_mean']
    trans_features = trans_features.reset_index()
    
    # Merge engineered features
    combined_data = combined_data.merge(event_features, on='id2', how='left')
    combined_data = combined_data.merge(trans_features, on='id2', how='left')
    
    # Create interaction features
    print("Creating interaction features...")
    # CTR-related interactions
    combined_data['ctr_1_to_30_ratio'] = combined_data['f310'] / (combined_data['f314'] + 1e-6)
    combined_data['clicks_to_impressions_ratio'] = combined_data['f319'] / (combined_data['f324'] + 1e-6)
    
    # Customer value interactions
    combined_data['spend_per_transaction'] = (combined_data['f39'] + combined_data['f40'] + combined_data['f41']) / (combined_data['f185'] + 1)
    combined_data['miles_per_transaction'] = combined_data['f43'] / (combined_data['f185'] + 1)
    
    # Time-based features
    combined_data['time_of_day_bin'] = pd.cut(combined_data['f168'], bins=4, labels=['morning', 'afternoon', 'evening', 'night'])
    combined_data['is_weekend'] = combined_data['f349'].isin([6, 7]).astype(int)
    
    # Fill missing values
    print("Handling missing values...")
    # Fill numerical features with median
    numerical_features = combined_data.select_dtypes(include=[np.number]).columns
    for col in numerical_features:
        if combined_data[col].isnull().sum() > 0:
            combined_data[col].fillna(combined_data[col].median(), inplace=True)
    
    # Fill categorical features with mode
    categorical_features = combined_data.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if combined_data[col].isnull().sum() > 0:
            combined_data[col].fillna(combined_data[col].mode()[0] if len(combined_data[col].mode()) > 0 else 'unknown', inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        if col not in ['id1', 'id2', 'id3']:  # Skip ID columns
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col].astype(str))
            label_encoders[col] = le
    
    # Split back to train and test
    train_processed = combined_data[combined_data['is_train'] == 1].copy()
    test_processed = combined_data[combined_data['is_train'] == 0].copy()
    
    train_processed = train_processed.drop(['is_train'], axis=1)
    test_processed = test_processed.drop(['is_train', 'y'], axis=1)
    
    print(f"Processed train shape: {train_processed.shape}")
    print(f"Processed test shape: {test_processed.shape}")
    
    return train_processed, test_processed, label_encoders

def build_model(X_train, y_train, X_val, y_val):
    """Build and train LightGBM model"""
    print("\n=== MODEL TRAINING ===")
    
    # LightGBM parameters optimized for imbalanced dataset
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 100,
        'min_sum_hessian_in_leaf': 1,
        'is_unbalance': True,  # Handle class imbalance
        'random_state': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    print("\n=== MODEL EVALUATION ===")
    
    # Predictions
    val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = (val_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc_score = roc_auc_score(y_val, val_pred_proba)
    print(f"Validation AUC: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, val_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20))
    
    return auc_score, feature_importance

def main():
    """Main execution function"""
    print("AMEX Campus Challenge Round 2: Offer Click Prediction")
    print("=" * 60)
    
    # Load data
    train_data, test_data, add_event, add_trans, offer_metadata = load_data()
    
    # Basic EDA
    target_dist, missing_features = basic_eda(train_data)
    
    # Feature engineering
    train_processed, test_processed, label_encoders = feature_engineering(
        train_data, test_data, add_event, add_trans, offer_metadata
    )
    
    # Prepare features and target
    feature_cols = [col for col in train_processed.columns if col not in ['id1', 'id2', 'id3', 'id4', 'id5', 'y']]
    X = train_processed[feature_cols]
    y = train_processed['y']
    
    print(f"\nUsing {len(feature_cols)} features for modeling")
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Build model
    model = build_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    auc_score, feature_importance = evaluate_model(model, X_val, y_val)
    
    # Generate predictions for test set
    print("\n=== GENERATING PREDICTIONS ===")
    test_features = test_processed[feature_cols]
    test_predictions = model.predict(test_features, num_iteration=model.best_iteration)
    
    # Create submission file
    submission = pd.read_csv('685404e30cfdb_submission_template.csv')
    submission['pred'] = test_predictions
    
    submission_filename = 'submission_round2.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"Submission saved to: {submission_filename}")
    
    # Save feature importance
    feature_importance.to_csv('feature_importance_round2.csv', index=False)
    print("Feature importance saved to: feature_importance_round2.csv")
    
    print(f"\nFinal Validation AUC: {auc_score:.4f}")
    print(f"Average prediction: {test_predictions.mean():.4f}")
    print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")

if __name__ == "__main__":
    main()
