"""
Fix Submission File - Ensure Perfect Format Compliance
=====================================================
"""

import pandas as pd
import numpy as np

def fix_submission_format():
    """Create a perfectly formatted submission file"""
    print("Fixing submission format for platform compatibility...")
    
    # Load template and our predictions
    template = pd.read_csv('685404e30cfdb_submission_template.csv')
    our_submission = pd.read_csv('r2_submission_fileTVIJAYABALAJI.csv')
    
    print(f"Template shape: {template.shape}")
    print(f"Our submission shape: {our_submission.shape}")
    
    # Create a clean copy from template
    fixed_submission = template.copy()
    
    # Ensure exact same order and format
    fixed_submission['pred'] = our_submission['pred'].values
    
    # Ensure data types match exactly
    fixed_submission['id1'] = fixed_submission['id1'].astype(str)
    fixed_submission['id2'] = fixed_submission['id2'].astype(str) 
    fixed_submission['id3'] = fixed_submission['id3'].astype(str)
    fixed_submission['id5'] = fixed_submission['id5'].astype(str)
    fixed_submission['pred'] = fixed_submission['pred'].astype(float)
    
    # Remove any potential whitespace or formatting issues
    for col in ['id1', 'id2', 'id3', 'id5']:
        if fixed_submission[col].dtype == 'object':
            fixed_submission[col] = fixed_submission[col].astype(str).str.strip()
    
    # Ensure predictions are in valid range
    fixed_submission['pred'] = np.clip(fixed_submission['pred'], 0.0, 1.0)
    
    # Round predictions to reasonable precision
    fixed_submission['pred'] = fixed_submission['pred'].round(8)
    
    # Save with exact same format as template
    team_name = "TVIJAYABALAJI"
    output_file = f'r2_submission_file{team_name}_fixed.csv'
    
    # Save without index and with same format
    fixed_submission.to_csv(output_file, index=False, float_format='%.8f')
    
    print(f"Fixed submission saved as: {output_file}")
    
    # Validation
    print("\nValidation:")
    print(f"Shape: {fixed_submission.shape}")
    print(f"Columns: {list(fixed_submission.columns)}")
    print(f"Data types: {fixed_submission.dtypes.to_dict()}")
    print(f"Prediction stats: mean={fixed_submission['pred'].mean():.6f}, range=[{fixed_submission['pred'].min():.6f}, {fixed_submission['pred'].max():.6f}]")
    print(f"Any missing values: {fixed_submission.isnull().sum().sum()}")
    
    # Check first few rows
    print(f"\nFirst 3 rows:")
    print(fixed_submission.head(3))
    
    return output_file

def create_alternative_submission():
    """Create alternative submission using the exact template structure"""
    print("\nCreating alternative submission with exact template preservation...")
    
    # Re-run the model to get fresh predictions
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    
    # Load data
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    # Quick preprocessing
    train_data['y'] = pd.to_numeric(train_data['y']).astype(int)
    
    feature_cols = [col for col in train_data.columns if col.startswith('f')]
    
    # Convert features
    for col in feature_cols:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    
    # Fill missing
    for col in feature_cols:
        median_val = train_data[col].median()
        train_data[col] = train_data[col].fillna(median_val)
        test_data[col] = test_data[col].fillna(median_val)
    
    # Prepare data
    X = train_data[feature_cols]
    y = train_data['y']
    X_test = test_data[feature_cols]
    
    # Train model
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 64,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'is_unbalance': True,
        'verbosity': -1,
        'random_state': 42
    }
    
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    model = lgb.train(params, train_set, valid_sets=[val_set], 
                      num_boost_round=1000, callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    # Get predictions
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Load template and preserve exact structure
    template = pd.read_csv('685404e30cfdb_submission_template.csv')
    
    # Create submission preserving exact template order
    submission = template.copy()
    submission['pred'] = test_pred
    
    # Save alternative
    team_name = "TVIJAYABALAJI" 
    alt_file = f'r2_submission_file{team_name}_alt.csv'
    submission.to_csv(alt_file, index=False)
    
    print(f"Alternative submission saved as: {alt_file}")
    return alt_file

def main():
    print("AMEX Round 2 - Submission Format Fix")
    print("=" * 50)
    
    # Try to fix the original submission
    try:
        fixed_file = fix_submission_format()
        print(f"✅ Fixed submission created: {fixed_file}")
    except Exception as e:
        print(f"❌ Error fixing submission: {e}")
    
    # Create alternative submission
    try:
        alt_file = create_alternative_submission()
        print(f"✅ Alternative submission created: {alt_file}")
    except Exception as e:
        print(f"❌ Error creating alternative: {e}")
    
    print("\n" + "="*50)
    print("SUBMISSION FILES READY:")
    print("1. r2_submission_fileTVIJAYABALAJI_fixed.csv")
    print("2. r2_submission_fileTVIJAYABALAJI_alt.csv")
    print("Try submitting both to see which format works!")

if __name__ == "__main__":
    main()
