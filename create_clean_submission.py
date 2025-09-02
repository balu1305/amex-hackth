"""
Create Ultra-Clean Submission
============================
Minimal, clean submission that should work
"""

import pandas as pd
import numpy as np

def create_clean_submission():
    """Create the cleanest possible submission"""
    print("Creating ultra-clean submission...")
    
    # Load the exact template
    template = pd.read_csv('685404e30cfdb_submission_template.csv')
    print(f"Template loaded: {template.shape}")
    
    # Load our best predictions
    best_sub = pd.read_csv('r2_submission_fileTVIJAYABALAJI.csv')
    print(f"Best submission loaded: {best_sub.shape}")
    
    # Start fresh with template
    clean_submission = pd.DataFrame()
    
    # Copy ID columns exactly as they are in template
    clean_submission['id1'] = template['id1'].copy()
    clean_submission['id2'] = template['id2'].copy() 
    clean_submission['id3'] = template['id3'].copy()
    clean_submission['id5'] = template['id5'].copy()
    
    # Add predictions, ensuring they're clean floats
    clean_submission['pred'] = best_sub['pred'].copy()
    
    # Clean the predictions
    clean_submission['pred'] = pd.to_numeric(clean_submission['pred'], errors='coerce')
    clean_submission['pred'] = clean_submission['pred'].fillna(0.048)  # Fill any NaN with base rate
    clean_submission['pred'] = np.clip(clean_submission['pred'], 0.0001, 0.9999)  # Ensure valid range
    
    # Round to 6 decimal places for cleanliness
    clean_submission['pred'] = clean_submission['pred'].round(6)
    
    # Final validation
    print("Final validation:")
    print(f"Shape: {clean_submission.shape}")
    print(f"Columns: {list(clean_submission.columns)}")
    print(f"Any NaN: {clean_submission.isnull().sum().sum()}")
    print(f"Pred range: [{clean_submission['pred'].min():.6f}, {clean_submission['pred'].max():.6f}]")
    print(f"Pred mean: {clean_submission['pred'].mean():.6f}")
    
    # Save with simple name
    output_file = 'r2_submission_fileTVIJAYABALAJI_clean.csv'
    clean_submission.to_csv(output_file, index=False)
    
    print(f"✅ Clean submission saved: {output_file}")
    
    # Show sample
    print("\nSample rows:")
    print(clean_submission.head(3))
    
    return output_file

if __name__ == "__main__":
    create_clean_submission()
