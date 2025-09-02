# Methodology Documentation

## AMEX Campus Challenge Round 2 - Detailed Methodology

### Problem Analysis

#### Initial Challenge Assessment
- **Competition**: AMEX Campus Challenge Round 2
- **Task**: Binary classification to predict customer offer click behavior
- **Timeline**: Performance improvement from 0.154 to 0.8474 AUC
- **Key Insight**: Employee psychology approach to understanding customer behavior

#### Data Characteristics
```
Training Dataset: 770,164 samples × 372 features
Test Dataset: 369,301 samples × 371 features
Target Distribution: 95.2% no-click, 4.8% click (severe imbalance)
Feature Types: Mix of numerical, categorical, and behavioral features
```

### Technical Discoveries

#### 1. Binary String Target Issue
**Problem**: Target column contained binary strings ('True'/'False') instead of numerical values (0/1)
```python
# Original problematic code
y = train_data['target']  # Results in string values

# Fixed solution
y = train_data['target'].astype(str).map({'True': 1, 'False': 0})
```

**Impact**: This fix alone improved baseline performance significantly and enabled proper model training.

#### 2. Overfitting Detection
**Problem**: Models achieved extremely high validation AUC (0.95+) but poor competition performance
**Root Cause**: Data leakage and overfitting in cross-validation setup

**Solutions Implemented**:
- Reduced model complexity drastically
- Implemented extreme early stopping (500-800 patience)
- Used conservative hyperparameters
- Applied strong regularization (L1=5.0, L2=5.0)

### Employee Psychology Framework

#### Core Behavioral Insights
Our breakthrough came from modeling customers as **working employees** rather than abstract data points:

##### 1. Temporal Behavior Patterns
```python
# Evening engagement (after work hours)
evening_activity = (hour >= 18) | (hour <= 6)

# Weekend browsing behavior  
weekend_pattern = (day_of_week >= 5)

# End-of-month spending cycles
month_end_behavior = (day_of_month >= 25) | (day_of_month <= 5)
```

##### 2. Engagement Level Categorization
```python
# Highly engaged employees (top 25%)
highly_engaged = (feature_366 > np.percentile(feature_366, 75))

# Conservative users (below median activity)
conservative_behavior = (feature_350 <= median_value)

# Risk-averse patterns
low_risk_profile = (transaction_frequency < threshold)
```

##### 3. Spending Psychology
- **Salary Cycle Impact**: Higher activity at beginning/end of month
- **Work-Life Balance**: More offer engagement during personal time
- **Financial Caution**: Conservative clicking patterns typical of employed individuals
- **Mobile Usage**: Evening and weekend mobile browsing increase

### Model Architecture Evolution

#### Version 1: Baseline Model
```python
params_baseline = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}
```
**Result**: 0.9469 validation AUC, poor generalization

#### Version 2: Optimized Model
```python
params_optimized = {
    'num_leaves': 15,
    'learning_rate': 0.05,
    'min_data_in_leaf': 500,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0
}
```
**Result**: 0.9550 validation AUC, still overfitting

#### Version 3: Psychology Model (Breakthrough)
```python
params_psychology = {
    'num_leaves': 8,
    'learning_rate': 0.01,
    'min_data_in_leaf': 1000,
    'lambda_l1': 3.0,
    'lambda_l2': 3.0,
    'max_depth': 4
}
```
**Result**: 0.8581 validation AUC, balanced performance

#### Version 4: Ultra Conservative (Final)
```python
params_ultra_conservative = {
    'num_leaves': 6,           # Extremely small trees
    'learning_rate': 0.003,    # Very slow learning
    'min_data_in_leaf': 2000,  # Large leaf requirement
    'lambda_l1': 5.0,          # Strong L1 regularization
    'lambda_l2': 5.0,          # Strong L2 regularization
    'max_depth': 3,            # Very shallow trees
    'feature_fraction': 0.6,   # Use only 60% features
    'bagging_fraction': 0.6,   # Use only 60% data
    'early_stopping_round': 800 # Very patient training
}
```
**Result**: 0.8474 validation AUC, excellent 5-fold CV consistency (±0.0010)

### Cross-Validation Strategy

#### 5-Fold Stratified Validation
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train model
    model = lgb.train(params, train_set, valid_sets=[val_set])
    
    # Evaluate
    pred = model.predict(X_val_fold)
    auc = roc_auc_score(y_val_fold, pred)
    cv_scores.append(auc)
```

#### Results Analysis
```
Fold 1: 0.8473 AUC
Fold 2: 0.8476 AUC  
Fold 3: 0.8467 AUC
Fold 4: 0.8489 AUC
Fold 5: 0.8494 AUC

Mean: 0.8480 AUC
Std:  0.0010 AUC (Excellent consistency!)
```

### Feature Engineering Philosophy

#### Employee-Centric Features
Rather than creating complex mathematical transformations, we focused on features that reflect real employee behavior:

```python
def create_psychology_features(data):
    # Work schedule alignment
    data['after_work_hours'] = (data['hour'] >= 18) | (data['hour'] <= 6)
    
    # Engagement level (based on activity)
    data['high_engagement'] = (data['f366'] > data['f366'].quantile(0.75))
    
    # Conservative behavior pattern
    data['conservative_user'] = (data['f350'] <= data['f350'].median())
    
    # Weekend behavior
    data['weekend_user'] = (data['day_of_week'] >= 5)
    
    # Monthly cycle patterns
    data['month_end_period'] = (data['day'] >= 25) | (data['day'] <= 5)
    
    return data
```

### Lessons Learned

#### 1. Validation vs. Reality Gap
**Lesson**: High validation scores can be misleading in competitions
**Solution**: Focus on cross-validation consistency rather than peak performance

#### 2. Domain Knowledge Impact
**Lesson**: Understanding the customer psychology is more valuable than complex feature engineering
**Solution**: Think like the target audience (working professionals)

#### 3. Overfitting Prevention
**Lesson**: Conservative approaches often generalize better
**Solution**: Extreme regularization and simple models for complex datasets

#### 4. Target Data Quality
**Lesson**: Always validate data types and formats
**Solution**: Implement data validation as first step in any pipeline

### Hyperparameter Tuning Philosophy

#### Conservative Grid Search
```python
param_grid = {
    'num_leaves': [6, 8, 10],        # Very small trees
    'learning_rate': [0.001, 0.003, 0.005],  # Slow learning
    'min_data_in_leaf': [1000, 2000, 3000],  # Large leaves
    'lambda_l1': [3.0, 5.0, 7.0],   # Strong regularization
    'lambda_l2': [3.0, 5.0, 7.0],   # Strong regularization
}
```

#### Selection Criteria
1. **Cross-validation consistency** (low standard deviation)
2. **Reasonable validation scores** (avoid too-good-to-be-true results)
3. **Training stability** (smooth learning curves)
4. **Feature importance interpretability**

### Model Interpretability

#### Feature Importance Analysis
```python
# Top features by importance
feature_importance = model.feature_importance(importance_type='gain')
top_features = sorted(zip(feature_names, feature_importance), 
                     key=lambda x: x[1], reverse=True)[:20]
```

#### Psychology Feature Performance
- **High Engagement Features**: Strong predictive power for click behavior
- **Temporal Features**: Evening/weekend activity crucial
- **Conservative Behavior**: Negative correlation with clicking
- **Activity Level**: Moderate activity users show highest click rates

### Competition Strategy

#### Model Selection Logic
1. **Primary Model**: Ultra Conservative (best generalization)
2. **Secondary Model**: Psychology Model (balanced performance)  
3. **Baseline Model**: Quick results for validation

#### Submission Strategy
- Focus on models with consistent cross-validation
- Avoid models with suspiciously high validation scores
- Prioritize interpretable feature importance
- Test multiple approaches but submit most conservative

### Future Improvements

#### Potential Enhancements
1. **Ensemble Methods**: Combine multiple conservative models
2. **Advanced Psychology Features**: Time series patterns, sequence modeling
3. **External Data**: Economic indicators, seasonal patterns
4. **Deep Learning**: Neural networks with regularization
5. **Fairness Considerations**: Bias detection and mitigation

#### Recommended Next Steps
1. Implement ensemble of top 3 conservative models
2. Add temporal sequence features (last N interactions)
3. Create user clustering based on psychology profiles
4. Implement automated hyperparameter search with CV consistency constraints
5. Add model explainability tools (SHAP, LIME)

---

*This methodology documentation captures the complete journey from initial poor performance (0.154 AUC) to final strong results (0.8474 AUC) through employee psychology modeling and conservative machine learning practices.*
