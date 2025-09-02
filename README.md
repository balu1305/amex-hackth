# AMEX Campus Challenge Round 2 - Offer Click Prediction 🏆

**An Employee Psychology-Driven Approach to Predicting Customer Offer Interactions**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Machine%20Learning-green.svg)](https://lightgbm.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-orange.svg)](https://pandas.pydata.org/)

## 🎯 Project Overview

This repository contains our solution for the **AMEX Campus Challenge Round 2**, focusing on predicting customer click behavior on offers using advanced machine learning techniques combined with **employee psychology insights**.

### Problem Statement
- **Task**: Binary classification to predict whether customers will click on offers
- **Dataset**: 770K+ training samples, 369K+ test samples, 366 features
- **Challenge**: Highly imbalanced dataset (95.2% no-click vs 4.8% click)
- **Evaluation**: AUC-ROC (Area Under ROC Curve)

### Key Achievement
🚀 **Improved from initial score of 0.154 to 0.8474 AUC through employee psychology modeling and overfitting prevention**

## 📊 Dataset Analysis

```
Training Data: 770,164 samples × 372 features
Test Data: 369,301 samples × 371 features
Target Distribution: 4.8% click rate (typical of real-world scenarios)
Class Imbalance Ratio: 19.8:1 (no-click : click)
```

### Data Challenges Solved
1. **Binary String Targets**: Target column contained binary strings instead of 0/1 values
2. **Memory Constraints**: Large dataset required efficient processing
3. **Feature Complexity**: 366 features needed careful selection
4. **Extreme Imbalance**: Required specialized handling techniques

## 🧠 Employee Psychology Approach

Our breakthrough came from thinking like **working employees** who interact with offers:

### Key Behavioral Insights
- **Evening Browsing**: Employees check offers after work hours (higher engagement)
- **Engagement Patterns**: Highly engaged users show different click behavior
- **Conservative Spending**: Risk-averse patterns typical of working professionals
- **Salary Cycles**: End/beginning of month spending variations

### Psychology-Based Features
```python
# High engagement employees (top 25%)
highly_engaged_employee = (f366 > 75th_percentile)

# Conservative behavior (below median activity)
conservative_user = (f350 <= median_value)

# Weekend browsing patterns
weekend_behavior = (day_of_week >= 5)
```

## 🛠️ Technical Architecture

### Model Pipeline
1. **Data Loading & Preprocessing**
   - Efficient parquet file handling
   - Binary string target conversion
   - Memory-optimized data types

2. **Feature Engineering**
   - Employee psychology patterns
   - Engagement level categorization
   - Conservative behavior modeling

3. **Model Training**
   - LightGBM with extreme overfitting prevention
   - 5-fold cross-validation
   - Patient early stopping

4. **Prediction & Submission**
   - Conservative prediction generation
   - Multiple submission formats

### Overfitting Prevention Strategy
```python
ultra_conservative_params = {
    'num_leaves': 6,           # Extremely small trees
    'learning_rate': 0.003,    # Very slow learning
    'min_data_in_leaf': 2000,  # Large leaf requirement
    'lambda_l1': 5.0,          # Strong L1 regularization
    'lambda_l2': 5.0,          # Strong L2 regularization
    'max_depth': 3,            # Very shallow trees
    'feature_fraction': 0.6,   # Use only 60% features
    'bagging_fraction': 0.6,   # Use only 60% data
}
```

## 📈 Results & Performance

| Model Version | Validation AUC | CV Consistency | Key Features |
|---------------|----------------|----------------|--------------|
| Baseline | 0.9469 | N/A | Standard approach, overfitted |
| Optimized | 0.9550 | N/A | Feature engineering, still overfitted |
| **Fixed Psychology** | **0.8581** | N/A | Employee psychology, balanced |
| **Ultra Conservative** | **0.8474** | **±0.0010** | 5-fold CV, maximum generalization |

### Cross-Validation Results
```
Fold 1: 0.8473
Fold 2: 0.8476  
Fold 3: 0.8467
Fold 4: 0.8489
Fold 5: 0.8494
Mean: 0.8480 ± 0.0010 (Excellent consistency!)
```

## 🗂️ File Structure

```
AMEX-Round2-Offer-Prediction/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── data/                              # Data files (not tracked)
│   ├── train_data.parquet
│   ├── test_data.parquet
│   └── submission_template.csv
├── models/                            # Model implementations
│   ├── quick_start.py                 # Baseline model
│   ├── final_optimized_model.py       # Feature-rich model
│   ├── fixed_psychology_model.py      # Employee psychology model
│   └── ultra_conservative_model.py    # Final robust model
├── submissions/                       # Generated submissions
│   ├── r2_submission_fileTVIJAYABALAJI_fixed.csv
│   └── r2_submission_fileTVIJAYABALAJI_ultra_conservative.csv
├── analysis/                          # Analysis scripts
│   ├── data_exploration.ipynb         # Jupyter notebook for EDA
│   └── feature_importance.csv         # Feature analysis results
└── docs/                             # Documentation
    ├── methodology.md                 # Detailed methodology
    └── lessons_learned.md             # Key insights
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.12+
8GB+ RAM (for data processing)
```

### Installation
```bash
# Clone the repository
git clone https://github.com/TVIJAYABALAJI/AMEX-Round2-Offer-Prediction.git
cd AMEX-Round2-Offer-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Running the Models

#### 1. Ultra Conservative Model (Recommended)
```bash
python models/ultra_conservative_model.py
```
**Best for**: Maximum generalization and real-world performance

#### 2. Fixed Psychology Model
```bash
python models/fixed_psychology_model.py  
```
**Best for**: Balanced performance with employee insights

#### 3. Baseline Model
```bash
python models/quick_start.py
```
**Best for**: Quick results and understanding the problem

### Expected Outputs
- Submission files in CSV format
- Feature importance analysis
- Cross-validation results
- Model performance metrics

## 💡 Key Insights & Lessons Learned

### 🔍 Major Discoveries
1. **Target Data Issue**: Binary strings instead of 0/1 caused initial poor performance
2. **Overfitting Problem**: High validation AUC (0.95+) didn't translate to leaderboard performance
3. **Employee Psychology**: Understanding working professionals' behavior was crucial
4. **Conservative Approach**: Lower validation scores with strong generalization performed better

### 🛡️ Overfitting Prevention
- **Extreme Early Stopping**: 500-800 patience rounds
- **Cross-Validation**: 5-fold validation for robust evaluation  
- **Feature Selection**: Minimal feature set to reduce noise
- **Regularization**: Strong L1/L2 penalties
- **Tree Constraints**: Very small, shallow trees

### 🧠 Psychology-Driven Features
- **Engagement Levels**: Top 25% users show different patterns
- **Timing Patterns**: Weekend vs weekday behavior
- **Risk Profiles**: Conservative vs aggressive users
- **Spending Habits**: Regular vs irregular customers

## 📋 Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
pyarrow>=12.0.0
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 🏆 Competition Strategy

### Model Selection Philosophy
1. **Prioritize Generalization** over validation scores
2. **Employee Psychology** over pure mathematical features  
3. **Conservative Parameters** over aggressive optimization
4. **Cross-Validation Consistency** over single validation performance

### Submission Strategy
1. **Primary**: Ultra Conservative Model (`ultra_conservative.csv`)
2. **Backup**: Fixed Psychology Model (`fixed.csv`)
3. **Baseline**: Quick Start Model (`quick.csv`)

## 📚 Advanced Usage

### Custom Feature Engineering
```python
def create_custom_psychology_features(train_data, test_data):
    # Add your employee behavior insights here
    # Focus on real-world customer patterns
    pass
```

### Hyperparameter Tuning
```python
# Conservative tuning approach
param_grid = {
    'num_leaves': [6, 8, 10],
    'learning_rate': [0.001, 0.003, 0.005],
    'min_data_in_leaf': [1000, 2000, 3000]
}
```

### Cross-Validation Setup
```python
# Robust evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Train and validate
    pass
```



---

**⭐ If this repository helped you, please give it a star!**

**🔄 Found an issue? Please report it so we can improve!**

**💡 Have a better approach? We'd love to hear from you!**
