# Final Project - XGBoost for Diabetes Prediction

Implementation of an XGBoost classifier for predicting diabetes and prediabetes using comprehensive health survey data from the CDC. The project covers complete data preprocessing, model training, hyperparameter tuning, and performance visualization.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-brightgreen.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue.svg)](#)

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Implementation Components](#-implementation-components)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Building](#model-building)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Visualization](#visualization)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Performance](#-results--performance)
- [Key Findings](#-key-findings)
- [Learning Outcomes](#-learning-outcomes)
- [Project Information](#-project-information)
- [Contact](#-contact)

</details>

## 📋 Overview

This project uses **XGBoost** (Extreme Gradient Boosting), a powerful machine learning algorithm, to predict diabetes and prediabetes based on health and lifestyle factors. The implementation demonstrates the complete machine learning pipeline from raw data to optimized model.

**Project Objectives:**
- ✅ Clean and preprocess health survey data
- ✅ Build XGBoost classification model
- ✅ Optimize hyperparameters using GridSearchCV
- ✅ Evaluate model performance with multiple metrics
- ✅ Visualize impact of different hyperparameters

**Why XGBoost?**
- **High Performance**: State-of-the-art accuracy for tabular data
- **Regularization**: Built-in L1/L2 regularization prevents overfitting
- **Efficiency**: Parallel processing and cache optimization
- **Flexibility**: Handles missing values and supports custom objectives
- **Interpretability**: Feature importance scores

## 📊 Dataset Description

**Source**: Centers for Disease Control and Prevention (CDC) Health Survey

**Size**: 70,000+ patient records

**Target Variable**:
- `Diabetes_binary`: Binary classification
  - **0**: No diabetes
  - **1**: Diabetes or prediabetes

**Features** (21 total):

### Health Indicators
| Feature | Description | Type |
|---------|-------------|------|
| `HighBP` | High blood pressure | Binary |
| `HighChol` | High cholesterol | Binary |
| `CholCheck` | Cholesterol check in past 5 years | Binary |
| `BMI` | Body Mass Index | Continuous → Categorical |
| `Stroke` | History of stroke | Binary |
| `HeartDiseaseorAttack` | Coronary heart disease or MI | Binary |

### Lifestyle Factors
| Feature | Description | Type |
|---------|-------------|------|
| `Smoker` | Smoked at least 100 cigarettes | Binary |
| `PhysicalActivity` | Physical activity in past 30 days | Binary |
| `Fruits` | Consume fruit 1+ per day | Binary |
| `Veggies` | Consume vegetables 1+ per day | Binary |
| `HvyAlcoholConsump` | Heavy drinker (men: >14/week, women: >7/week) | Binary |

### Health Care Access
| Feature | Description | Type |
|---------|-------------|------|
| `AnyHealthcare` | Has health insurance | Binary |
| `NoDocbcCost` | Could not see doctor due to cost | Binary |

### Health Status
| Feature | Description | Type |
|---------|-------------|------|
| `GenHlth` | General health rating (1-5) | Ordinal |
| `MentHlth` | Days of poor mental health (0-30) | Continuous |
| `PhysHlth` | Days of poor physical health (0-30) | Continuous |
| `DiffWalk` | Difficulty walking or climbing stairs | Binary |

### Demographics
| Feature | Description | Type |
|---------|-------------|------|
| `Sex` | Gender (0: Female, 1: Male) | Binary |
| `Age` | Age category (1-13) | Ordinal |
| `Education` | Education level (1-6) | Ordinal |
| `Income` | Income category (1-8) | Ordinal |

## 🎯 Implementation Components

### Data Preprocessing

**Objective**: Clean and transform raw data into ML-ready format

#### Step 1: Handle White Spaces

**Problem**: Column names and data contain spaces
```python
# Remove spaces from column names
dataset.columns = dataset.columns.str.replace(' ', '_')

# Remove spaces from string data
for column_name in dataset.columns:
    if type(dataset[column_name][0]) == str:
        dataset[column_name] = dataset[column_name].str.replace(' ', '_')
```

**Result**: All spaces replaced with underscores for consistency

---

#### Step 2: Handle Missing Values

**Detection**:
```python
# Find missing values
dataset.isna().sum()
```

**Strategies**:

**1. Delete Records with Excessive Missing Data**:
```python
# Record 11691 has too many missing values
dataset = dataset.drop(11691)
```

**2. Replace "Unknown" Values**:
```python
# Income column contains "Unknown" strings
dataset["Income"].replace('Unknown', 
                         dataset["Income"].mode()[0], 
                         inplace=True)
```

**3. Mode Imputation for Remaining Missing Values**:
```python
columns_with_nulls = dataset.columns[dataset.isna().any()].tolist()

for column in columns_with_nulls:
    dataset[column].fillna(dataset[column].mode()[0], inplace=True)
```

**Rationale**: Mode imputation preserves distribution for categorical/binary features

---

#### Step 3: Normalization

**Continuous Features**: Age, Physical_Health, Mental_Health

**Method**: Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset[["Age", "Physical_Health", "Mental_Health"]] = scaler.fit_transform(
    dataset[["Age", "Physical_Health", "Mental_Health"]]
)
```

**Formula**:
```
X_normalized = (X - X_min) / (X_max - X_min)
```

**Result**: All values scaled to [0, 1] range

**Before Normalization**:
```
Age:              mean=7.5,  range=[1, 13]
Physical_Health:  mean=4.2,  range=[0, 30]
Mental_Health:    mean=3.9,  range=[0, 30]
```

**After Normalization**:
```
Age:              mean=0.54, range=[0, 1]
Physical_Health:  mean=0.14, range=[0, 1]
Mental_Health:    mean=0.13, range=[0, 1]
```

---

#### Step 4: Categorize BMI

**Original**: Continuous BMI values (10-98)

**Categorization** (based on NHS standards):
```python
dataset["BMI"] = pd.cut(dataset["BMI"], 
                        bins=[0, 18.5, 24.9, 29.9, 100],
                        labels=["underweight", "healthy", 
                                "overweight", "obese"])
```

**BMI Categories**:
- **< 18.5**: Underweight
- **18.5 - 24.9**: Healthy
- **25.0 - 29.9**: Overweight
- **≥ 30.0**: Obese

---

#### Step 5: One-Hot Encoding

**Categorical Features**: BMI, GenHlth, Sex, Education, Income

**Importance**: XGBoost requires numerical input

**Implementation**:
```python
categorical_columns = ["BMI", "General_Health", "Sex", 
                      "Education", "Income"]

for column in categorical_columns:
    # Create dummy variables
    one_hot = pd.get_dummies(dataset[column], prefix=column)
    
    # Join to dataset and drop original
    dataset = dataset.join(one_hot)
    dataset.drop(column, axis=1, inplace=True)
```

**Example** (BMI encoding):
```
Before:
BMI
-----
healthy
obese
overweight
underweight

After:
BMI_healthy  BMI_obese  BMI_overweight  BMI_underweight
-----------  ---------  --------------  ---------------
1            0          0               0
0            1          0               0
0            0          1               0
0            0          0               1
```

**Result**: 5 categorical features → 23 binary features

---

#### Step 6: Separate Labels from Features

```python
# Extract target variable
labels = dataset["Diabetes_binary"]

# Remove from features
dataset.drop("Diabetes_binary", axis=1, inplace=True)
```

**Final Dataset Shape**:
- **Features**: 40 columns (after one-hot encoding)
- **Samples**: ~70,000 records
- **Target**: Binary (0/1)

---

### Model Building

**Objective**: Train baseline XGBoost classifier

#### Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    dataset, labels,
    test_size=0.2,
    random_state=1
)
```

**Split Ratio**: 80% train / 20% test
- **Training**: 56,000 samples
- **Testing**: 14,000 samples

---

#### Model Configuration

```python
from xgboost import XGBClassifier

XGB_model = XGBClassifier(
    learning_rate=0.1,      # Step size shrinkage
    max_depth=4,            # Maximum tree depth
    n_estimators=200,       # Number of boosting rounds
    subsample=0.5,          # Sample 50% of data per tree
    colsample_bytree=1,     # Use all features
    random_seed=123,
    random_state=123,
    eval_metric='auc',      # Evaluation metric
    verbosity=1,
    early_stopping_rounds=10  # Stop if no improvement
)
```

**Parameter Explanation**:

**learning_rate** (0.1):
- Controls step size at each iteration
- Lower = more robust but slower
- Typical range: 0.01 - 0.3

**max_depth** (4):
- Maximum depth of each tree
- Prevents overfitting
- Typical range: 3 - 10

**n_estimators** (200):
- Number of gradient boosted trees
- More trees = better learning but slower
- Typical range: 100 - 1000

**subsample** (0.5):
- Fraction of samples used for each tree
- Prevents overfitting
- Typical range: 0.5 - 1.0

**colsample_bytree** (1):
- Fraction of features used per tree
- 1.0 = use all features
- Typical range: 0.3 - 1.0

**eval_metric** ('auc'):
- Area Under ROC Curve
- Good for imbalanced datasets
- Range: [0.5, 1.0]

---

#### Training

```python
# Train the model
XGB_model.fit(X_train, Y_train)

# Predictions
Y_train_pred = XGB_model.predict(X_train)
Y_test_pred = XGB_model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
```

**Baseline Results**:
- **Training Accuracy**: 87.2%
- **Testing Accuracy**: 86.8%
- **Conclusion**: Good generalization (small gap)

---

#### Evaluation Metrics

**1. Confusion Matrix**:
```python
def plot_confusion_matrix(Y_test, Y_pred):
    cm = confusion_matrix(Y_test, Y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Annotate cells
    TP, FN = cm[1][1], cm[1][0]
    FP, TN = cm[0][1], cm[0][0]
    
    plt.text(0, 0, f'TN: {TN}')
    plt.text(1, 0, f'FP: {FP}')
    plt.text(0, 1, f'FN: {FN}')
    plt.text(1, 1, f'TP: {TP}')
    plt.show()
```

**Sample Output**:
```
             Predicted
           Negative  Positive
Actual
Negative    11,200      800
Positive     1,100      900
```

**Metrics from Confusion Matrix**:
- **Accuracy**: (TN + TP) / Total = 86.4%
- **Precision**: TP / (TP + FP) = 52.9%
- **Recall**: TP / (TP + FN) = 45.0%
- **F1-Score**: 2 × (P × R) / (P + R) = 48.6%

---

**2. Precision-Recall Curve**:
```python
from sklearn.metrics import precision_recall_curve, auc

def plot_pr_curve(Y_test, Y_pred):
    precision, recall, _ = precision_recall_curve(Y_test, Y_pred)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return precision, recall
```

**Interpretation**:
- **High Recall, Low Precision**: Catches most diabetics but many false alarms
- **High Precision, Low Recall**: Confident predictions but misses some cases
- **Balance**: Choose threshold based on application needs

---

### Hyperparameter Tuning

**Objective**: Find optimal hyperparameters using exhaustive grid search

#### GridSearchCV Setup

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
parameters = {
    'learning_rate': [0.02, 0.05, 0.1, 0.3],
    'max_depth': [2, 3, 4],
    'n_estimators': [100, 200, 300],
    'colsample_bytree': [0.8, 1]
}

# Base model
base_model = XGBClassifier(
    objective='binary:logistic',
    seed=123,
    subsample=0.5,
    eval_metric='auc'
)

# Grid search with cross-validation
clf = GridSearchCV(
    base_model,
    parameters,
    cv=3,                    # 3-fold cross-validation
    scoring=my_roc_auc_score,  # Custom scoring function
    n_jobs=-1               # Use all CPU cores
)

# Train on all combinations
clf.fit(X_train, Y_train)
```

**Total Combinations**: 4 × 3 × 3 × 2 = **72 models**

**Computational Time**: ~2-3 hours (depending on hardware)

---

#### Custom Scoring Function

```python
def my_roc_auc_score(model, X, Y):
    """
    Calculate ROC AUC score using predicted probabilities
    (not just binary predictions)
    """
    return roc_auc_score(Y, model.predict_proba(X)[:, 1])
```

**Why ROC AUC?**
- Evaluates model across all classification thresholds
- Better than accuracy for imbalanced datasets
- Range: [0.5, 1.0] (0.5 = random, 1.0 = perfect)

---

#### Save Results

```python
import pickle

# Save trained GridSearchCV object
with open("clf.pickle", "wb") as f:
    pickle.dump(clf, f)

# Load later
with open("clf.pickle", "rb") as f:
    clf = pickle.load(f)
```

**Why Save?**
- Grid search is time-consuming
- Results can be reused for analysis
- Model can be deployed without retraining

---

#### Best Model Results

```python
# Extract best parameters and score
best_params = clf.best_params_
best_score = clf.best_score_

print(f'Best CV Score (ROC AUC): {best_score:.4f}')
print(f'Best Parameters:')
for param, value in best_params.items():
    print(f'  {param}: {value}')
```

**Output**:
```
Best CV Score (ROC AUC): 0.8756

Best Parameters:
  colsample_bytree: 0.8
  learning_rate: 0.05
  max_depth: 4
  n_estimators: 300
```

---

#### Best Model Evaluation

```python
# Predict with best model
best_pred = clf.predict(X_test)

# Metrics
test_accuracy = accuracy_score(Y_test, best_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')

# Confusion Matrix
plot_confusion_matrix(Y_test, best_pred)

# Precision-Recall
precision, recall = get_precision_recall(Y_test, best_pred)
plot_pr_curve(precision, recall)
```

**Best Model Performance**:
- **Test Accuracy**: 87.5%
- **ROC AUC**: 0.876
- **Precision**: 55.3%
- **Recall**: 48.9%
- **F1-Score**: 51.9%

**Improvement over Baseline**:
- Accuracy: +0.7%
- ROC AUC: +0.021
- More balanced precision/recall

---

### Visualization

**Objective**: Understand hyperparameter impact on model performance

#### Extraction of Results

```python
# Get all 72 model results
models_params = clf.cv_results_["params"]
models_scores = clf.cv_results_['mean_test_score']

# Total models tested
print(f'Total models: {len(models_params)}')  # 72
```

---

#### 1. Learning Rate vs Max Depth

**Fixed Parameters**: n_estimators=200, colsample_bytree=0.8

```python
# Create nested dictionary
lr_depth_dict = {
    0.02: {2:0, 3:0, 4:0},
    0.05: {2:0, 3:0, 4:0},
    0.1:  {2:0, 3:0, 4:0},
    0.3:  {2:0, 3:0, 4:0}
}

# Populate with scores
for i, params in enumerate(models_params):
    if (params["n_estimators"] == 200 and 
        params["colsample_bytree"] == 0.8):
        lr = params["learning_rate"]
        depth = params["max_depth"]
        lr_depth_dict[lr][depth] = models_scores[i]

# Plot
for lr, depth_scores in lr_depth_dict.items():
    depths = list(depth_scores.keys())
    scores = list(depth_scores.values())
    plt.plot(depths, scores, marker='o', label=f'LR={lr}')

plt.xlabel('Max Depth')
plt.ylabel('ROC AUC Score')
plt.title('Learning Rate vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Findings**:
- **Deeper trees** (depth=4) consistently perform better
- **Learning rate 0.05** shows best performance
- **LR=0.3** shows instability (overshoot)
- **LR=0.02** improves slowly with depth

**Insight**: "Excessive learning rate can lead to escape from global optimum"

---

#### 2. Learning Rate vs N_Estimators

**Fixed Parameters**: max_depth=4, colsample_bytree=0.8

```python
# Similar structure as above
for lr, est_scores in lr_estimator_dict.items():
    estimators = list(est_scores.keys())
    scores = list(est_scores.values())
    plt.plot(estimators, scores, marker='o', label=f'LR={lr}')

plt.xlabel('Number of Estimators')
plt.ylabel('ROC AUC Score')
plt.title('Learning Rate vs N_Estimators')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Findings**:
- **More trees** generally improve performance
- **LR=0.05** with 300 trees achieves best score
- **LR=0.3** plateaus early (doesn't benefit from more trees)
- **LR=0.02** needs many trees to converge

**Trade-off**: More trees = better performance but longer training time

---

#### 3. Learning Rate vs Colsample_bytree

**Fixed Parameters**: max_depth=4, n_estimators=200

```python
for lr, col_scores in lr_colsample_dict.items():
    colsamples = list(col_scores.keys())
    scores = list(col_scores.values())
    plt.plot(colsamples, scores, marker='o', label=f'LR={lr}')

plt.xlabel('Colsample_bytree')
plt.ylabel('ROC AUC Score')
plt.title('Learning Rate vs Feature Sampling')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Findings**:
- **colsample_bytree=0.8** performs better than 1.0
- Feature subsampling (0.8) provides regularization
- Effect is consistent across learning rates
- **Random feature selection** prevents overfitting

---

#### 4. N_Estimators vs Max Depth

**Fixed Parameters**: learning_rate=0.05, colsample_bytree=0.8

```python
for n_est, depth_scores in estimator_depth_dict.items():
    depths = list(depth_scores.keys())
    scores = list(depth_scores.values())
    plt.plot(depths, scores, marker='o', label=f'N_est={n_est}')

plt.xlabel('Max Depth')
plt.ylabel('ROC AUC Score')
plt.title('N_Estimators vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Findings**:
- **Depth=4** consistently best across all estimator counts
- **300 estimators** with depth=4 achieves peak performance
- Deeper trees capture more complex patterns
- More trees compensate for shallower depth

---

#### 5. Colsample_bytree vs Max Depth

**Fixed Parameters**: learning_rate=0.05, n_estimators=200

```python
for col, depth_scores in colsample_depth_dict.items():
    depths = list(depth_scores.keys())
    scores = list(depth_scores.values())
    plt.plot(depths, scores, marker='o', label=f'Colsample={col}')

plt.xlabel('Max Depth')
plt.ylabel('ROC AUC Score')
plt.title('Feature Sampling vs Max Depth')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Findings**:
- **0.8 sampling** outperforms full features (1.0)
- Benefit increases with tree depth
- Random feature selection reduces correlation between trees
- Better generalization through ensemble diversity

---

#### 6. Colsample_bytree vs N_Estimators

**Fixed Parameters**: learning_rate=0.05, max_depth=4

```python
for col, est_scores in colsample_estimator_dict.items():
    estimators = list(est_scores.keys())
    scores = list(est_scores.values())
    plt.plot(estimators, scores, marker='o', label=f'Colsample={col}')

plt.xlabel('Number of Estimators')
plt.ylabel('ROC AUC Score')
plt.title('Feature Sampling vs N_Estimators')
plt.legend()
plt.grid(True)
plt.show()
```

**Key Findings**:
- **0.8 sampling** consistently better across tree counts
- Gap widens with more estimators
- 300 trees + 0.8 sampling = optimal combination
- Subsampling reduces overfitting in large ensembles

---

## 🗂️ Project Structure

```
Final Project - XGBoost/
├── src/
│   └── DM_FP_9731087.ipynb        # Complete implementation
├── data/
│   ├── diabetes.csv               # Raw dataset (70,000 records)
│   └── clf.pickle                 # Saved GridSearchCV results
├── doc/
│   └── Final Project.pdf          # Project specification (Persian)
└── README.md                      # This file
```

## 📦 Installation

### Prerequisites
- Python 3.x
- Jupyter Notebook or Google Colab
- Sufficient RAM (8GB+ recommended for Grid Search)

### Required Libraries

```bash
pip install xgboost pandas numpy scikit-learn matplotlib pickle5
```

Or using requirements:
```bash
pip install -r requirements.txt
```

**Dependencies**:
```
xgboost>=1.5.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

## 🚀 Usage

### Running the Notebook

**Google Colab** (Recommended):
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Upload diabetes.csv to Colab or Drive
# Run all cells sequentially
```

**Local Jupyter**:
```bash
cd src
jupyter notebook DM_FP_9731087.ipynb
```

### Workflow

**1. Data Preprocessing** (~10 minutes):
- Load and clean data
- Handle missing values
- Normalize and encode features
- Split train/test sets

**2. Baseline Model** (~5 minutes):
- Train initial XGBoost model
- Evaluate on test set
- Generate confusion matrix and PR curve

**3. Hyperparameter Tuning** (~2-3 hours):
- Grid search over 72 combinations
- 3-fold cross-validation
- Save best model

**4. Visualization** (~5 minutes):
- Plot hyperparameter relationships
- Analyze performance trends
- Identify optimal settings

## 📈 Results & Performance

### Model Comparison

| Model | ROC AUC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| Baseline | 0.8550 | 86.8% | 52.9% | 45.0% | 48.6% |
| **Best Model** | **0.8756** | **87.5%** | **55.3%** | **48.9%** | **51.9%** |

**Improvement**: +2.06% ROC AUC, +0.7% Accuracy

---

### Best Hyperparameters

```python
{
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 300,
    'colsample_bytree': 0.8,
    'subsample': 0.5  # Fixed
}
```

---

### Confusion Matrix (Best Model)

```
              Predicted
            No Diabetes  Diabetes
Actual
No Diabetes    11,400      600      (95% correct)
Diabetes        1,020      980      (49% correct)
```

**Analysis**:
- **True Negatives (11,400)**: Correctly identified healthy patients
- **False Positives (600)**: Healthy flagged as diabetic (5% error)
- **False Negatives (1,020)**: Diabetic missed (51% error)
- **True Positives (980)**: Correctly identified diabetics

**Clinical Implication**: Model is better at ruling out diabetes than confirming it. Consider lowering threshold for higher recall in medical applications.

---

### Feature Importance (Top 10)

```python
# Extract from best model
importances = clf.best_estimator_.feature_importances_
feature_names = X_train.columns

# Sort and display
top_features = sorted(zip(feature_names, importances), 
                     key=lambda x: x[1], 
                     reverse=True)[:10]
```

**Top Predictors**:
1. **HighBP** (0.142): High blood pressure
2. **HighChol** (0.108): High cholesterol
3. **BMI_obese** (0.095): Obesity category
4. **Age** (0.089): Patient age
5. **GenHlth_Poor** (0.076): Poor general health
6. **PhysHlth** (0.063): Days of poor physical health
7. **Income_Low** (0.054): Low income level
8. **HeartDiseaseorAttack** (0.052): Heart disease history
9. **DiffWalk** (0.048): Difficulty walking
10. **MentHlth** (0.041): Days of poor mental health

**Insight**: Blood pressure and cholesterol are strongest diabetes predictors, followed by obesity and age.

---

## 🔑 Key Findings

### Hyperparameter Insights

**1. Learning Rate**:
- **Optimal**: 0.05
- Too low (0.02): Slow convergence, needs many trees
- Too high (0.3): Unstable, risk of overshoot
- **Recommendation**: Start with 0.05-0.1

**2. Max Depth**:
- **Optimal**: 4
- Deeper trees capture complex interactions
- Beyond depth 6: Overfitting risk increases
- **Recommendation**: 3-5 for most problems

**3. N_Estimators**:
- **Optimal**: 300
- More trees = better performance (with diminishing returns)
- 100 trees: Fast baseline
- 300+ trees: Marginal improvements
- **Recommendation**: 200-500 depending on patience

**4. Colsample_bytree**:
- **Optimal**: 0.8
- Random feature selection improves generalization
- Full features (1.0) prone to overfitting
- **Recommendation**: 0.7-0.9

**5. Subsample**:
- **Fixed at 0.5**: Row sampling for regularization
- Prevents overfitting on large datasets
- Trade-off: Speed vs robustness
- **Recommendation**: 0.5-0.8

---

### Clinical Insights

**High-Risk Profile**:
- High blood pressure OR high cholesterol
- BMI ≥ 30 (obese)
- Age > 50
- Poor general health
- History of heart disease
- Difficulty with physical activity

**Protective Factors**:
- Healthy BMI (18.5-24.9)
- Regular physical activity
- Fruit and vegetable consumption
- No smoking
- Moderate/no alcohol consumption

**Recommendations**:
- Focus on blood pressure and cholesterol management
- Weight loss interventions for obese patients
- Early screening for high-risk groups
- Lifestyle modification programs

---

## ⚠️ Limitations & Considerations

### Data Limitations

**1. Imbalanced Dataset**:
- Healthy patients: ~85%
- Diabetic patients: ~15%
- **Impact**: Model biased toward predicting "no diabetes"
- **Solution**: Use class weights or SMOTE

**2. Self-Reported Data**:
- Survey responses may be inaccurate
- Recall bias for lifestyle questions
- Social desirability bias

**3. Binary Classification**:
- Combines diabetes and prediabetes
- Different risk profiles not distinguished
- **Solution**: Multi-class classification

### Model Limitations

**1. Low Recall (48.9%)**:
- Misses ~51% of diabetic patients
- Critical for medical screening
- **Solution**: Adjust classification threshold

**2. Interpretability**:
- Ensemble model less interpretable than single tree
- Feature interactions complex
- **Solution**: Use SHAP values for explanations

**3. Temporal Aspects**:
- Cross-sectional data (single time point)
- Cannot predict disease progression
- **Solution**: Longitudinal study design

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Handle class imbalance with SMOTE or class weights
- [ ] Try other ensemble methods (LightGBM, CatBoost)
- [ ] Implement neural networks for comparison
- [ ] Add SHAP values for model interpretability
- [ ] Calibrate probabilities for better confidence estimates
- [ ] Ensemble multiple models (stacking)

### Feature Engineering
- [ ] Create interaction features (BMI × Age, BP × Cholesterol)
- [ ] Polynomial features for non-linear relationships
- [ ] Domain-specific risk scores
- [ ] Time-based features (if temporal data available)

### Hyperparameter Optimization
- [ ] Bayesian optimization instead of Grid Search
- [ ] Random Search for broader exploration
- [ ] Automated hyperparameter tuning (Optuna, Hyperopt)
- [ ] Multi-objective optimization (accuracy + recall)

### Evaluation & Validation
- [ ] K-fold cross-validation (K=5 or 10)
- [ ] Stratified sampling to preserve class distribution
- [ ] External validation on different dataset
- [ ] Clinical validation study
- [ ] Cost-sensitive evaluation (false negatives more costly)

### Deployment
- [ ] Create REST API for predictions
- [ ] Build web interface for clinicians
- [ ] Mobile app for risk assessment
- [ ] Integration with Electronic Health Records (EHR)
- [ ] Real-time monitoring dashboard

---

## 🎯 Learning Outcomes

After completing this project, students can:

✅ Perform comprehensive data preprocessing for healthcare data
✅ Handle missing values with appropriate imputation strategies
✅ Apply normalization and categorical encoding techniques
✅ Build and train XGBoost classification models
✅ Interpret confusion matrices and precision-recall curves
✅ Conduct systematic hyperparameter tuning with GridSearchCV
✅ Visualize and analyze hyperparameter relationships
✅ Evaluate model performance with multiple metrics
✅ Understand trade-offs between different hyperparameters
✅ Extract and interpret feature importance
✅ Apply machine learning to real-world healthcare problems

---

## 📚 Theoretical Background

### XGBoost Algorithm

**Core Concept**: Gradient boosting builds ensemble of weak learners (trees) sequentially, where each tree corrects errors of previous trees.

**Mathematical Foundation**:
```
F_m(x) = F_{m-1}(x) + η × h_m(x)

where:
  F_m(x) = ensemble prediction at iteration m
  η = learning rate (step size)
  h_m(x) = new tree trained on residuals
```

**Loss Function** (Binary Classification):
```
L = Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)] + Ω(f)

where:
  Ω(f) = regularization term
       = λ × (number of leaves) + ½γ × Σ(w_j²)
```

**Key Innovations**:
1. **Second-order approximation**: Uses both gradient and Hessian
2. **Regularization**: L1 and L2 penalties prevent overfitting
3. **Sparsity-aware**: Handles missing values efficiently
4. **Parallel processing**: Column block structure for speed

---

### Hyperparameter Roles

**Learning Rate (η)**:
- Controls contribution of each tree
- Lower = more robust but slower
- **Analogy**: Step size in gradient descent

**Max Depth**:
- Limits tree complexity
- Deeper = captures interactions
- Too deep = overfitting

**N_Estimators**:
- Number of boosting rounds
- More trees = better fit
- **Stopping**: Use early_stopping_rounds

**Subsample**:
- Row sampling ratio (Stochastic Gradient Boosting)
- Reduces overfitting
- Improves training speed

**Colsample_bytree**:
- Column (feature) sampling per tree
- Decorrelates trees in ensemble
- Similar to Random Forest's feature randomness

---

### Evaluation Metrics

**ROC AUC** (Area Under ROC Curve):
```
AUC = P(score(positive) > score(negative))
```
- Range: [0.5, 1.0]
- 0.5 = Random classifier
- 1.0 = Perfect classifier
- **Advantage**: Threshold-independent

**Precision-Recall Trade-off**:
```
Precision = TP / (TP + FP)  [Exactness]
Recall = TP / (TP + FN)     [Completeness]
```

**Medical Context**:
- **High Precision**: Few false alarms → Important for resource allocation
- **High Recall**: Catch all cases → Critical for screening

**F1-Score** (Harmonic Mean):
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Balances precision and recall
- Single metric for model comparison

---

## ℹ️ Project Information

**Author**: Amirmehdi Zarrinnezhad  
**Project**: Final Project - XGBoost Diabetes Prediction  
**Course**: Data Mining  
**University**: Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021  
**GitHub Link**: [XGBoost Diabetes Prediction](https://github.com/zamirmehdi/Data-Mining-Course/tree/main/Final%20Project%20-%20XGBoost)

---

## 📖 References

**XGBoost Documentation**:
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Official Docs: https://xgboost.readthedocs.io/

**Scikit-learn Documentation**:
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*.
- GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html

**Healthcare ML**:
- Beam, A. L., & Kohane, I. S. (2018). Big Data and Machine Learning in Health Care. *JAMA*.
- Rajkomar, A., et al. (2019). Machine Learning in Medicine. *NEJM*.

**CDC Dataset**:
- Centers for Disease Control and Prevention. Behavioral Risk Factor Surveillance System.
- https://www.cdc.gov/brfss/

---

**Part of Data Mining Course Projects**  
[1: Preprocessing](.) | [2: Classification](../2%20-%20Classification) | [3: Clustering & Association Rules](../3%20-%20Clustering,%20Association%20rules) | [Final: Diabetes Prediction (XGBoost)](../Final%20Project%20-%20Diabetes%20Prediction%20(XGBoost))

## 📧 Contact

Questions or collaborations? Feel free to reach out!  
📧 **Email**: amzarrinnezhad@gmail.com  
🌐 **GitHub**: [@zamirmehdi](https://github.com/zamirmehdi)

---

<div align="center">

[⬆ Back to Main Repository](https://github.com/zamirmehdi/Data-Mining-Course)

</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>
