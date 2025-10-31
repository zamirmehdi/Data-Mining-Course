# HW1 - Data Preprocessing

A comprehensive data preprocessing pipeline implementing essential techniques for handling missing values, encoding categorical data, normalization, dimensionality reduction, and visualization using the Iris dataset.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Processing-green.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](#)

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Overview](#-overview)
- [Implementation Pipeline](#-implementation-pipeline)
- [Dataset](#-dataset)
- [Preprocessing Steps](#-preprocessing-steps)
  - [Part 1: Missing Value Detection & Removal](#part-1-missing-value-detection--removal)
  - [Part 2: Label Encoding](#part-2-label-encoding)
  - [Part 3: Feature Normalization](#part-3-feature-normalization)
  - [Part 4: Principal Component Analysis (PCA)](#part-4-principal-component-analysis-pca)
  - [Part 5: Data Visualization](#part-5-data-visualization)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Concepts Demonstrated](#-key-concepts-demonstrated)
- [Results & Analysis](#-results--analysis)
- [Known Issues & Limitations](#-known-issues--limitations)
- [Future Enhancements](#-future-enhancements)
- [Theoretical Background](#-theoretical-background)
- [Learning Outcomes](#-learning-outcomes)
- [Course Information](#-course-information)
- [Contact](#-contact)

</details>

## 📋 Overview

This project demonstrates fundamental data preprocessing techniques essential for data mining and machine learning pipelines. The implementation covers the complete workflow from raw data to visualization-ready features using the classic Iris dataset.

**Key Learning Objectives:**
- ✅ Identify and handle missing values (NaN detection and removal)
- ✅ Encode categorical variables using Label Encoding
- ✅ Normalize numerical features using StandardScaler (Z-score)
- ✅ Reduce dimensionality with PCA (4D → 2D transformation)
- ✅ Visualize data distributions and class separations

## 🎯 Implementation Pipeline

```
Raw Data (iris.data)
    ↓
[1] Missing Value Detection & Removal
    ↓
[2] Label Encoding (Target Classes)
    ↓
[3] Feature Normalization (StandardScaler)
    ↓
[4] Dimensionality Reduction (PCA)
    ↓
[5] Data Visualization (Scatter & Box Plots)
```

## 📊 Dataset

**Iris Dataset:**
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 150 instances (145 after removing missing values)
- **Features**: 4 continuous attributes
  - `sepal_length` (cm): Sepal length measurement
  - `sepal_width` (cm): Sepal width measurement
  - `petal_length` (cm): Petal length measurement
  - `petal_width` (cm): Petal width measurement
- **Target Classes**: 3 species
  - `Iris-setosa` → Encoded as **0**
  - `Iris-versicolor` → Encoded as **1**
  - `Iris-virginica` → Encoded as **2**
- **Missing Values**: 5 rows with NaN values (artificially introduced)

**Missing Value Distribution:**
```
sepal_length: 2 NaN values
sepal_width:  0 NaN values
petal_length: 2 NaN values
petal_width:  3 NaN values
target:       1 NaN value
```

## 🔧 Preprocessing Steps

### Part 1: Missing Value Detection & Removal

**Objective**: Identify and eliminate rows containing missing values.

**Implementation:**
```python
def print_nans():
    """Detect and display missing values in each column"""
    print('Number of sepal_length NaNs:', len(df[df['sepal_length'].isna()]))
    print('Number of petal_width NaNs:', len(df[df['petal_width'].isna()]))
    # ... for all columns

def remove_nans():
    """Remove all rows containing NaN values"""
    df = df.dropna()
    df.reset_index(drop=True)
```

**Results:**
- Original dataset: 150 rows
- After removal: 145 rows
- Removed: 5 rows (~3.3% data loss)

**Rationale**: With minimal missing data (<5%), row deletion is appropriate. Alternative imputation methods (mean, median, mode) could be used for larger missing data percentages.

### Part 2: Label Encoding

**Objective**: Convert categorical target labels to numerical values for ML algorithms.

**Technique**: Label Encoding using `sklearn.preprocessing.LabelEncoder`

**Transformation:**
```python
def label_encoder():
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['target'])
```

**Encoding Scheme:**
| Original Class | Encoded Value |
|---------------|---------------|
| Iris-setosa | 0 |
| Iris-versicolor | 1 |
| Iris-virginica | 2 |

**Important Note**: Label Encoding creates ordinal relationships (0 < 1 < 2) which may not exist in reality. For non-ordinal categorical data, **One-Hot Encoding** is often preferred to avoid implicit ordering assumptions.

### Part 3: Feature Normalization

**Objective**: Standardize features to improve model convergence and performance.

**Technique**: Z-score normalization (Standardization) using `sklearn.preprocessing.StandardScaler`

**Formula:**
```
x_normalized = (x - μ) / σ
where:
  μ = mean of feature
  σ = standard deviation of feature
```

**Implementation:**
```python
def normalization():
    standard_sc = StandardScaler()
    # Normalize only features, not target
    normalized_features = standard_sc.fit_transform(df.iloc[:, 0:-1])
```

**Before Normalization:**
```
sepal_length: mean=5.84, variance=0.68
sepal_width:  mean=3.06, variance=0.19
petal_length: mean=3.76, variance=3.09
petal_width:  mean=1.20, variance=0.58
```

**After Normalization:**
```
sepal_length: mean≈0.0, variance≈1.0
sepal_width:  mean≈0.0, variance≈1.0
petal_length: mean≈0.0, variance≈1.0
petal_width:  mean≈0.0, variance≈1.0
```

**Benefits:**
- All features on same scale → fair weight distribution
- Faster convergence for gradient-based algorithms
- Improved numerical stability
- Better distance metric calculations (e.g., Euclidean distance)

**Visualization**: Box plots generated before and after normalization show feature distributions centered at 0 with comparable spreads.

### Part 4: Principal Component Analysis (PCA)

**Objective**: Reduce feature dimensionality from 4D to 2D for visualization and computational efficiency.

**Technique**: PCA using `sklearn.decomposition.PCA`

**Implementation:**
```python
def do_pca():
    pca = PCA(n_components=2)
    # Transform 4D features to 2D
    transformed_data = pca.fit_transform(data_set.iloc[:, 0:-1])
```

**Dimensionality Reduction:**
```
Original: [sepal_length, sepal_width, petal_length, petal_width]
          ↓
PCA Transformation
          ↓
Reduced:  [PC1, PC2]
```

**Principal Components:**
- **PC1 (Principal Component 1)**: Captures maximum variance
- **PC2 (Principal Component 2)**: Captures second-most variance
- **Variance Retained**: ~95-97% (typical for Iris dataset)

**Benefits:**
- Reduces computational complexity
- Removes multicollinearity
- Enables 2D visualization
- Retains most important information

**Note**: PCA is performed **after normalization** to ensure equal feature importance weighting.

### Part 5: Data Visualization

**Objective**: Visualize data distributions and class separations.

#### 5.1 PCA Scatter Plot

**Implementation:**
```python
def visualize(data_set):
    colors = {'0': 'red', '1': 'grey', '2': 'yellow'}
    for i in range(len(data_set)):
        ax.scatter(data_set['col 1'][i], data_set['col 2'][i],
                   color=colors[str(int(data_set['target'][i]))])
```

**Visualization Features:**
- **X-axis**: First principal component (PC1)
- **Y-axis**: Second principal component (PC2)
- **Colors**: 
  - Red: Iris-setosa (0)
  - Grey: Iris-versicolor (1)
  - Yellow: Iris-virginica (2)

**Observations**:
- Clear separation of Iris-setosa from other classes
- Slight overlap between Iris-versicolor and Iris-virginica
- PCA successfully preserves class structure in 2D

#### 5.2 Box Plots

**Purpose**: Visualize feature distributions, detect outliers, and compare scales.

**Plots Generated**:
1. **Before Normalization**: Shows original feature scales and ranges
2. **After Normalization**: Shows standardized distributions centered at 0

**Box Plot Components**:
- **Box**: Interquartile range (IQR) - middle 50% of data
- **Line**: Median value
- **Whiskers**: Min/Max within 1.5×IQR
- **Dots**: Outliers beyond whiskers

## 🗂️ Project Structure

```
HW1 - Data Preprocessing/
├── data/
│   └── iris.data              # Raw Iris dataset (150 samples)
├── src/
│   └── main.py                # Complete preprocessing pipeline
├── doc/
|   ├── Project1.pdf               # Assignment instructions (Persian)
|   └── Report1_Amirmehdi Zarrinnezhad.pdf  # Implementation report
└── README.md                  # This file
```

## 📦 Installation

### Prerequisites
- Python 3.x
- pip package manager

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib
```

Or using requirements file:
```bash
pip install -r requirements.txt
```

**Dependencies:**
```
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
matplotlib>=3.3.0
```

## 🚀 Usage

### Basic Execution

1. **Update dataset path** in `main.py`:
```python
dataset_path = '/path/to/iris.data'
```

2. **Run the pipeline**:
```bash
cd src
python main.py
```

### Expected Output

```
-Part1:

Number of sepal_length NaNs: 2
Number of sepal_width NaNs: 0
Number of petal_length NaNs: 2
Number of petal_width NaNs: 3
Number of target NaNs: 1

Number of rows = 150
NaNs were removed. Number of rows = 145

-Part2:

Before encode:
   sepal_length  sepal_width  petal_length  petal_width        target
0           5.1          3.5           1.4          0.2   Iris-setosa
...

After encode:
   sepal_length  sepal_width  petal_length  petal_width  target
0           5.1          3.5           1.4          0.2       0
...

-Part3:

Before normalization:
sepal_length: mean=5.84, variance=0.68
...

After normalization:
sepal_length: mean≈0.0, variance≈1.0
...

-Part4:

After PCA:
     col 1    col 2  target
0  -2.68...  0.32...     0.0
...
```

### Visualization Windows

The script will display **3 matplotlib windows**:
1. **Box Plot (Before Normalization)**: Original feature scales
2. **Box Plot (After Normalization)**: Standardized features
3. **PCA Scatter Plot**: 2D projection with class colors

## 🎓 Key Concepts Demonstrated

### Data Preprocessing
- Missing value detection (`isna()`)
- Row deletion (`dropna()`)
- Data frame operations (pandas)

### Feature Engineering
- Label encoding for categorical variables
- Z-score normalization (standardization)
- Dimensionality reduction with PCA

### Scikit-learn Pipeline
- `LabelEncoder`: Categorical → numerical conversion
- `StandardScaler`: Feature normalization
- `PCA`: Dimensionality reduction

### Data Visualization
- Box plots for distribution analysis
- Scatter plots for class separation
- Color coding for multi-class visualization

## 📈 Results & Analysis

### Normalization Impact

**Before Normalization:**
- Features have different scales (e.g., petal_length variance = 3.09 vs sepal_width variance = 0.19)
- Models may give undue importance to larger-scale features
- Distance metrics biased toward high-variance features

**After Normalization:**
- All features centered at μ ≈ 0
- All features have σ ≈ 1
- Equal treatment by ML algorithms
- Improved model performance

### PCA Effectiveness

**Variance Explained:**
- PC1: ~72-73% of total variance
- PC2: ~22-23% of total variance
- **Total**: ~95-97% variance retained with 50% dimension reduction

**Class Separation:**
- Excellent separation of Iris-setosa
- Moderate separation of Iris-versicolor and Iris-virginica
- Minimal information loss despite dimension reduction

## ⚠️ Known Issues & Limitations

### Current Implementation

1. **Hardcoded Path**: Dataset path is hardcoded in script
   - **Fix**: Use command-line arguments or config file

2. **No Imputation**: Missing values are simply deleted
   - **Alternative**: Implement mean/median/KNN imputation

3. **Fixed PCA Components**: Always reduces to 2D
   - **Enhancement**: Make n_components configurable

4. **No Train-Test Split**: All data used for transformation
   - **Risk**: Information leakage if used for modeling

5. **No Pipeline Object**: Manual step-by-step execution
   - **Better**: Use `sklearn.pipeline.Pipeline` for reproducibility

### Important Notes

- **Label Encoding Caveat**: Creates ordinal relationships (0 < 1 < 2) that don't exist in nature
  - Iris species are nominal (no inherent ordering)
  - For tree-based models: Acceptable
  - For distance-based models: Consider One-Hot Encoding

- **Normalization Timing**: Must normalize before PCA
  - PCA is affected by feature scales
  - Always standardize first for proper variance calculation

## 🔮 Future Enhancements

- [ ] Implement multiple imputation strategies (mean, median, KNN)
- [ ] Add One-Hot Encoding comparison
- [ ] Create sklearn Pipeline for reproducibility
- [ ] Add command-line interface (argparse)
- [ ] Generate automated report with plots
- [ ] Implement train-test splitting
- [ ] Add more normalization techniques (Min-Max, Robust Scaler)
- [ ] Include outlier detection and removal options
- [ ] Add correlation heatmap visualization
- [ ] Compare PCA with other dimensionality reduction (t-SNE, UMAP)

## 📚 Theoretical Background

### Covered Topics (From Project1.pdf)

**Written Section Topics:**
1. Data mining concepts (Dimension, Outlier, Variables, Sampling)
2. Normalization methods (Min-Max, Z-Score, Decimal Scaling)
3. Discretization techniques (ChiMerge algorithm)
4. Similarity measures (Cosine, Correlation, Euclidean, Jaccard)
5. Data reduction strategies
6. Feature selection vs Feature extraction
7. Wavelet Transform for dimensionality reduction
8. Box plot interpretation
9. Noise vs Outlier distinction
10. Quantile-quantile plots
11. Dissimilarity measures for mixed data types

**Programming Section:**
- Missing value handling
- Categorical encoding
- Feature normalization
- PCA transformation
- Data visualization

## 🎯 Learning Outcomes

After completing this project, students can:

✅ Detect and handle missing values appropriately
✅ Choose between deletion and imputation strategies
✅ Encode categorical variables for ML algorithms
✅ Normalize features to improve model performance
✅ Apply PCA for dimensionality reduction
✅ Visualize high-dimensional data in 2D/3D
✅ Use scikit-learn preprocessing tools effectively
✅ Understand preprocessing impact on ML pipelines

## ℹ️ Project Information

**Author**: Amirmehdi Zarrinnezhad  
**Assignment**: Data Preprocessing  
**Course**: Data Mining  
**University**: Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021    
**Github Link**: [Data-Preprocessing](https://github.com/zamirmehdi/Data-Mining-Course/new/main/HW1%20-%20Data%20Preprocessing)

<div align="center">

**Part of Data Mining Course Projects**

[HW1: Preprocessing](.) | [HW2: Classification](../HW2%20-%20Classification) | [HW3: Clustering & Association Rules](../HW3%20-%20Clustering,%20Association%20rules) | [Final: XGBoost](../Final%20Project%20-%20XGBoost)

</div>

## 📧 Contact

Questions or collaborations? Feel free to reach out!  
📧 Email: amzarrinnezhad@gmail.com  
🌐 GitHub: [@zamirmehdi](https://github.com/zamirmehdi)

---

<div align="center">

[⬆ Back to Main Repository](ps://github.com/zamirmehdi/Data-Mining-Course)

</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>
