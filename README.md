<h1 align="center">Data Mining Course Projects</h1>

A comprehensive collection of 4 data mining and machine learning projects covering preprocessing, classification, clustering, association rules, and gradient boosting. Implemented as part of the Data Mining course at Amirkabir University of Technology.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![University](https://img.shields.io/badge/University-AUT-red.svg)](https://aut.ac.ir/en)
[![Course](https://img.shields.io/badge/Course-Data%20Mining-orange.svg)](#)

<div align="center">

![Data Mining](https://img.shields.io/badge/Data-Mining-brightgreen?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-orange?style=for-the-badge)

</div>

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Projects](#-projects)
  - [HW1: Data Preprocessing](#hw1-data-preprocessing)
  - [HW2: Classification](#hw2-classification)
  - [HW3: Clustering & Association Rules](#hw3-clustering--association-rules)
  - [Final Project: XGBoost](#final-project-xgboost)
- [Technologies & Tools](#%EF%B8%8F-technologies--tools)
- [Repository Structure](#-repository-structure)
- [Installation](#%EF%B8%8F-installation)
- [Key Concepts Covered](#-key-concepts-covered)
- [Learning Outcomes](#-learning-outcomes)
- [Course Information](#ℹ%EF%B8%8F-course-information)
- [Contact](#-contact)
<!-- 
 - [Acknowledgments](#-acknowledgments)
 - [License](#-license) 
 -->

---

## 🎯 Overview

This repository contains four comprehensive projects that explore fundamental data mining and machine learning techniques. Each project demonstrates end-to-end implementation from data preprocessing to model evaluation, with detailed documentation and analysis.

**Repository Highlights:**
- 🔥 **4 Complete Projects** with full implementation
- 📊 **Real-world Datasets** (Iris, Fashion MNIST, Hypermarket, CDC Health Survey)
- 🤖 **Multiple ML Algorithms** (K-Means, DBSCAN, Neural Networks, XGBoost)
- 📈 **Comprehensive Visualizations** and performance analysis
- 📝 **Detailed Documentation** in both English and Persian
- ✅ **Production-Ready Code** with best practices

---

## 📂 Projects

### HW1: Data Preprocessing

<img src="https://img.shields.io/badge/Status-Complete-success" alt="Complete"/> <img src="https://img.shields.io/badge/Difficulty-Beginner-green" alt="Beginner"/>

**Focus**: Data cleaning, normalization, dimensionality reduction, and visualization

**Key Techniques:**
- Missing value detection and imputation
- Label encoding for categorical variables
- Z-score normalization (StandardScaler)
- Principal Component Analysis (PCA) for 4D→2D reduction
- Box plots and scatter visualizations

**Dataset**: Iris (150 samples, 4 features, 3 classes)

**Highlights:**
- ✅ Handled 5 missing values with dropna strategy
- ✅ Reduced dimensionality while retaining 95-97% variance
- ✅ Achieved clear class separation in 2D visualization

**Skills Demonstrated:**
- Data quality assessment
- Feature scaling and transformation
- Dimensionality reduction
- Exploratory data analysis (EDA)

[📖 View Full Documentation →](./HW1%20-%20Data%20Preprocessing)

---

### HW2: Classification

<img src="https://img.shields.io/badge/Status-Complete-success" alt="Complete"/> <img src="https://img.shields.io/badge/Difficulty-Intermediate-yellow" alt="Intermediate"/>

**Focus**: Neural networks and deep learning for classification tasks

**Key Techniques:**
- Decision tree construction (manual calculation)
- Neural network architecture design
- Activation functions (ReLU, Sigmoid, Softmax)
- Convolutional Neural Networks (CNN)
- Evaluation metrics (Accuracy, Precision, Recall, F1-Score)

**Datasets**: 
- Circular data (1500 samples) - Binary classification
- Fashion MNIST (70,000 images) - 10-class classification

**Highlights:**
- ✅ Systematic experiments with 6 different architectures
- ✅ Achieved 98% accuracy on circular data with optimal configuration
- ✅ CNN achieved 89.4% accuracy on Fashion MNIST
- ✅ Comprehensive hyperparameter analysis (learning rate, activation functions)

**Skills Demonstrated:**
- Neural network design and training
- Overfitting detection and prevention
- Confusion matrix interpretation
- Model optimization techniques

[📖 View Full Documentation →](./HW2%20-%20Classification)

---

### HW3: Clustering & Association Rules

<img src="https://img.shields.io/badge/Status-Complete-success" alt="Complete"/> <img src="https://img.shields.io/badge/Difficulty-Intermediate-yellow" alt="Intermediate"/>

**Focus**: Unsupervised learning for pattern discovery and grouping

**Key Techniques:**

**Clustering:**
- K-Means clustering with elbow method
- DBSCAN for arbitrary-shaped clusters
- Hierarchical clustering (Single-link, Complete-link)
- Automatic ε selection using K-distance

**Association Rules:**
- Apriori algorithm for frequent itemset mining
- Support, Confidence, and Lift metrics
- Market basket analysis

**Datasets**:
- Synthetic data (blobs, circles, varying density)
- MNIST digits (1797 samples, 64 features)
- Hypermarket transactions (9000+ transactions, 160 products)

**Highlights:**
- ✅ K-Means: 85% accuracy on digit clustering
- ✅ DBSCAN: 0.92 V-Measure on anisotropic data
- ✅ Image compression: 75% size reduction with K=4 colors
- ✅ Association Rules: Discovered 243 rules (lift > 1.2)

**Skills Demonstrated:**
- Unsupervised learning algorithms
- Cluster validation metrics
- Market basket analysis
- Business intelligence from transactional data

[📖 View Full Documentation →](./HW3%20-%20Clustering,%20Association%20rules)

---

### Final Project: XGBoost

<img src="https://img.shields.io/badge/Status-Complete-success" alt="Complete"/> <img src="https://img.shields.io/badge/Difficulty-Advanced-red" alt="Advanced"/>

**Focus**: Gradient boosting for healthcare prediction with hyperparameter optimization

**Key Techniques:**
- Comprehensive data preprocessing (missing values, normalization, encoding)
- XGBoost classification model
- GridSearchCV with 72 model configurations
- 3-fold cross-validation
- Hyperparameter visualization and analysis

**Dataset**: CDC Diabetes Health Survey (70,000 patients, 22 features)

**Highlights:**
- ✅ Preprocessed 70,000 patient records
- ✅ Baseline accuracy: 86.8% | Best model: 87.5%
- ✅ ROC AUC: 0.876 (excellent discrimination)
- ✅ Systematic tuning of 4 hyperparameters
- ✅ Identified top 10 diabetes risk factors

**Best Model Configuration:**
```python
{
    'learning_rate': 0.05,
    'max_depth': 4,
    'n_estimators': 300,
    'colsample_bytree': 0.8
}
```

**Skills Demonstrated:**
- Healthcare data analysis
- Ensemble learning (gradient boosting)
- Hyperparameter optimization at scale
- Model evaluation for imbalanced data
- Clinical insight extraction

[📖 View Full Documentation →](./Final%20Project%20-%20XGBoost)

---

## 🛠️ Technologies & Tools

### Programming Languages
- ![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)

### Machine Learning & Data Science
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white)
- ![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-brightgreen)
- ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue?logo=pandas&logoColor=white)
- ![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy&logoColor=white)

### Visualization
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
- ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Viz-lightblue)

### Development Environment
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
- ![Google Colab](https://img.shields.io/badge/Google-Colab-yellow?logo=googlecolab&logoColor=white)

### Version Control
- ![Git](https://img.shields.io/badge/Git-Version%20Control-red?logo=git&logoColor=white)
- ![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github&logoColor=white)

---

## 📁 Repository Structure

```
Data-Mining-Course/
│
├── HW1 - Data Preprocessing/
│   ├── src/
│   │   └── main.py                    # Preprocessing pipeline
│   ├── data/
│   │   └── iris.data                  # Iris dataset
│   ├── doc/
│   │   ├── Project1.pdf               # Assignment (Persian)
│   │   └── Report1_Amirmehdi Zarrinnezhad.pdf
│   └── README.md                      # Full documentation
│
├── HW2 - Classification/
│   ├── src/
│   │   ├── Programming_part1.ipynb    # Neural network experiments
│   │   └── Programming_part2.ipynb    # Fashion MNIST CNN
│   ├── doc/
│   │   ├── Project2.pdf               # Assignment (Persian)
│   │   └── Report2_Amirmehdi Zarrinnezhad.pdf
│   └── README.md                      # Full documentation
│
├── HW3 - Clustering, Association rules/
│   ├── src/
│   │   ├── Clustering.ipynb           # K-Means, DBSCAN
│   │   └── AssociationRules.ipynb     # Apriori algorithm
│   ├── data/
│   │   ├── Hypermarket_dataset.csv    # Transaction data
│   │   └── bird.jpg                   # Image compression demo
│   ├── doc/
│   │   ├── Project3.pdf               # Assignment (Persian)
│   │   └── Report3_Amirmehdi Zarrinnezhad.pdf
│   └── README.md                      # Full documentation
│
├── Final Project - XGBoost/
│   ├── src/
│   │   └── DM_FP_9731087.ipynb        # Complete implementation
│   ├── data/
│   │   ├── diabetes.csv               # CDC health survey
│   │   └── clf.pickle                 # Trained model
│   ├── doc/
│   │   └── Final Project.pdf          # Assignment (Persian)
│   └── README.md                      # Full documentation
│
├── .gitignore
├── LICENSE
└── README.md                          # This file
```

---

## ⚙️ Installation

### Prerequisites
```bash
Python 3.7+
pip or conda
Jupyter Notebook (optional)
```

### Clone Repository
```bash
git clone https://github.com/zamirmehdi/Data-Mining-Course.git
cd Data-Mining-Course
```

### Install Dependencies

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Manual installation**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost mlxtend kneed pillow
```

### Verify Installation
```python
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import xgboost as xgb

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"XGBoost: {xgb.__version__}")
```

---

## 🎓 Key Concepts Covered

### Supervised Learning
- ✅ Classification (Binary & Multi-class)
- ✅ Decision Trees
- ✅ Neural Networks (Feedforward, CNN)
- ✅ Ensemble Methods (XGBoost)
- ✅ Model Evaluation & Selection

### Unsupervised Learning
- ✅ Clustering (K-Means, DBSCAN, Hierarchical)
- ✅ Dimensionality Reduction (PCA, Isomap)
- ✅ Association Rule Mining (Apriori)
- ✅ Pattern Discovery

### Data Preprocessing
- ✅ Missing Value Handling
- ✅ Feature Scaling & Normalization
- ✅ Encoding (Label, One-Hot)
- ✅ Data Transformation

### Model Optimization
- ✅ Hyperparameter Tuning (Grid Search, Cross-Validation)
- ✅ Regularization (Dropout, L1/L2)
- ✅ Early Stopping
- ✅ Ensemble Techniques

### Evaluation Metrics
- ✅ Classification: Accuracy, Precision, Recall, F1-Score, ROC AUC
- ✅ Clustering: V-Measure, Silhouette Score
- ✅ Association Rules: Support, Confidence, Lift

---

## 🏆 Learning Outcomes

By completing these projects, you will gain:

### Technical Skills
✅ **Data Manipulation**: Proficiency with Pandas, NumPy for data wrangling  
✅ **Visualization**: Creating insightful plots with Matplotlib, Seaborn  
✅ **ML Algorithms**: Understanding and implementing 10+ algorithms  
✅ **Deep Learning**: Building and training neural networks with TensorFlow  
✅ **Model Optimization**: Systematic hyperparameter tuning strategies  
✅ **Production Code**: Writing clean, documented, reproducible code

### Theoretical Understanding
✅ **Algorithm Theory**: Mathematical foundations of ML algorithms  
✅ **Bias-Variance Tradeoff**: Understanding model complexity  
✅ **Overfitting vs Underfitting**: Detection and prevention  
✅ **Feature Engineering**: Creating informative features  
✅ **Model Selection**: Choosing appropriate algorithms for problems

### Practical Experience
✅ **Real-world Datasets**: Handling messy, incomplete data  
✅ **End-to-End Pipeline**: From raw data to deployed model  
✅ **Performance Analysis**: Interpreting metrics and results  
✅ **Business Insights**: Extracting actionable recommendations  
✅ **Documentation**: Writing clear technical documentation

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Projects** | 4 |
| **Lines of Code** | ~5,000+ |
| **Datasets Processed** | 5 |
| **Total Samples** | ~150,000+ |
| **ML Algorithms** | 12+ |
| **Models Trained** | 100+ |
| **Visualizations** | 50+ |
| **Documentation Pages** | 40+ |

---

## 🎯 Use Cases

These projects demonstrate skills applicable to:

### Industries
- 🏥 **Healthcare**: Disease prediction, patient risk stratification
- 🛒 **Retail**: Customer segmentation, market basket analysis
- 💰 **Finance**: Credit scoring, fraud detection
- 📱 **Technology**: Image recognition, recommendation systems
- 🏭 **Manufacturing**: Quality control, predictive maintenance

### Job Roles
- 📊 Data Scientist
- 🤖 Machine Learning Engineer
- 📈 Data Analyst
- 🔬 Research Scientist
- 💼 Business Intelligence Analyst

---

## ℹ️ Course Information  
**Author**: Amirmehdi Zarrinnezhad  
**Course**: Data Mining (دادهکاوی)  
**University**: Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021  
**GitHub Link:** [Data Mining Course](https://github.com/zamirmehdi/Data-Mining-Course)  

<div align="center">

**Data Mining Course Projects Links**

[HW1: Preprocessing](./HW1%20-%20Data%20Preprocessing) • [HW2: Classification](./HW2%20-%20Classification) • [HW3: Clustering](./HW3%20-%20Clustering,%20Association%20rules) • [Final: XGBoost](./Final%20Project%20-%20XGBoost)
</div>

## 📧 Contact

Questions or collaborations? Feel free to reach out!  
📧 Email: amzarrinnezhad@gmail.com  
💬 Open an [Issue](https://github.com/zamirmehdi/Data-Mining-Course/issues)  
🌐 GitHub: [@zamirmehdi](https://github.com/zamirmehdi) 

<!--

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2021 Amirmehdi Zarrinnezhad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
-->

---

<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>

<!--
---

<div align="center">

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zamirmehdi/Data-Mining-Course&type=Date)](https://star-history.com/#zamirmehdi/Data-Mining-Course&Date)

-->
