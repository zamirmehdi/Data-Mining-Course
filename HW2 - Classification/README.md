# HW2 - Classification

Implementation of classification algorithms including Decision Trees, Neural Networks, and Convolutional Neural Networks using TensorFlow. The project covers theoretical concepts, manual decision tree construction, and deep learning for image classification.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](#)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue.svg)](#)

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Overview](#-overview)
- [Assignment Components](#-assignment-components)
  - [Theoretical Questions](#theoretical-questions)
  - [Part 1: Neural Networks with TensorFlow Playground](#part-1-neural-networks-with-tensorflow-playground)
  - [Part 2: Fashion MNIST Classification](#part-2-fashion-mnist-classification)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Concepts Demonstrated](#-key-concepts-demonstrated)
- [Results & Performance](#-results--performance)
- [Learning Outcomes](#-learning-outcomes)
- [Course Information](#-course-information)
- [Contact](#-contact)

</details>

## 📋 Overview

This project explores classification algorithms through both theoretical analysis and practical implementation. Students learn to:
- Build decision trees manually using information theory concepts
- Understand classification vs regression problems
- Implement neural networks with various architectures
- Work with real-world image datasets (Fashion MNIST)
- Evaluate models using multiple metrics

**Key Topics Covered:**
- ✅ **Classification vs Regression**: Understanding the fundamental differences
- ✅ **Decision Trees**: Manual construction using entropy and GINI index
- ✅ **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- ✅ **Neural Networks**: Architecture design and activation functions
- ✅ **Overfitting**: Detection and prevention techniques
- ✅ **Deep Learning**: CNN implementation for image classification

## 🎯 Assignment Components

### Theoretical Questions

#### 1. Classification vs Regression
**Task**: Provide 5 real-world examples for each category
- **Classification Examples**: Email spam detection, disease diagnosis, sentiment analysis, image recognition, credit approval
- **Regression Examples**: House price prediction, stock price forecasting, temperature prediction, sales forecasting, age estimation

#### 2. Evaluation Metrics (Part 1)
**Definitions with Formulas:**

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Measures overall correctness of predictions.

**Precision**:
```
Precision = TP / (TP + FP)
```
Measures accuracy of positive predictions.

**Recall** (Sensitivity/True Positive Rate):
```
Recall = TP / (TP + FN)
```
Measures ability to find all positive instances.

**F1-Score**:
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of Precision and Recall.

Where:
- **TP** (True Positive): Correctly predicted positive
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Incorrectly predicted positive
- **FN** (False Negative): Incorrectly predicted negative

#### 2. Evaluation Metrics (Part 2)
**Why Not Just Use Accuracy?**

**Scenario**: Network intrusion detection system
- Dataset: 10,000 packets, 9,900 normal, 100 malicious
- Model A: Detects 90% of malicious packets, Accuracy = 98.9%
- Model B: Detects 10% of malicious packets, Accuracy = 97.1%
- Naive Model: Predicts everything as "normal", Accuracy = 99%

**Analysis**:
- Accuracy alone is misleading for **imbalanced datasets**
- The naive model has highest accuracy but is useless
- Model A is better despite lower accuracy (higher Recall)
- **Recall is critical** when missing positives is costly
- **Precision matters** when false alarms have consequences

**Key Insight**: Different metrics serve different purposes based on:
- Class distribution (balanced vs imbalanced)
- Cost of false positives vs false negatives
- Application requirements

#### 3. Decision Tree Construction - Medical Diagnosis
**Dataset**:
| Age | Exercise | Cholesterol | Heart Disease |
|-----|----------|-------------|---------------|
| Old | Yes | Yes | Yes |
| Old | Yes | Yes | Yes |
| Old | No | Yes | Yes |
| Old | No | No | Yes |

**Task**: Build decision tree step-by-step showing:
1. Entropy calculations
2. Information Gain for each attribute
3. Best split selection
4. Tree structure

#### 4. Decision Tree Construction - Preferences
**Dataset**:
| Hat Lover | Sunglasses Lover | Age | Likes Red Hat |
|-----------|------------------|-----|---------------|
| Yes | Yes | 7 | No |
| Yes | No | 12 | No |
| No | Yes | 18 | Yes |
| No | Yes | 35 | Yes |
| Yes | Yes | 38 | Yes |
| Yes | No | 50 | No |
| No | No | 83 | No |

**Task**: Construct decision tree demonstrating understanding of:
- Continuous vs categorical attributes
- Split point selection for Age
- Information gain calculations

#### 5. GINI Index
**Questions**:
1. What is GINI index and its purpose?
2. What does high vs low GINI indicate?
3. Are there alternative splitting criteria?
4. Calculate GINI for example nodes

**GINI Index Formula**:
```
GINI = 1 - Σ(pi²)
```
where pi is the probability of class i

**Properties**:
- Range: [0, 0.5] for binary classification
- Lower GINI = More pure node
- Used in CART (Classification and Regression Trees)

#### 6. Overfitting
**Definition**: Model learns training data too well, including noise, reducing generalization to new data.

**Prevention Techniques**:
1. **Cross-validation**: K-fold validation for better estimates
2. **Regularization**: L1/L2 penalties on weights
3. **Early stopping**: Stop training when validation error increases
4. **Dropout**: Randomly deactivate neurons during training
5. **Data augmentation**: Increase training data diversity
6. **Pruning**: Remove unnecessary tree branches/neurons

### Part 1: Neural Networks with TensorFlow Playground

**Objective**: Understand neural network behavior through interactive experimentation

**Tasks**:

#### 1.1 Linear Model (No Activation Functions)
- Build network without activation functions
- **Question**: Can it classify circular data patterns?
- **Expected Result**: No - linear models cannot learn non-linear patterns

#### 1.2 Non-linear Model (Linear Activation)
- Add linear activation function
- **Question**: Does classification improve?
- **Expected Result**: Minimal improvement - linear activation doesn't add non-linearity

#### 1.3 Regression Loss for Classification
- Use regression loss (MSE) instead of classification loss
- **Question**: Does it work properly?
- **Expected Result**: Poor performance - wrong loss for binary classification

#### 1.4 Single Hidden Layer
- Create network with 1 hidden layer
- **Question**: Can it now classify the data?
- **Expected Result**: Yes - with non-linear activation, can learn patterns

#### 1.5 Learning Rate Tuning
- Experiment with different learning rates
- Document: Why certain values work better
- Test: Very low (0.001), low (0.01), medium (0.1), high (0.5), very high (1.0)

**Observations**:
- **Too low**: Slow convergence, may not converge
- **Too high**: Overshooting, instability, divergence
- **Optimal**: Balances speed and stability

#### 1.6 Custom Architecture
- Design best architecture for the problem
- Explain: Architecture choices and hyperparameters
- Include: Number of layers, neurons per layer, activation functions

**Requirements**:
- Screenshot results for each experiment
- Plot accuracy and loss curves
- Explain observations

### Part 2: Fashion MNIST Classification

**Dataset**: Fashion MNIST
- **Size**: 60,000 training + 10,000 test images
- **Image**: 28×28 grayscale pixels
- **Classes**: 10 clothing categories
  - T-shirt/top, Trouser, Pullover, Dress, Coat
  - Sandal, Shirt, Sneaker, Bag, Ankle boot

**Implementation Details**:

#### Data Loading & Preprocessing
```python
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

#### CNN Architecture
```python
model = Sequential([
    # First Convolutional Block
    Conv2D(32, kernel_size=3, activation='relu', input_shape=[28, 28, 1]),
    MaxPool2D(pool_size=2, strides=2),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),
    Dropout(0.25),
    
    # Fully Connected Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')  # 10 classes
])
```

**Architecture Explanation**:
- **Conv2D Layers**: Extract spatial features
  - 32 filters: Learn 32 different patterns
  - 3×3 kernel: Small receptive field
  - ReLU activation: Non-linearity
  
- **MaxPool2D**: Downsample feature maps
  - 2×2 pool size: Reduce dimensions by 50%
  - Reduces overfitting and computation
  
- **Dropout(0.25)**: Regularization
  - Randomly drops 25% of connections
  - Prevents overfitting
  
- **Dense Layers**: Classification
  - 128 neurons: Feature combination
  - 10 outputs: One per class
  - Softmax: Probability distribution

#### Training Configuration
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=25,
    verbose=1
)
```

**Hyperparameters**:
- **Optimizer**: Adam (adaptive learning rate)
- **Loss**: Categorical crossentropy (multi-class)
- **Batch size**: 32 samples per update
- **Epochs**: 25 complete passes through data

#### Evaluation & Results
```python
# Predictions
y_pred = model.predict(x_test)

# Accuracy calculation
correct = sum(y_pred.argmax(axis=1) == y_test.argmax(axis=1))
accuracy = correct / len(y_test)

# Confusion Matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
```

**Task Requirements**:
1. Train CNN model
2. Plot training loss and accuracy curves
3. Generate confusion matrix
4. Calculate test accuracy
5. Analyze misclassifications

## 🗂️ Project Structure

```
HW2 - Classification/
├── src/
│   ├── Programming_part1.ipynb    # TensorFlow Playground experiments
│   └── Programming_part2.ipynb    # Fashion MNIST CNN
├── doc/
│   ├── Project2.pdf               # Assignment instructions (Persian)
│   └── Report2_Amirmehdi Zarrinnezhad.pdf  # Implementation report
└── README.md                      # This file
```

## 📦 Installation

### Prerequisites
- Python 3.x
- Jupyter Notebook
- CUDA-compatible GPU (optional, for faster training)

### Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Or using requirements:
```bash
pip install -r requirements.txt
```

**Dependencies**:
```
tensorflow>=2.4.0
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## 🚀 Usage

### Running Part 1 (TensorFlow Playground)
```bash
cd src
jupyter notebook Programming_part1.ipynb
```

Follow the instructions in the notebook to:
1. Experiment with different architectures
2. Document observations
3. Save screenshots of results

### Running Part 2 (Fashion MNIST)
```bash
cd src
jupyter notebook Programming_part2.ipynb
```

The notebook will:
1. Load Fashion MNIST dataset
2. Build and train CNN model
3. Generate evaluation plots
4. Calculate performance metrics

**Expected Runtime**:
- With GPU: ~5-10 minutes
- Without GPU: ~20-30 minutes

## 🎓 Key Concepts Demonstrated

### Classification Fundamentals
- Binary vs multi-class classification
- Evaluation metrics selection
- Class imbalance handling

### Decision Trees
- Information gain calculation
- Entropy-based splitting
- GINI index usage
- Tree pruning concepts

### Neural Networks
- Activation functions (ReLU, Sigmoid, Softmax)
- Backpropagation
- Gradient descent optimization
- Learning rate impact

### Deep Learning
- Convolutional layers
- Pooling operations
- Dropout regularization
- Batch normalization concepts

### Model Evaluation
- Confusion matrix interpretation
- Precision-recall tradeoff
- F1-score calculation
- Cross-validation

## 📈 Results & Performance

### Part 2: Fashion MNIST Results

**Model Performance**:
- **Training Accuracy**: ~88.4% (after 25 epochs)
- **Test Accuracy**: ~89.4%
- **Training Loss**: ~0.31 (final)

**Training Progression**:
```
Epoch 1/25:  loss: 1.0542 - accuracy: 0.7093
Epoch 5/25:  loss: 0.3956 - accuracy: 0.8546
Epoch 10/25: loss: 0.3438 - accuracy: 0.8719
Epoch 15/25: loss: 0.3218 - accuracy: 0.8799
Epoch 20/25: loss: 0.3126 - accuracy: 0.8838
Epoch 25/25: loss: 0.3116 - accuracy: 0.8843
```

**Confusion Matrix Analysis**:
- **Best Classified**: Trouser (96.4%), Bag (97.6%), Ankle Boot (96.6%)
- **Most Confused**: Shirt ↔ T-shirt/top, Coat ↔ Pullover
- **Overall**: Good separation between most classes

**Key Observations**:
- Model converges steadily without major overfitting
- Dropout effectively prevents overfitting
- Similar-looking items (shirts vs t-shirts) harder to distinguish
- Distinct items (bags, shoes) classified with high accuracy

## ⚠️ Common Issues & Solutions

### 1. Training Too Slow
**Solution**: 
- Reduce batch size
- Use GPU acceleration
- Decrease number of epochs for testing

### 2. Overfitting
**Symptoms**: Training accuracy >> Test accuracy
**Solutions**:
- Increase dropout rate
- Add more data augmentation
- Reduce model complexity
- Use early stopping

### 3. Underfitting
**Symptoms**: Low training and test accuracy
**Solutions**:
- Increase model complexity
- Train for more epochs
- Reduce dropout
- Check learning rate

### 4. Memory Errors
**Solutions**:
- Reduce batch size
- Use smaller model
- Clear session between runs

## 🔮 Extensions & Improvements

### Potential Enhancements:
- [ ] Data augmentation (rotation, flip, zoom)
- [ ] Transfer learning with pre-trained models
- [ ] Ensemble methods
- [ ] Hyperparameter tuning (Grid/Random search)
- [ ] Batch normalization
- [ ] Different architectures (ResNet, VGG)
- [ ] Learning rate scheduling
- [ ] Class weights for imbalanced data

## 🎯 Learning Outcomes

After completing this assignment, students can:

✅ Distinguish between classification and regression problems
✅ Build decision trees manually using information theory
✅ Calculate and interpret evaluation metrics appropriately
✅ Design neural network architectures for specific tasks
✅ Implement CNNs for image classification
✅ Identify and prevent overfitting
✅ Tune hyperparameters effectively
✅ Evaluate model performance with multiple metrics
✅ Interpret confusion matrices
✅ Use TensorFlow/Keras for deep learning

## ℹ️ Course Information

**Author**: Amirmehdi Zarrinnezhad  
**Assignment**: Homework 2 - Classification  
**Course**: Data Mining  
**University**: Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021  
**GitHub Link:** [Classification](https://github.com/zamirmehdi/Data-Mining-Course/new/main/HW2%20-%20Classification)

**Part of Data Mining Course Projects**

[HW1: Preprocessing](../HW1%20-%20Data%20Preprocessing) | [HW2: Classification](.) | [HW3: Clustering & Association Rules](../HW3%20-%20Clustering,%20Association%20rules) | [Final: XGBoost](../Final%20Project%20-%20XGBoost)


## 📚 Theoretical Background

### Topics Covered:
1. **Classification vs Regression**: Problem types and examples
2. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
3. **Decision Trees**: ID3, C4.5, CART algorithms
4. **Entropy & Information Gain**: Tree splitting criteria
5. **GINI Index**: Alternative splitting criterion
6. **Overfitting**: Causes, detection, and prevention
7. **Neural Networks**: Architecture, activation functions, backpropagation
8. **CNNs**: Convolution, pooling, feature extraction
9. **Optimization**: Gradient descent, Adam optimizer
10. **Regularization**: Dropout, L1/L2, early stopping

## 📧 Contact

Questions or collaborations? Feel free to reach out!    
**Email**: amzarrinnezhad@gmail.com  
**GitHub**: [@zamirmehdi](https://github.com/zamirmehdi)

---

<div align="center">

[⬆ Back to Main Repository](https://github.com/zamirmehdi/Data-Mining-Course)

</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>
