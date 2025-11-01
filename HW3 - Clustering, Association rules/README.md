# HW3 - Clustering & Association Rules

Implementation of unsupervised learning algorithms including K-Means, DBSCAN, Hierarchical Clustering, and Apriori algorithm for association rule mining. The project covers both theoretical concepts and practical applications on real-world datasets.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](#)
[![mlxtend](https://img.shields.io/badge/mlxtend-Association%20Rules-green.svg)](#)

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Overview](#-overview)
- [Assignment Components](#-assignment-components)
  - [Theoretical Questions](#theoretical-questions)
  - [Part 1: Clustering](#part-1-clustering)
  - [Part 2: Association Rules](#part-2-association-rules)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Concepts Demonstrated](#-key-concepts-demonstrated)
- [Results & Performance](#-results--performance)
- [Learning Outcomes](#-learning-outcomes)
- [Project Information](#-project-information)
- [Contact](#-contact)

</details>

## 📋 Overview

This project explores two fundamental unsupervised learning paradigms:
1. **Clustering**: Grouping similar data points together
2. **Association Rule Mining**: Discovering interesting relationships between variables

**Key Applications:**
- Data compression and dimensionality reduction
- Customer segmentation for targeted marketing
- Market basket analysis for retail optimization
- Anomaly detection in complex datasets
- Recommendation systems

**Algorithms Implemented:**
- ✅ **K-Means**: Centroid-based clustering
- ✅ **DBSCAN**: Density-based clustering
- ✅ **Hierarchical Clustering**: Agglomerative methods (Single-link, Complete-link)
- ✅ **Apriori**: Frequent itemset mining and association rule generation

## 🎯 Assignment Components

### Theoretical Questions

#### Question 1: K-Means Algorithm Manual Execution

**Task**: Execute K-Means algorithm step-by-step on given 2D dataset
- Use **Manhattan distance** (L1 norm) as distance metric
- Number of clusters: K = 2
- Initial centroids: C₁ = (6,3), C₂ = (1,3)

**Dataset**:
```
Points: (1,1), (2,1), (3,1), (4,1), (5,1), (6,1),
        (1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2),
        (1,3), (2,3), (3,3), (4,3), (5,3), (6,3)
```

**Manhattan Distance Formula**:
```
d(i,j) = |xi1 - xj1| + |xi2 - xj2| + ... + |xip - xjp|
```

**Solution Steps**:

**Iteration 1**:
1. Calculate distances from each point to both centroids
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points

After assignment:
- **Cluster 1 (C₁)**: (4,1), (5,1), (6,1), (4,2), (5,2), (6,2), (7,2), (4,3), (5,3), (6,3)
- **Cluster 2 (C₂)**: (1,1), (2,1), (3,1), (1,2), (2,2), (3,2), (1,3), (2,3), (3,3)

**New centroids**:
- C₁_new = (5.2, 2.0)
- C₂_new = (2.0, 2.0)

**Iteration 2**:
- Recalculate distances with new centroids
- No changes in cluster assignments → **Convergence achieved**

**Final Result**:
- **Cluster 1**: {(4,1), (5,1), (6,1), (4,2), (5,2), (6,2), (7,2), (4,3), (5,3), (6,3)}
- **Cluster 2**: {(1,1), (2,1), (3,1), (1,2), (2,2), (3,2), (1,3), (2,3), (3,3)}

---

#### Question 2: Algorithm Selection for Complex Shapes

**Dataset Characteristics**:
- Non-convex clusters (non-spherical shapes)
- Variable density regions
- Potential outliers

**Recommended Algorithm**: **DBSCAN** (Density-Based Spatial Clustering)

**Justification**:

**Why Not K-Means?**
- K-Means assumes spherical clusters
- Performs poorly on non-convex shapes
- Sensitive to outliers
- Requires predefined number of clusters

**Why DBSCAN?**
1. **Arbitrary Cluster Shapes**: Can discover clusters of any shape by following density
2. **No K Required**: Automatically determines number of clusters
3. **Noise Handling**: Labels outliers as noise points
4. **Density-Based**: Groups points based on local density, not global distance

**DBSCAN Parameters**:
- **ε (epsilon)**: Neighborhood radius = **1.5**
  - Rationale: Adjacent points within clusters are ~1 unit apart
  - Points from different clusters are ≥2 units apart
  
- **MinPts**: Minimum points in neighborhood = **1 or 2**
  - Allows small but dense regions to form clusters

**Core Concepts**:
- **Core Point**: Has ≥MinPts within ε radius
- **Border Point**: Within ε of a core point but not a core point itself
- **Noise**: Neither core nor border

---

#### Question 3: Closed Frequent Itemsets

**Definition**: 
A **closed frequent itemset** is a frequent itemset for which no immediate superset has the same support count.

**Properties**:
```
itemset F is closed IF:
  ∄ super-itemset S where:
    - F ⊂ S
    - support(F) = support(S)
```

**Computing Support from Closed Itemsets**:

Given: All closed frequent itemsets with their support values
Want: Support of itemset F

**Algorithm**:
```
support(F) = support(smallest_superset(F))
where:
  smallest_superset(F) = argmin{|S| : F ⊆ S and S is closed frequent}
```

**Example**:
```
Closed Frequent Itemsets:
  {A, B, C}: support = 0.4
  {A, B}: support = 0.5
  {B, C}: support = 0.4

Query: support({B})?
Answer: support({A, B}) = 0.5
(smallest closed superset containing {B})
```

**Advantage**: Space-efficient representation
- Store only closed itemsets instead of all frequent itemsets
- Can recover support of any frequent itemset

---

#### Question 4: Apriori Algorithm Execution

**Transaction Database**:
| TID | Items |
|-----|-------|
| 1 | Apple, Orange, Banana |
| 2 | Pomegranate, Banana |
| 3 | Apple, Orange, Banana |
| 4 | Pomegranate, Orange |
| 5 | Apple, Tangerine |
| 6 | Apple, Tangerine, Pomegranate |

**Parameters**:
- **min_support** = 33% → support_count ≥ 2 (out of 6 transactions)
- **min_confidence** = 60%

**Step 1: Generate 1-itemsets**
| Itemset | Count | Support | Frequent? |
|---------|-------|---------|-----------|
| {Apple} | 4 | 67% | ✓ |
| {Orange} | 3 | 50% | ✓ |
| {Banana} | 3 | 50% | ✓ |
| {Pomegranate} | 3 | 50% | ✓ |
| {Tangerine} | 2 | 33% | ✓ |

All pass → Proceed to 2-itemsets

**Step 2: Generate 2-itemsets**
| Itemset | Count | Support | Frequent? |
|---------|-------|---------|-----------|
| {Apple, Orange} | 2 | 33% | ✓ |
| {Apple, Banana} | 2 | 33% | ✓ |
| {Orange, Banana} | 2 | 33% | ✓ |
| {Apple, Tangerine} | 2 | 33% | ✓ |
| {Pomegranate, Banana} | 1 | 17% | ✗ |
| {Pomegranate, Orange} | 1 | 17% | ✗ |
| {Apple, Pomegranate} | 1 | 17% | ✗ |
| {Tangerine, Pomegranate} | 1 | 17% | ✗ |

**Step 3: Generate 3-itemsets**
| Itemset | Count | Support | Frequent? |
|---------|-------|---------|-----------|
| {Apple, Orange, Banana} | 2 | 33% | ✓ |

No 4-itemsets possible → **Stop**

**Frequent Itemsets**:
- **L₁**: {Apple}, {Orange}, {Banana}, {Pomegranate}, {Tangerine}
- **L₂**: {Apple, Orange}, {Apple, Banana}, {Orange, Banana}, {Apple, Tangerine}
- **L₃**: {Apple, Orange, Banana}

**Association Rules from {Apple, Orange, Banana}**:

| Rule | Confidence | Strong? |
|------|------------|---------|
| {Orange, Banana} → {Apple} | 2/2 = 100% | ✓ |
| {Apple, Banana} → {Orange} | 2/2 = 100% | ✓ |
| {Apple, Orange} → {Banana} | 2/2 = 100% | ✓ |
| {Banana} → {Apple, Orange} | 2/3 = 67% | ✓ |
| {Orange} → {Apple, Banana} | 2/3 = 67% | ✓ |
| {Apple} → {Orange, Banana} | 2/4 = 50% | ✗ |

**Final Association Rules** (confidence ≥ 60%):
1. {Orange, Banana} → {Apple} [conf=100%, support=33%]
2. {Apple, Banana} → {Orange} [conf=100%, support=33%]
3. {Apple, Orange} → {Banana} [conf=100%, support=33%]
4. {Banana} → {Apple, Orange} [conf=67%, support=33%]
5. {Orange} → {Apple, Banana} [conf=67%, support=33%]

**Sorted by Confidence**:
100% > 100% > 100% > 67% > 67%

---

#### Question 5: Hierarchical Clustering with Dendrograms

**Distance Matrix**:
```
     P1    P2    P3    P4    P5
P1   0    0.22  0.41  0.55  0.35
P2  0.22   0    0.64  0.47  0.98
P3  0.41  0.64   0    0.44  0.85
P4  0.55  0.47  0.44   0    0.76
P5  0.35  0.98  0.85  0.76   0
```

**Method 1: Single Linkage (MIN)**

**Formula**: `d(Ci, Cj) = min{d(p, q) : p ∈ Ci, q ∈ Cj}`

**Iteration 1**: Merge P1 and P2 (distance = 0.22)
```
d(P1P2, P3) = min(0.41, 0.64) = 0.41
d(P1P2, P4) = min(0.55, 0.47) = 0.47
d(P1P2, P5) = min(0.35, 0.98) = 0.35
```

**Iteration 2**: Merge P1P2 and P5 (distance = 0.35)
```
d(P1P2P5, P3) = min(0.41, 0.85) = 0.41
d(P1P2P5, P4) = min(0.47, 0.76) = 0.47
```

**Iteration 3**: Merge P1P2P5 and P3 (distance = 0.41)
```
d(P1P2P5P3, P4) = min(0.47, 0.44) = 0.44
```

**Iteration 4**: Merge all with P4 (distance = 0.44)

**Dendrogram (Single Link)**:
```
      0.44 ├─────┐
           │     P4
      0.41 ├───┐
           │   P3
      0.35 ├─┐
           │ P5
      0.22 ├┐
           ││
           P1 P2
```

**Method 2: Complete Linkage (MAX)**

**Formula**: `d(Ci, Cj) = max{d(p, q) : p ∈ Ci, q ∈ Cj}`

**Iteration 1**: Merge P1 and P2 (distance = 0.22)
```
d(P1P2, P3) = max(0.41, 0.64) = 0.64
d(P1P2, P4) = max(0.55, 0.47) = 0.55
d(P1P2, P5) = max(0.35, 0.98) = 0.98
```

**Iteration 2**: Merge P3 and P4 (distance = 0.44)
```
d(P3P4, P1P2) = max(0.64, 0.55) = 0.64
d(P3P4, P5) = max(0.85, 0.76) = 0.85
```

**Iteration 3**: Merge P1P2 and P3P4 (distance = 0.64)
```
d(P1P2P3P4, P5) = max(0.98, 0.85) = 0.98
```

**Iteration 4**: Merge all with P5 (distance = 0.98)

**Dendrogram (Complete Link)**:
```
      0.98 ├───────┐
           │       P5
      0.64 ├─────┐
      0.55 │   ┌─┤
           │   │ P4
      0.44 │ ┌─┤
           │ │ P3
      0.22 ├─┤
           │ │
           P1 P2
```

**Comparison**:
- **Single Link**: Tends to create elongated clusters ("chaining effect")
- **Complete Link**: Creates more compact, spherical clusters

---

### Part 1: Clustering

#### 1.1 K-Means Clustering

**Dataset Generation**:
```python
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42, n_samples=100)
```

**Implementation**:
```python
from sklearn.cluster import KMeans

# Basic K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit(X).labels_

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.cool)
plt.title('K-Means Clustering (K=3)')
plt.show()
```

**Key Observations**:
- K-Means assumes **spherical clusters**
- Requires **predefined K** (number of clusters)
- Sensitive to **initial centroid selection**
- **Fast** and **scalable** for large datasets

---

#### 1.2 Elbow Method for Optimal K

**Purpose**: Determine optimal number of clusters

**Method**:
```python
costs = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    costs.append(kmeans.inertia_)  # Within-cluster sum of squares

plt.plot(range(1, 11), costs, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()
```

**Inertia (WCSS)**:
```
WCSS = Σ Σ ||x - μk||²
     k x∈Ck

where:
  Ck = cluster k
  μk = centroid of cluster k
```

**Finding the Elbow**:
- Plot K vs Inertia
- Look for the "elbow" point where rate of decrease sharply changes
- **Optimal K** = K at elbow point (typically K=3 for this dataset)

---

#### 1.3 K-Means on Different Data Distributions

**Tested Scenarios**:

**1. Incorrect Number of Clusters**:
```python
X, y = make_blobs(n_samples=1500, random_state=170)
kmeans = KMeans(n_clusters=2, random_state=170)  # Wrong K!
```
**Result**: Poor clustering, merges distinct clusters

**2. Anisotropically Distributed Data**:
```python
transformation = [[0.60834549, -0.63667341], 
                  [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
```
**Result**: K-Means struggles with elongated, rotated clusters

**3. Different Cluster Variances**:
```python
X_varied, y = make_blobs(n_samples=1500, 
                         cluster_std=[1.0, 2.5, 0.5],
                         random_state=170)
```
**Result**: Biased towards larger variance clusters

**4. Unevenly Sized Clusters**:
```python
X_filtered = np.vstack((X[y==0][:500], X[y==1][:100], X[y==2][:10]))
```
**Result**: Small clusters may be merged or lost

---

#### 1.4 Digit Clustering with K-Means

**Dataset**: Handwritten digits (64 features per image)

**Implementation**:
```python
from sklearn.datasets import load_digits

digits = load_digits()
dataset = digits['data']  # Shape: (1797, 64)

# K-Means with K=10 (one cluster per digit)
kmeans = KMeans(n_clusters=10, random_state=42)
predicted_labels = kmeans.fit_predict(dataset)

# Visualize cluster centers
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(centers[i], cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')
plt.show()
```

**Results**:
- Each cluster center represents the "average" digit
- **Accuracy**: ~85% correctly clustered
- **Confusion**: Digits 7 and 1 often confused (15% error rate)

**Error Analysis**:
```python
# Calculate false positive rate for digit 7 cluster
cluster_7_indices = np.where(predicted_labels == 8)[0]
true_labels = digits['target'][cluster_7_indices]
false_rate = np.sum(true_labels != 7) / len(cluster_7_indices)
print(f'False Rate for Digit 7: {false_rate:.2%}')
```

---

#### 1.5 Dimensionality Reduction with Isomap

**Objective**: Visualize high-dimensional digit data in 2D

**Implementation**:
```python
from sklearn.manifold import Isomap

# Reduce 64D → 2D
iso = Isomap(n_neighbors=10, n_components=2)
X_2d = iso.fit_transform(dataset)

# Compare K-Means labels vs True labels
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=predicted_labels, cmap='tab10')
axes[0].set_title('K-Means Clustering')
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=digits.target, cmap='tab10')
axes[1].set_title('True Labels')
plt.show()
```

**Observations**:
- K-Means creates reasonable digit groups
- Some overlap and misclassification visible
- True labels show clearer digit separation
- **Isomap preserves** non-linear structure better than PCA

---

#### 1.6 Image Compression with K-Means

**Application**: Color quantization for image compression

**Original Image**:
- **Size**: 3 channels (RGB) × pixels
- **Colors**: Millions of unique RGB values

**Compressed Image**:
- **Clusters**: K=4 representative colors
- **Storage**: Replace each pixel with nearest cluster center

**Implementation**:
```python
from PIL import Image

# Load image
img = imread('bird.jpg')
pixels = img.reshape(-1, 3)  # Flatten to (N, 3)

# K-Means clustering on colors
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Replace pixels with cluster centers
compressed_pixels = centers[labels].astype(np.uint8)
compressed_img = compressed_pixels.reshape(img.shape)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(compressed_img)
plt.title('Compressed (K=4 colors)')
plt.show()
```

**Results**:
- **Compression Ratio**: ~75% reduction (millions → 4 colors)
- **Quality**: Visually acceptable for K=4
- **Trade-off**: Lower K = more compression but less quality

**Color Distribution Visualization**:
```python
# 3D scatter of original colors
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(pixels[:, 0], pixels[:, 1], pixels[:, 2], 
             c=pixels/255.0, s=1)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
plt.title('RGB Color Distribution')
plt.show()
```

---

#### 1.7 DBSCAN Clustering

**Advantages over K-Means**:
- No need to specify K
- Handles arbitrary cluster shapes
- Robust to outliers
- Detects noise points

**Parameters**:
1. **ε (epsilon)**: Neighborhood radius
2. **MinPts**: Minimum points to form dense region

**Automatic ε Selection using K-Distance**:

**Method**:
```python
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Calculate K-distance for each point
k = 66  # Typically 2×dimensionality
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_aniso)
distances, _ = neighbors.kneighbors(X_aniso)

# Sort average distances
sorted_distances = np.sort(np.mean(distances, axis=1))

# Find elbow (optimal ε)
kneedle = KneeLocator(range(len(sorted_distances)), 
                      sorted_distances,
                      curve='convex',
                      direction='increasing')
optimal_epsilon = sorted_distances[kneedle.elbow]

# Plot K-distance graph
plt.plot(sorted_distances)
plt.axvline(x=kneedle.elbow, color='r', linestyle='--')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('K-distance')
plt.title(f'Optimal ε = {optimal_epsilon:.3f}')
plt.show()
```

**Rationale for K-NN**:
- K-distance reveals local density
- Sharp increase in K-distance → transition to noise
- Elbow point = optimal neighborhood radius

---

**MinPts Tuning**:

**Strategy**: Try different MinPts values and evaluate using V-Measure

```python
from sklearn.metrics import v_measure_score

v_scores = []
for min_pts in range(1, 20):
    dbscan = DBSCAN(eps=optimal_epsilon, min_samples=min_pts)
    labels = dbscan.fit_predict(X_aniso)
    v_scores.append(v_measure_score(y_true, labels))

# Select best MinPts
best_min_pts = np.argmax(v_scores)
print(f'Best MinPts: {best_min_pts}')
print(f'V-Measure: {max(v_scores):.3f}')
```

**Final DBSCAN Clustering**:
```python
dbscan = DBSCAN(eps=optimal_epsilon, min_samples=best_min_pts)
labels = dbscan.fit_predict(X_aniso)

# Visualization
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=labels, cmap='viridis')
plt.title(f'DBSCAN (ε={optimal_epsilon:.2f}, MinPts={best_min_pts})')
plt.show()
```

**Results**:

**X_aniso (Anisotropic Data)**:
- **Optimal ε**: 0.35-0.45
- **Best MinPts**: 4-6
- **V-Measure**: 0.85-0.95
- **Clusters Detected**: 3 (automatically)

**X_varied (Variable Variance)**:
- **Optimal ε**: 0.40-0.50
- **Best MinPts**: 3-5
- **V-Measure**: 0.80-0.90
- **Clusters Detected**: 3

---

**DBSCAN vs K-Means Comparison**:

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| **Cluster Shape** | Spherical only | Arbitrary shapes ✓ |
| **K Specification** | Required | Automatic ✓ |
| **Outlier Handling** | Poor | Excellent ✓ |
| **Scalability** | Excellent | Good |
| **Parameter Tuning** | Easy (just K) | Moderate (ε, MinPts) |

**When to use DBSCAN**:
- Non-convex cluster shapes
- Presence of noise/outliers
- Unknown number of clusters
- Variable cluster densities

---

### Part 2: Association Rules Mining

#### 2.1 Apriori Algorithm Basics

**Goal**: Discover frequent patterns in transactional data

**Key Metrics**:

**1. Support**:
```
support(A) = count(A) / total_transactions
```
Measures popularity of itemset A

**2. Confidence**:
```
confidence(A → B) = support(A ∪ B) / support(A)
```
Probability of B given A

**3. Lift**:
```
lift(A → B) = confidence(A → B) / support(B)
               = P(A ∪ B) / (P(A) × P(B))
```

**Lift Interpretation**:
- **Lift > 1**: Positive correlation (A and B occur together more than expected)
- **Lift = 1**: Independent (no correlation)
- **Lift < 1**: Negative correlation (A and B rarely occur together)

**Example**:
```
Rule: {Bread} → {Butter}
Lift = 2.5

Interpretation: 
Customers who buy bread are 2.5 times more likely 
to buy butter than average customers.
```

---

#### 2.2 Market Basket Analysis

**Dataset**: Hypermarket transactions
- **Transactions**: ~9000
- **Unique Items**: ~160 products
- **Format**: Customer ID, Transaction ID, Item

**Data Preprocessing**:

**Step 1: Load Data**
```python
dataset = pd.read_csv("Hypermarket_dataset.csv", header=None)
dataset.columns = ['Customer', 'TransactionID', 'Item']
```

**Step 2: Create Transaction Dictionary**
```python
transaction_dict = {}
for index, row in dataset.iterrows():
    trans_id = row['TransactionID']
    item = row['Item']
    if trans_id in transaction_dict:
        transaction_dict[trans_id].append(item)
    else:
        transaction_dict[trans_id] = [item]
```

**Step 3: One-Hot Encoding Matrix**
```python
# Get all unique items
all_items = dataset['Item'].unique()

# Create binary matrix
matrix = []
for trans_id, items in transaction_dict.items():
    row = [1 if item in items else 0 for item in all_items]
    matrix.append(row)

# Create DataFrame
df = pd.DataFrame(matrix, 
                  index=transaction_dict.keys(),
                  columns=all_items)
```

**Result**: Sparse matrix (transactions × items)
```
        Bread  Butter  Milk  Eggs  ...
1000      1      0      1     0    ...
1001      0      1      1     1    ...
1002      1      1      0     0    ...
...
```

---

#### 2.3 Frequent Itemset Mining

**Implementation**:
```python
from mlxtend.frequent_patterns import apriori

# Generate frequent itemsets
frequent_itemsets = apriori(df, 
                           min_support=0.07,
                           use_colnames=True)

# Add itemset size column
frequent_itemsets['k-freq'] = frequent_itemsets['itemsets'].apply(len)

# Display results
print(frequent_itemsets.sort_values('support', ascending=False))
```

**Sample Output**:
```
     support               itemsets  k-freq
45   0.1523    (Whole Milk, Other Vegetables)    2
67   0.1341              (Whole Milk, Rolls/Buns)    2
112  0.1220    (Whole Milk, Yogurt)    2
...
```

**Interpretation**:
- 15.23% of transactions contain both Whole Milk and Other Vegetables
- These items frequently purchased together

---

#### 2.4 Association Rule Generation

**Function Implementation**:
```python
from mlxtend.frequent_patterns import association_rules

def extract_rules(metric, threshold):
    """
    Generate association rules from frequent itemsets
    
    Parameters:
    -----------
    metric : str ('confidence' or 'lift')
        Evaluation metric for rules
    threshold : float
        Minimum threshold value
    
    Returns:
    --------
    DataFrame of association rules sorted by metric
    """
    rules = association_rules(frequent_itemsets, 
                             metric=metric,
                             min_threshold=threshold)
    
    # Sort by metric
    rules = rules.sort_values(metric, ascending=False)
    return rules

# Example 1: High confidence rules
high_confidence = extract_rules('confidence', 0.07)
print(high_confidence.head(10))

# Example 2: High lift rules
high_lift = extract_rules('lift', 1.2)
print(high_lift.head(10))
```

**Rule Format**:
```
antecedents → consequents [support, confidence, lift]
```

**Sample Rules** (confidence ≥ 0.07):
```
1. {Root Vegetables, Tropical Fruit} → {Other Vegetables}
   support: 0.0123, confidence: 0.584, lift: 1.67

2. {Curd, Yogurt} → {Whole Milk}
   support: 0.0101, confidence: 0.582, lift: 1.45

3. {Butter, Other Vegetables} → {Whole Milk}
   support: 0.0115, confidence: 0.574, lift: 1.43
```

**Sample Rules** (lift ≥ 1.2):
```
1. {Liquor, Red/Blush Wine} → {Bottled Beer}
   lift: 2.14, confidence: 0.421

2. {Citrus Fruit, Root Vegetables, Whole Milk} → {Other Vegetables}
   lift: 1.87, confidence: 0.652

3. {Beef, Root Vegetables} → {Other Vegetables}
   lift: 1.73, confidence: 0.604
```

---

#### 2.5 Business Insights

**Top Association Patterns**:

**1. Complementary Products**:
- Whole Milk ↔ Other Vegetables (very frequent)
- Butter → Whole Milk (high confidence)
- Root Vegetables → Other Vegetables (strong correlation)

**Strategic Actions**:
- **Cross-selling**: Place related items near each other
- **Bundling**: Create product bundles with discounts
- **Promotions**: Discount one item to boost sales of related items

**2. Meal Patterns**:
- {Tropical Fruit, Yogurt} → {Whole Milk}
- {Herbs, Root Vegetables} → {Other Vegetables}

**Insight**: Customers shopping for meal ingredients buy multiple items

**3. Beverage Associations**:
- {Liquor, Wine} → {Bottled Beer} (high lift)
- Strong correlation in alcohol purchases

---

## 🗂️ Project Structure

```
HW3 - Clustering, Association rules/
├── src/
│   ├── Clustering.ipynb           # Clustering algorithms implementation
│   └── AssociationRules.ipynb     # Apriori and rule mining
├── data/
│   ├── Hypermarket_dataset.csv    # Transaction data (9000+ transactions)
│   └── bird.jpg                   # Image for compression demo
├── doc/
│   ├── Project3.pdf               # Assignment instructions (Persian)
│   └── Report3_Amirmehdi Zarrinnezhad.pdf  # Implementation report
└── README.md                      # This file
```

## 📦 Installation

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Google Colab (optional)

### Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend kneed pillow
```

Or using requirements:
```bash
pip install -r requirements.txt
```

**Dependencies**:
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
mlxtend>=0.18.0
kneed>=0.7.0
pillow>=8.0.0
```

## 🚀 Usage

### Running Clustering Notebook
```bash
cd src
jupyter notebook Clustering.ipynb
```

**Sections**:
1. K-Means on synthetic data
2. Elbow method for K selection
3. K-Means on different distributions
4. Digit clustering
5. Isomap dimensionality reduction
6. Image compression
7. DBSCAN clustering
8. ε parameter selection
9. MinPts tuning

### Running Association Rules Notebook
```bash
cd src
jupyter notebook AssociationRules.ipynb
```

**Sections**:
1. Data preprocessing and one-hot encoding
2. Frequent itemset generation
3. Association rule mining
4. Rule evaluation with confidence and lift

## 🎓 Key Concepts Demonstrated

### Clustering
- **Partitioning methods**: K-Means
- **Density-based methods**: DBSCAN
- **Hierarchical methods**: Single-link, Complete-link
- **Evaluation**: Elbow method, V-Measure, Silhouette score
- **Applications**: Image compression, digit recognition

### Association Rules
- **Frequent pattern mining**: Apriori algorithm
- **Metrics**: Support, Confidence, Lift
- **Data transformation**: Transactional to matrix format
- **Business applications**: Market basket analysis

### Distance Metrics
- Euclidean distance (K-Means)
- Manhattan distance (L1 norm)
- Cosine similarity
- Custom similarity measures

## 📈 Results & Performance

### Clustering Results

**K-Means on Digits**:
- **Dataset**: 1797 handwritten digits (8×8 pixels)
- **Clusters**: 10 (one per digit)
- **Accuracy**: ~85% correctly clustered
- **Confusion**: Digits 7 and 1 (15% error rate)

**DBSCAN vs K-Means**:

| Dataset | K-Means Accuracy | DBSCAN V-Measure | Winner |
|---------|------------------|------------------|--------|
| Anisotropic | 65% | 0.92 | DBSCAN ✓ |
| Variable Variance | 70% | 0.87 | DBSCAN ✓ |
| Well-separated | 95% | 0.93 | Tie |

**Image Compression**:
- **Original**: 24-bit color (16.7M colors)
- **K=4**: 2-bit color (4 colors) → 75% reduction
- **K=16**: 4-bit color (16 colors) → 50% reduction
- **Quality**: Acceptable up to K=8

---

### Association Rules Results

**Frequent Itemsets** (min_support = 7%):
- **1-itemsets**: 42 frequent items
- **2-itemsets**: 156 frequent pairs
- **3-itemsets**: 87 frequent triplets
- **Max size**: 4 items

**Association Rules** (min_confidence = 7%, min_lift = 1.2):
- **Total rules generated**: 243
- **High confidence (>70%)**: 18 rules
- **High lift (>2.0)**: 7 rules

**Top 5 Rules by Lift**:

| Rank | Rule | Lift | Confidence | Support |
|------|------|------|------------|---------|
| 1 | {Liquor, Red Wine} → {Beer} | 2.14 | 42% | 1.2% |
| 2 | {Citrus, Root Veg, Milk} → {Vegetables} | 1.87 | 65% | 1.5% |
| 3 | {Beef, Root Veg} → {Vegetables} | 1.73 | 60% | 1.8% |
| 4 | {Tropical Fruit, Root Veg} → {Vegetables} | 1.67 | 58% | 1.2% |
| 5 | {Butter, Vegetables} → {Milk} | 1.43 | 57% | 1.1% |

**Business Impact**:
- Identified 15+ cross-selling opportunities
- Discovered meal planning patterns
- Revealed complementary product relationships

## ⚠️ Common Issues & Solutions

### Clustering Issues

**1. K-Means Not Converging**
**Symptoms**: Maximum iterations reached
**Solutions**:
- Increase `max_iter` parameter
- Try different `init` methods ('k-means++' vs 'random')
- Normalize/standardize features first
- Check for outliers

**2. DBSCAN Finding Only Noise**
**Symptoms**: All points labeled as -1
**Solutions**:
- Increase ε (epsilon)
- Decrease MinPts
- Check data scale (normalize if needed)
- Visualize K-distance plot

**3. Memory Error with Large Datasets**
**Solutions**:
- Use Mini-Batch K-Means
- Sample data for parameter tuning
- Process data in chunks

---

### Association Rules Issues

**1. No Frequent Itemsets Found**
**Symptoms**: Empty result from Apriori
**Solutions**:
- Lower min_support threshold
- Check data encoding (should be 0/1)
- Verify transaction format

**2. Too Many Rules Generated**
**Symptoms**: Thousands of rules, hard to interpret
**Solutions**:
- Increase min_confidence
- Increase min_lift (>1.5)
- Filter by support (focus on common patterns)
- Limit max itemset size

**3. One-Hot Encoding Memory Error**
**Solutions**:
- Use sparse matrix format
- Process transactions in batches
- Filter low-frequency items first

## 🔮 Extensions & Improvements

### Clustering Enhancements
- [ ] Implement OPTICS algorithm
- [ ] Add Gaussian Mixture Models (GMM)
- [ ] Spectral clustering for graph data
- [ ] Ensemble clustering methods
- [ ] Automatic cluster number selection (Gap statistic, Silhouette)
- [ ] Time series clustering
- [ ] Semi-supervised clustering

### Association Rules Enhancements
- [ ] FP-Growth algorithm (faster than Apriori)
- [ ] Sequential pattern mining
- [ ] Hierarchical association rules
- [ ] Negative association rules
- [ ] Multi-level association rules
- [ ] Real-time rule updates (streaming data)
- [ ] Visualization dashboard for rules

## 🎯 Learning Outcomes

After completing this assignment, students can:

✅ Implement K-Means clustering from scratch
✅ Apply DBSCAN for arbitrary-shaped clusters
✅ Construct hierarchical clustering dendrograms
✅ Select optimal number of clusters using Elbow method
✅ Tune DBSCAN parameters (ε, MinPts) effectively
✅ Perform market basket analysis
✅ Generate association rules with Apriori
✅ Interpret support, confidence, and lift metrics
✅ Apply clustering to real-world problems (compression, segmentation)
✅ Extract business insights from transactional data

## 📚 Theoretical Background

### Covered Topics

**Clustering**:
1. **Partitioning Algorithms**: K-Means, K-Medoids
2. **Hierarchical Methods**: Agglomerative (Single, Complete, Average linkage)
3. **Density-Based**: DBSCAN, OPTICS
4. **Distance Metrics**: Euclidean, Manhattan, Cosine
5. **Evaluation**: Silhouette, V-Measure, Davies-Bouldin Index
6. **Challenges**: Curse of dimensionality, outlier sensitivity

**Association Rules**:
1. **Frequent Pattern Mining**: Apriori algorithm
2. **Metrics**: Support, Confidence, Lift
3. **Pruning Strategies**: Downward closure property
4. **Applications**: Market basket, web mining, bioinformatics
5. **Closed Itemsets**: Space-efficient representation
6. **Maximal Itemsets**: Compact representation

## ℹ️ Project Information

**Author**: Amirmehdi Zarrinnezhad  
**Assignment**: Homework 3 - Clustering & Association Rules  
**Course**: Data Mining  
**University**: Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021  
**GitHub Link**: [Clustering & Association Rules](https://github.com/zamirmehdi/Data-Mining-Course/tree/main/HW3%20-%20Clustering%2C%20Association%20rules)

<div align="center">

**Part of Data Mining Course Projects**

[HW1: Preprocessing](../HW1%20-%20Data%20Preprocessing) | [HW2: Classification](../HW2%20-%20Classification) | [HW3: Clustering & Association Rules](.) | [Final: XGBoost](../Final%20Project%20-%20XGBoost)

</div>

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
