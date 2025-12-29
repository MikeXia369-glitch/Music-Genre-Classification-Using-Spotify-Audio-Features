# 🎵 Music Genre Classification Using Spotify Audio Features

**Author:** Mike  

This project develops machine learning models to classify music genres using Spotify audio features from approximately 50,000 songs across ten genres. The project emphasizes **reproducibility, careful data preprocessing, dimensionality reduction, model comparison, and interpretation of misclassification patterns**.

---

## 📥 Dataset and Reproducibility

- ~50,000 tracks  
- 10 genres  
- Spotify audio features

To ensure **reproducibility**, a fixed random seed (derived from my N-number) was used for:

- data processing  
- train–test split  
- model training  

---

## 🧹 Data Cleaning

Steps performed:

- removed duplicate rows  
- deleted tracks with invalid duration values (negative `duration_ms`)
- converted `tempo` from string to numeric  
- handled missing values introduced after conversion

Invalid and corrupted entries were removed before modeling.

---

## 🧾 Feature Selection and Processing

Removed **non-informative identifier-style fields**, including:

- artist name  
- track name  
- instance ID  
- obtained date  

Remaining features contained **numeric audio descriptors and categorical music attributes**.

---

## 🔧 Handling Missing Values

- Missing values (mostly in `tempo`) were imputed with **median imputation**
- Chosen because many audio features are **skewed** rather than normally distributed

---

## 🔢 Encoding Categorical Variables

- Musical **key** → one-hot encoding (`drop_first=True`)
- Musical **mode**:
  - Major → 1  
  - Minor → 0  

All features were transformed into numeric format suitable for modeling.

---

## ✂️ Train–Test Split

To avoid leakage and ensure fair evaluation:

- genre-stratified split  
- **500 songs per genre** included in test set  
- total test size = **5,000 songs**
- remaining songs used for training

---

## 📉 Feature Scaling and PCA

Pipeline included:

- median imputation  
- feature standardization  
- principal component analysis (PCA)

PCA retained **≈90% variance**, resulting in:

- **17 principal components**

A strong correlation was observed between **loudness and energy**, indicating redundancy addressed by PCA.

---

## 👁 Low-Dimensional Visualization & Clustering

- PCA 2D visualization used to inspect genre structure  
- K-Means clustering applied  
- elbow method suggested **k = 10–12**

### Key observations

- Some genres cluster well (e.g., **Classical**)
- Strong overlap between:

  - Hip-Hop & Rap  
  - Jazz & Blues  
  - Alternative & Rock  

👉 Indicates **fuzzy genre boundaries**, not model weakness

---

## 🤖 Models Trained

All models used **PCA-transformed features** and identical splits.

### Metrics

- macro-averaged ROC curves  
- macro AUROC  

### Models

| Model | AUROC |
|-------|-------|
| Decision Tree | 0.82 |
| Random Forest | 0.86 |
| XGBoost | 0.88 |
| Simple Neural Network (1-layer MLP) | 0.90 |
| **Multi-Layer Perceptron (final)** | **0.91** |

---

## 🧠 Model Descriptions

- **Decision Tree**
  - depth limited to 10
  - min samples/leaf = 50
  - baseline performance

- **Random Forest**
  - 100 trees
  - reduced variance
  - AUROC = 0.86

- **XGBoost**
  - gradient boosting
  - depth 10, min child weight 50
  - AUROC = 0.88

- **Simple neural network (1 hidden layer MLP)**
  - 100 hidden neurons
  - ReLU + Adam
  - AUROC = 0.90

- **Final MLP (3 hidden layers)**
  - 100 → 50 → 25 neurons
  - ReLU + Adam
  - early stopping enabled
  - **best AUROC = 0.91**

---

## 🏆 Final Model Selection

The **three-layer MLP** was selected because:

- highest AUROC (≈ 0.91)
- stable training behavior
- better capture of nonlinear structure

---

## 🗂 Confusion Matrix Insights (Extra Credit)

Non-trivial observations:

- Hip-Hop ↔ Rap commonly misclassified
- Jazz ↔ Blues frequently overlap
- Alternative ↔ Rock also confused

These errors are **musically intuitive**, suggesting:

- genre boundaries are fuzzy
- overlapping acoustic feature spaces
- classification difficulty is **data-inherent**, not model failure

---

## ⭐ Key Takeaway

> **Data preprocessing was the most important factor.**

Once issues like:

- missing value handling  
- scaling  
- encoding correctness  

were fixed, **all models improved significantly**.

Hyperparameter tuning helped, but **preprocessing determined the ceiling**.

---

## 🚀 Final Result
Final multi-layer perceptron (MLP):
AUROC = 0.91
Selected as final classification model.

