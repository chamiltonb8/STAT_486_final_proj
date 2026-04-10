# Supervised Learning Methods — Neural Network and KNN

## Problem Framing and Validation Design

The goal of this supervised learning task is to predict whether a user will **like a movie**, using both engineered features and learned representations from the Two Towers model. While movie ratings are originally given on a 1–5 scale, we reformulate the problem as a **binary classification task** to better align with recommendation objectives.

Specifically: 
- A movie is classified as **“liked”** if the true rating is **greater than 3**
- A movie is classified as **“not liked”** if the true rating is **less than or equal to 3**

All models are evaluated based on their ability to correctly classify user–movie interactions into these two categories. For models that output continuous predictions (such as the neural network), predictions are thresholded at 3 to convert them into binary outcomes.

### Feature Construction

The feature set combines both engineered and learned signals:

- **Two Towers affinity score** (primary feature capturing user–movie similarity)
- Leave-one-out movie average rating (prevents leakage)
- User-level genre preferences
- Interaction terms between user preferences and movie genres

This combination allows the model to capture both global trends and personalized preferences.

### Validation Strategy and Leakage Prevention

We use an **80/20 train/validation split**, where the validation set consists of unseen user–movie interactions. This ensures that model performance reflects generalization rather than memorization.

To prevent data leakage:
- Movie-level averages are computed using a **leave-one-out strategy**, ensuring that the target observation is not included in its own feature
- Feature scaling is fit **only on the training set** and then applied to validation data
- All model evaluation is performed strictly on held-out validation data

This workflow ensures a **reproducible and leakage-free evaluation pipeline**.

---

## Model Implementation Breadth

We implement three supervised models to provide a meaningful comparison between simple and more expressive approaches.

### Logistic Regression (Baseline Model)

Logistic regression serves as a **linear baseline classifier**. It predicts whether a movie will be liked using the full feature set. This model is valuable because:
- It is simple and interpretable
- Coefficients provide direct insight into feature importance

---

### K-Nearest Neighbors (KNN)

KNN is a **non-parametric model** that predicts outcomes based on similarity in feature space. For each observation:
- The model identifies the \(k\) nearest neighbors
- Assigns a class based on majority vote

We tune \(k\) using cross-validation, selecting the value that maximizes the F1 score. This model captures **local structure** but does not learn global relationships.

---

### Neural Network with Embeddings (Primary Model)

The neural network is a **high-capacity model** that combines:

- User embeddings (dimension = 32)
- Movie embeddings (dimension = 32)
- Engineered features (including Two Towers score)

The model:
- Computes an interaction between user and movie embeddings via element-wise multiplication
- Concatenates this with engineered features
- Passes the result through fully connected layers with ReLU activations and dropout

Although trained using **mean squared error (MSE)** on continuous ratings, predictions are converted into binary classifications using a threshold of 3. This allows direct comparison with other models using F1 score.

---

## Tuning, Metrics, and Reporting

### Evaluation Metric

All models are evaluated using the **F1 score**, which balances precision and recall.

A prediction is considered correct if:
- The model predicts **> 3** and the true rating is **> 3** (correctly identifies a liked movie), or
- The model predicts **≤ 3** and the true rating is **≤ 3** (correctly identifies a disliked movie)

This metric directly reflects recommendation quality.

---

### Hyperparameter Tuning

- **KNN:** \(k\) tuned over values from 5 to 20 using 5-fold cross-validation
- **Neural Network:**
  - Learning rates tested: 1e-3, 5e-4, 1e-4
  - Best performance achieved at **5e-4**
  - Embedding dimension: 32
  - Dropout: 0.2
  - Early stopping applied based on validation loss

---

### Model Performance Summary

| Model | Task | Key Hyperparameters | Validation Setup | Metric | Score |
|------|------|--------------------|------------------|--------|------|
| Logistic Regression | Classification | Default | 80/20 split | F1 | 0.684 |
| KNN | Classification | \(k\) tuned (5–20) | 80/20 + CV | F1 | 0.679 |
| Neural Network | Regression → Classification | lr = 5e-4, embed_dim = 32, dropout = 0.2 | 80/20 split | F1 | **0.8965** |
| Logistic Regression (No TT) | Classification | Default | 80/20 split | F1 | 0.520 |

---

## Model Comparison and Interpretability

### Model Comparison

Several key trends emerge:

- Logistic regression and KNN perform similarly (F1 ≈ 0.68), indicating that both linear and local methods capture some structure but are limited in flexibility.
- The neural network significantly outperforms both models (F1 = 0.8965), demonstrating its ability to capture **nonlinear and personalized interactions**.
- Removing the Two Towers score from logistic regression reduces performance from **0.684 to 0.520**, confirming its importance.

### Selected Model

The **neural network** is selected as the best-performing model due to its substantially higher F1 score and ability to model complex relationships.

---

### Interpretability

To interpret model behavior, we analyze **logistic regression coefficients**:

- The **Two Towers affinity score** is one of the most influential predictors
- Genre interaction terms are also significant, indicating personalized preferences

This provides clear evidence that:
- Learned embeddings capture meaningful latent relationships
- Engineered features alone are insufficient without the affinity signal

---

## Takeaways and Repository Evidence

### Key Findings

Supervised learning reveals that:

- The **Two Towers affinity score** is a critical predictor of user preference
- Simple models (logistic regression, KNN) provide reasonable baselines but are limited
- The neural network significantly improves performance by capturing nonlinear and latent relationships
- Framing the problem as a **binary classification task aligned with user satisfaction** leads to more meaningful evaluation

---

### Connection to Research Question

This analysis directly answers the core question:

> *Can we accurately predict which movies a user will like?*

Yes — and the results show that incorporating learned representations dramatically improves predictive performance.

---

### Repository Evidence

The code for the knn and logistic regression are found in `knn-and-logistic.ipynb` under the src/notebooks folders.
The code the neural network is found in `neural_networks.ipynb` under the src/notebooks folders