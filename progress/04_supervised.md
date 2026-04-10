# **Supervised Learning Methods — Neural Network and KNN**

## **Problem Framing and Validation Design**

The objective of our supervised learning approach is to predict user preferences for movies using both engineered features and learned representations from our Two Towers model. We consider two formulations of this problem. First, we model the prediction of movie ratings on a continuous 1–5 scale using a neural network. Second, we frame the problem as a binary classification task, where we predict whether a user will rate a movie above their average rating.

Our feature set incorporates several sources of information. Most importantly, we include the **Two Towers affinity score**, which captures latent similarity between users and movies. In addition, we use movie-level statistics such as a leave-one-out average rating, user-level genre preferences, and interaction terms between user preferences and movie genres. These features allow us to model both general trends and individualized behavior.

To evaluate our models, we split the dataset into training and validation sets using an 80/20 random split. Care was taken to avoid data leakage. In particular, movie-level averages were computed using a leave-one-out strategy, ensuring that the target rating was not included in its own feature calculation. Feature scaling was performed using parameters fit only on the training data and then applied to the validation set. While users appear in both training and validation sets, all predictions are made on unseen user–movie interactions, ensuring a valid evaluation of generalization performance.

---

## **Models Implemented**

To provide a meaningful comparison, we implemented both simple baseline models and a more expressive neural network model.

The first model is a **logistic regression classifier**, which serves as a linear baseline. This model predicts whether a movie will be rated above the user’s average rating using the full feature set, including the Two Towers score. Logistic regression is useful because it provides interpretable coefficients that allow us to assess the importance of each feature.

The second model is a **K-Nearest Neighbors (KNN) classifier**, which predicts outcomes based on similarity in feature space. For each observation, the model identifies the nearest neighbors and assigns a class based on majority vote. We tuned the number of neighbors \(k\) using cross-validation over a range of values from 5 to 20, selecting the value that maximized the F1 score. KNN provides a non-parametric baseline that captures local structure in the data but does not learn global patterns.

Our primary model is a **neural network with user and movie embeddings**, designed to predict ratings on a continuous scale. This model maps each user and movie to a 32-dimensional embedding and computes an interaction between them through element-wise multiplication. These embeddings are concatenated with the engineered numeric features, including the Two Towers score and genre-based interactions, and passed through a series of fully connected layers with ReLU activations and dropout regularization. The model also includes user and movie bias terms, as well as a global mean rating, allowing it to capture both individual tendencies and overall trends.

The neural network is trained using mean squared error (MSE) loss, optimized with the Adam optimizer. A learning rate scheduler is used to reduce the learning rate when validation performance plateaus, and early stopping is applied to prevent overfitting. This model is significantly more flexible than the baseline models, as it can capture nonlinear relationships and complex interactions between features.

---

## **Tuning, Metrics, and Results**

We evaluated our models using appropriate metrics for each task. For the classification models (logistic regression and KNN), we used the F1 score to balance precision and recall. For the neural network, we used root mean squared error (RMSE), which penalizes large prediction errors and is well-suited for continuous rating prediction.

We performed targeted hyperparameter tuning for both KNN and the neural network. For KNN, we conducted a grid search over values of \(k\) from 5 to 20 using 5-fold cross-validation. For the neural network, we explored different learning rates to assess their impact on training stability and performance. Specifically, we tested learning rates of 1e-3, 5e-4, and 1e-4. Among these, a learning rate of 5e-4 produced the best validation performance, achieving an RMSE of 1.018. Higher learning rates resulted in slightly worse performance due to instability during training, while lower learning rates slowed convergence without improving accuracy.

A summary of model performance is provided below:

| Model | Task | Key Hyperparameters | Metric | Score |
|------|------|--------------------|--------|------|
| Logistic Regression | Classification | Default settings | F1 | 0.684 |
| KNN | Classification | \(k\) tuned via CV | F1 | 0.679 |
| Neural Network | Regression | lr = 5e-4, embed_dim = 32, dropout = 0.2 | RMSE | 1.018 |
| Logistic Regression (No TT) | Classification | Default settings | F1 | 0.520 |

---

## **Model Comparison and Interpretability**

Across models, several clear trends emerge. Logistic regression and KNN achieve similar performance, with F1 scores of 0.684 and 0.679 respectively. This suggests that both linear and local similarity-based methods are capable of capturing meaningful structure in the data, but neither is substantially more powerful than the other.

The neural network, while evaluated using a different metric, provides a more flexible modeling approach and achieves strong performance in predicting continuous ratings. Its ability to incorporate embeddings allows it to capture latent relationships between users and movies that are not directly observable in the feature space.

To better understand feature importance, we examined the coefficients of the logistic regression model. This provides a clear and interpretable measure of how each feature contributes to the prediction. Notably, the Two Towers affinity score emerged as one of the most influential predictors. Genre interaction terms also played a significant role, indicating that user-specific preferences interact meaningfully with movie characteristics.

To further evaluate the importance of the Two Towers score, we trained a logistic regression model without this feature. Performance dropped substantially, with the F1 score decreasing from 0.684 to 0.520. This demonstrates that the affinity score captures critical information about user–movie relationships that is not present in traditional features alone.

---

## **Conclusion**

Our supervised modeling results demonstrate the value of combining engineered features with learned representations from the Two Towers system. The affinity score provides a strong signal that significantly improves predictive performance across models.

While simpler models such as logistic regression and KNN offer competitive baseline performance and useful interpretability, the neural network provides greater flexibility and the ability to model complex interactions. This makes it particularly well-suited for recommendation tasks where user preferences are highly individualized and nonlinear.

Overall, the integration of unsupervised and supervised methods allows us to better capture the underlying structure of user preferences, leading to more accurate and meaningful movie recommendations.