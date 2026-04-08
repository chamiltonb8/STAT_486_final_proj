import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MovieRecommenderEmb(nn.Module):
    def __init__(self, num_users, num_movies, embed_dim, input_dim, mean_rating, dropout=0.2):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_movies, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim + 3*embed_dim, 64),  # note +3*embed_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.movie_bias.weight, 0.0)
        self.mean_rating = mean_rating

    def forward(self, numeric_x, user_x, movie_x):
        user_vec = self.user_embed(user_x) / np.sqrt(self.user_embed.embedding_dim)
        movie_vec = self.movie_embed(movie_x) / np.sqrt(self.movie_embed.embedding_dim)
        interaction = user_vec * movie_vec   # element-wise product
        
        x = torch.cat([numeric_x, user_vec, movie_vec, interaction], dim=1)
        base = self.fc(x)
        bias = self.user_bias(user_x) + self.movie_bias(movie_x)
        return base + bias + self.mean_rating

def compute_user_popularity_weight(user_id, df, max_weight=1.0, min_weight=0.0):
    """
    Compute a user-specific popularity weight based on how many movies they have rated.

    - Sparse users → weight closer to max_weight (more popularity-based)
    - Active users → weight closer to min_weight (more model-based)
    """
    n_rated = len(df[df['userId'] == user_id])
    # Example: use a log-scaling for smoother transition
    weight = max_weight * np.exp(-0.1 * n_rated)
    weight = np.clip(weight, min_weight, max_weight)
    return weight

def predict_top_movies(
    user_id, df, movies_df, feature_cols, model, scaler, movie2idx,
    top_n=10, min_rating=3.5, random_seed=None,
    top_pool_factor=3, noise_std=0.02, popularity_weight=0.2
):
    import torch
    import numpy as np
    import pandas as pd

    # --- Set seeds ---
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    # --- Candidate movies (exclude already rated) ---
    rated_movies = df[df['userId'] == user_id]['movieId'].tolist()
    movie_candidates = df[~df['movieId'].isin(rated_movies)][['movieId','title']].drop_duplicates('movieId')

    if len(movie_candidates) == 0:
        return pd.DataFrame(columns=['title','pred_rating'])

    # --- Select features from movies_df by index mapping ---
    # Assuming movie2idx maps movieId -> row index in movies_df
    movie_indices = [movie2idx[m] for m in movie_candidates['movieId'].values]
    X_candidates = movies_df.iloc[movie_indices][feature_cols].values
    X_scaled = scaler.transform(X_candidates)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # --- Embedding indices ---
    movie_idx_tensor = torch.tensor([movie2idx[m] for m in movie_candidates['movieId'].values], dtype=torch.long)
    user_idx_tensor = torch.tensor([user_id] * len(movie_candidates), dtype=torch.long)

    # --- Predict ---
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor, user_idx_tensor, movie_idx_tensor).cpu().numpy().flatten()

    # --- Clip predictions ---
    preds = np.clip(preds, min_rating, 5)

    # --- Popularity boost ---
    popularity = df.groupby('movieId')['userId'].count()
    popularity = np.log1p(popularity)
    popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min() + 1e-8)
    popularity_scores = movie_candidates['movieId'].map(popularity).fillna(0)

    # --- Combine predictions + popularity + noise ---
    combined_scores = preds + popularity_weight * popularity_scores
    combined_scores = np.clip(combined_scores, min_rating, 5)
    combined_scores += np.random.normal(0, noise_std, size=combined_scores.shape)

    # --- Build top candidate pool ---
    pool_size = min(len(movie_candidates), top_pool_factor * top_n)
    top_pool_idx = np.argsort(combined_scores)[::-1][:pool_size]
    top_pool = movie_candidates.iloc[top_pool_idx].copy()
    combined_scores = np.array(combined_scores)  # ensure it's positional
    top_pool_scores = combined_scores[top_pool_idx]

    # --- Probabilistic sampling ---
    exp_preds = np.exp(top_pool_scores - top_pool_scores.max())
    probabilities = exp_preds / exp_preds.sum()

    if pool_size > top_n:
        sampled_idx = np.random.choice(pool_size, size=top_n, replace=False, p=probabilities)
        top_pool = top_pool.iloc[sampled_idx].copy()
        top_pool_scores = top_pool_scores[sampled_idx]

    top_pool['pred_rating'] = np.round(top_pool_scores, 2)

    # --- Return sorted ---
    top_movies = top_pool.sort_values('pred_rating', ascending=False)[['title','pred_rating']].reset_index(drop=True)
    top_movies.index += 1
    return top_movies