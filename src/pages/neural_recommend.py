# src/pages/neural_recommend.py
import streamlit as st
import pandas as pd
import os
import torch
import pickle
from neural_predict import MovieRecommenderEmb, predict_top_movies

# -----------------------------
# LOAD MODEL + DATA FUNCTION
# -----------------------------
@st.cache_resource
def load_model():
    # Project root is one level above src
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # src/
    ROOT_DIR = os.path.dirname(BASE_DIR)                  # project_root/
    
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(ROOT_DIR, "data")  # <-- data folder is outside src

    # Full ratings dataset (for user filtering, popularity, etc.)
    df = pd.read_csv(os.path.join(DATA_DIR, "joined_with_tt_scores.csv"))

    # Movie features dataset (only features used for prediction)
    movies_df = pd.read_csv(os.path.join(MODEL_DIR, "movie_features.csv"))

    # Feature columns for model
    feature_cols = [col for col in movies_df.columns]  # all columns in movies_df

    # Load scaler and movie2idx
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    movie2idx = pickle.load(open(os.path.join(MODEL_DIR, "movie2idx.pkl"), "rb"))

    # User/movie counts
    num_users = df['userId'].nunique()
    num_movies = len(movie2idx)  # must match checkpoint

    # Load model
    embedding_dim = 32
    input_dim = len(feature_cols)
    mean_rating = df['rating'].mean()

    model = MovieRecommenderEmb(num_users, num_movies, embedding_dim, input_dim, mean_rating)
    state_dict = torch.load(os.path.join(MODEL_DIR, "movie_model_state.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model. Missing keys: {missing}, Unexpected keys: {unexpected}")

    return model, scaler, movie2idx, movies_df, feature_cols, df


# -----------------------------
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(page_title="Neural Movie Recommender", layout="wide")
st.title("🎬 Neural Movie Recommendation System")

# Load everything
model, scaler, movie2idx, movies_df, feature_cols, df = load_model()

# User selector
user_ids = df['userId'].unique()
selected_user = st.selectbox("Select a user:", user_ids)

# Recommendation button
if st.button("Get Recommendations"):
    top_movies = predict_top_movies(
        user_id=selected_user,
        df=df,
        movies_df=movies_df,
        feature_cols=feature_cols,
        model=model,
        scaler=scaler,
        movie2idx=movie2idx,
        top_n=10,
        min_rating=3.5,
        random_seed=42,
        top_pool_factor=3,
        noise_std=0.02,
        popularity_weight=0.2
    )

    st.subheader("🎯 Recommended Movies")
    for idx, row in top_movies.iterrows():
        st.write(f"{idx}. 🎥 {row['title']} — Predicted Rating: {row['pred_rating']:.2f}")