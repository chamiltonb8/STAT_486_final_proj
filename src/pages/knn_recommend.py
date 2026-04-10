import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from utils import load_csv


# ----------------------------------
# TRAIN FULL PIPELINE (cached)
# ----------------------------------
@st.cache_resource
def build_pipeline():
    
    df = load_csv("data/supervised_split.csv") 
    df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

    # -------- DEFINE GENRES --------
    exclude_cols = [
        'userId', 'movieId', 'title', 'rating',
        'tag', 'rating_timestamp', 'tag_timestamp',
        'two_tower_score', '(no genres listed)'
    ]
    
    genre_cols = [col for col in df.columns if col not in exclude_cols]
    
    # -------- USER PROFILE --------
    df_weighted = df.copy()
    
    for col in genre_cols:
        df_weighted[col] = df_weighted[col] * df_weighted['rating']
    
    user_profile = df_weighted.groupby('userId')[genre_cols].mean()
    
    # Fill missing values
    user_avg = df.groupby('userId')['rating'].mean()
    user_profile = user_profile.apply(
        lambda row: row.fillna(user_avg[row.name]),
        axis=1
    )
    
    # -------- MERGE --------
    df_model = df.merge(user_profile, on='userId', suffixes=('', '_user'))
    
    # -------- INTERACTIONS --------
    for col in genre_cols:
        df_model[f'{col}_interaction'] = df_model[col] * df_model[f'{col}_user']
    
    # -------- FEATURES --------
    feature_cols = [f'{col}_interaction' for col in genre_cols] + ['two_tower_score']
    
    X = df_model[feature_cols]
    y = df_model['rating'] >= 4.0
    
    # -------- SPLIT --------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # -------- SCALE --------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------- LOGISTIC --------
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_scaled, y_train)
    
    y_pred_log = log_model.predict(X_test_scaled)
    f1_log = f1_score(y_test, y_pred_log)
    
    # -------- KNN --------
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train_scaled, y_train)
    
    y_pred_knn = knn.predict(X_test_scaled)
    f1_knn = f1_score(y_test, y_pred_knn)
    
    # -------- MOVIE BASE --------
    movies_df = df[['movieId', 'title'] + genre_cols + ['two_tower_score']].drop_duplicates()
    
    return {
        "df": df,
        "user_profile": user_profile,
        "movies_df": movies_df,
        "genre_cols": genre_cols,
        "scaler": scaler,
        "knn": knn,
        "log_model": log_model,
        "f1_log": f1_log,
        "f1_knn": f1_knn
    }


# ----------------------------------
# RECOMMENDER FUNCTION
# ----------------------------------
def recommend_movies(user_id, bundle, top_n=10):
    
    user_profile = bundle["user_profile"]
    movies_df = bundle["movies_df"]
    genre_cols = bundle["genre_cols"]
    scaler = bundle["scaler"]
    knn = bundle["knn"]
    df = bundle["df"]
    
    user_prefs = user_profile.loc[user_id]
    temp_df = movies_df.copy()
    
    # Build interaction features
    for col in genre_cols:
        temp_df[f'{col}_interaction'] = temp_df[col] * user_prefs[col]
    
    feature_cols = [f'{col}_interaction' for col in genre_cols] + ['two_tower_score']
    
    X_pred = temp_df[feature_cols]
    X_scaled = scaler.transform(X_pred)
    
    scores = knn.predict_proba(X_scaled)[:, 1]
    temp_df['score'] = scores
    
    # Remove watched
    watched = df[df['userId'] == user_id]['movieId']
    temp_df = temp_df[~temp_df['movieId'].isin(watched)]
    
    return temp_df.sort_values('score', ascending=False).head(top_n)


# ----------------------------------
# STREAMLIT UI
# ----------------------------------
st.set_page_config(page_title="KNN + Logistic Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

bundle = build_pipeline()

# Show model performance
st.subheader("📊 Model Performance")
st.write(f"Logistic Regression F1: {bundle['f1_log']:.3f}")
st.write(f"KNN F1: {bundle['f1_knn']:.3f}")

# User selector
user_ids = bundle["user_profile"].index.tolist()
selected_user = st.selectbox("Select a user:", user_ids)

# Button
if st.button("Get Recommendations"):
    
    recs = recommend_movies(selected_user, bundle)
    
    st.subheader("🎯 Recommended Movies")
    
    for _, row in recs.iterrows():
        st.write(f"🎥 {row['title']} — Score: {row['score']:.3f}")


# ----------------------------------
# OPTIONAL: FEATURE IMPORTANCE
# ----------------------------------
if st.checkbox("Show Logistic Feature Importance"):
    
    log_model = bundle["log_model"]
    genre_cols = bundle["genre_cols"]
    
    feature_cols = [f'{col}_interaction' for col in genre_cols] + ['two_tower_score']
    
    importance = pd.Series(log_model.coef_[0], index=feature_cols)
    importance = importance.sort_values(key=abs, ascending=False).head(15)
    
    st.write("Top Features:")
    st.dataframe(importance)