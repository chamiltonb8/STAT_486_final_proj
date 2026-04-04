import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="UMAP Visualization", layout="wide")

st.title("🧠 UMAP Movie Embeddings")

df = pd.read_csv("data/umap.csv")

movie_selection = st.selectbox("Highlight this movie:", df["title"], index=1)

# Background
bg_df = df[df["title"] != movie_selection]

fig = px.scatter(
    bg_df,
    x='x',
    y='y',
    color='rating',   # 🔥 color by rating instead of genre
    hover_data=['title'],
    opacity=0.4,
    height=800,
    title=f"UMAP View — Highlighting: {movie_selection}"
)

# Highlight selected movie
selected_df = df[df["title"] == movie_selection]

fig.add_scatter(
    x=selected_df['x'],
    y=selected_df['y'],
    name="Selected Movie",
    mode='markers',
    marker=dict(
        size=18,
        color='yellow',
        line=dict(width=2, color='black')
    ),
    hovertext=selected_df['title']
)

st.plotly_chart(fig, use_container_width=True)