import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/tsne.csv")

st.set_page_config(page_title="Movie Recommendation Applet", layout="wide")
st.title("🎬 Movie Recommendation Applet")

movie_selection = st.selectbox("Highlight this movie:", df["title"], index=1)

# --- LAYER 1: The Background ---
# Filter for everything EXCEPT the selected movie
bg_df = df[df["title"] != movie_selection]

fig = px.scatter(
    bg_df, 
    x='x', 
    y='y', 
    color='genre', 
    hover_data=['title'],
    opacity=0.45,  # Makes all background points faint
    height=800,
    title=f"Highlighting: {movie_selection}"
)

# --- LAYER 2: The Highlight ---
# Filter for ONLY the selected movie
selected_df = df[df["title"] == movie_selection]

fig.add_scatter(
    x=selected_df['x'], 
    y=selected_df['y'],
    name="Selected Movie",
    mode='markers',
    marker=dict(
        size=18, 
        color='yellow', 
        opacity=1.0, # Pure solid
        line=dict(width=2, color='black') # Outline makes it pop!
    ),
    hovertext=selected_df['title']
)

st.plotly_chart(fig, use_container_width=True)