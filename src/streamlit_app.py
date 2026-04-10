import streamlit as st

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

st.markdown("""
Welcome! Use the sidebar to navigate:

- 📊 UMAP Visualization  
- 🤖 KNN Recommendations  
- 🧠 Neural Recommendations  
""")

st.sidebar.success("Select a page above 👆")