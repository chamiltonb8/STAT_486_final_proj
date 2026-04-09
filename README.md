# STAT_486_final_proj

A movie recommendation project that combines exploratory data analysis, unsupervised learning, and neural recommendation models.

Take a look at the [Streamlit App](https://stat486finalproj.streamlit.app/UMAP_view) that we built for this project!

## Project Overview

This repository contains the code, data, experiments, and documentation for a movie recommender system built with both classical and neural methods. The goal is to analyze movie-rating data, build embeddings, and provide interactive recommendations via a Streamlit app.

## Runtime Order

1. `data_cleaning.ipynb`

2. `two_towers_setup.py`

3. `tt_validation.ipynb` (to view a t-SNE plot of the data)

4. All other notebooks

## Repository Structure

- `data/`
  - `joined_df.csv`: main dataset combining ratings, movie metadata, and features.
  - `joined_with_tt_scores.csv`: dataset augmented with two-tower model scores.
  - `movie_vectors.csv`: saved movie embedding vectors from the two-tower approach.
  - `movie_embeddings.npy`, `movie_id_index.npy`: embedding artifacts used for similarity-based retrieval.
  - `movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`: raw MovieLens-style source files.
  - `umap.csv`, `tsne.csv`: dimension reduction outputs for visualization.

- `progress/`
  - Weekly updates and project notes.
  - `01_proposal.md`, `02_eda.md`, `03_unsupervised.md`

- `src/`
  - `streamlit_app.py`: main Streamlit launcher for the interactive recommender.
  - `two_towers_setup.py`: builds and trains a user/movie two-tower model and exports movie embeddings.
  - `neural_predict.py`: PyTorch-based recommender architecture and top-movie prediction helper.
  - `supervised.py`: placeholder for supervised modeling code.
  - `pages/`: Streamlit app pages for navigation and visualization.

- `models/`
  - Saved model files and weights for prediction and embedding reuse.

- `notebooks/`
  - Analysis notebooks covering data cleaning, KNN, neural networks, and validation.

## Getting Started

### Requirements

The project requires Python 3.11+ and the dependencies listed in `pyproject.toml` or `requirements.txt`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run the Streamlit App

Start the interactive application from the repository root:

```bash
streamlit run src/streamlit_app.py
```

The app uses `src/pages/` to render pages for:

- `UMAP_view.py`: visualize movie embeddings via UMAP
- `knn_recommend.py`: KNN recommendation interface
- `neural_recommend.py`: neural recommendation interface

## Core Components

### `src/two_towers_setup.py`

This script builds a two-tower network:

- A user embedding tower and a movie embedding tower
- Trains on `data/joined_df.csv`
- Saves `user_tower.keras` and `movie_tower.keras`
- Produces `data/movie_vectors.csv`
- Writes out `data/joined_with_tt_scores.csv` with two-tower predictions

### `src/neural_predict.py`

Contains a PyTorch recommender model and a `predict_top_movies` helper that:

- computes predictions for unseen movies
- applies popularity-based weighting and noise for ranking
- returns the top N movie recommendations

### `src/streamlit_app.py`

The Streamlit launcher provides a user-facing interface for exploring visualizations and getting movie recommendations.

## Data Workflow

1. Raw data is stored in `data/`.
2. `two_towers_setup.py` trains the embedding-based model and exports scored data.
3. `neural_predict.py` supports prediction logic used in experiment notebooks or the app.
4. Saved model artifacts live in `models/` for reuse.

## Notebooks

The `notebooks/` directory includes exploratory and modeling work:

- `data_cleaning.ipynb`
- `full_knn.ipynb`
- `knn-and-logistic.ipynb`
- `neural_networks.ipynb`
- `tt_validation.ipynb`

## Notes

- `src/supervised.py` is currently a placeholder for supervised learning code.
- The Streamlit app is the best starting point to explore recommendations interactively.
- Use the notebooks to reproduce experiments and understand how the models were developed.
