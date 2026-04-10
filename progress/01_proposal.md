# Project Proposal

## Final Research Question

Can we predict which movies a user will like given what they have watched in the past?

### Logistics

To do this, we will use Movie Ratings data to explore which movies people tend to like. Using various machine learning techniques, we will attempt to predict what users will rate a movie. Specifically, our supervised learning target is the `rating` variable.

In addition to raw predictions, we would like to use a Two Towers or an SVD recommender system to filter the massive amounts of data. This will help us to create a 'profile' for a user and obtain general movie recommendations for them. We will then use a regression or classification model to predict the user's rating.

In our analysis, we will use the [grouplens movie dataset](https://grouplens.org/datasets/movielens/) which has data on user ratings and tags. If this data does not work out for us, we will use [this data](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=ratings.csv) which contains similar content.

This project will definitely stretch us, but we feel that it would be a valuable demonstration of our machine learning skills. We will need to use feature engineering techniques and unfamiliar models and packages to finish this project, but anticipate it being a valuable use of our time.

We don't have any major ethical concerns, but we should make sure to abide by the guidelines given by the grouplens license.

## Note on AI

AI was very helpful in generating various ideas for us to work with. We had interest in using a recommender system, and AI helped us find that movie data is accessible and relevant. In addition to our above proposal, AI also suggested:

- Predicting whether a user would begin watching a certain movie
- Predicting how well a movie would perform based solely on metadata

although we decided to move forward with our recommender system as it felt the most feasible. AI also assisted in formulating our model structure by identifying useful recommender systems and datasets. We used AI to gauge how effective certain systems would work, and brainstormed potential 'hybrid' models that utilize multiple methods. To highlight this, we present a brief excerpt of our conversation:

Prompt:

> Tell me specifically how we could implement a regression/classification system as well as a recommender model.

Model:

> To implement this, you will build a **Hybrid Recommender System.** ... You start with a standard **Collaborative Filtering** model, like **Singular Value Decomposition.** ... Now, you treat the **Collaborative Score** from Step 1 as just one feature in a larger tabular dataset.
