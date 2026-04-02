## Additional ML Method- Two Towers Recommender System

As a prerequisite to any supervised modeling, we thought it helpful to use a Two Towers system to extract meaningful differences between movies that are not explicitly included in our data. The key feature we extract from this system is an Affinity score, which looks at which movies users have watched and then calculates how close the user is to the movie in a certain space we defined.

The Affinity between a user and a movie is extremely important. Were we to predict user ratings of specific movies, we would only be able to take into account (a) user average ratings, and (b) movie average ratings. As a result, most user recommendations would look exactly the same, and predictions would be heavily genre-based (and not specific movie-based). The Affinity scores we calculate help us extract other factors, particularly in the form of movies that similar users tend to like.

Effectively, our Two Towers model examines the relationships between users and movies they liked and disliked. Using these factors, it maps out both the users and the movies to the same 32D space.

Code for the Two Towers system may be found [here](https://github.com/chamiltonb8/STAT_486_final_proj/blob/main/src/two_towers_setup.py).

### Methodology

To initialize our system, we structured two different "towers" of neural networks for users and movies respectively. These towers convert user and movie ID's into a 32D vector. Then, using Dense layers within a neural network, a dropout rate of 0.1, and normalization, our model has an output vector with magnitude 1. 

We then compile both of these models to create a single neural network that converts users and movies into the same 32D space. Using user ratings on each movie, the model attempts to separate users from movies they rated poorly, and keep them together with movies they enjoyed. We then calculated an affinity score for each user and movie pair using the cosine distance between the two, then scaling it to a rating between 1-5 (so that it fits with our rating scale).

Eventually, we hope to use these affinity scores in a supervised method. Because of the mappings our model gave us, we can calculate the affinity score between any user and movie. Then, we effectively turn our supervised models into a sort of 'boosting' method where we try to predict the rating they will give a movie based on the affinity score.

### Results 

Using t-SNE, we mapped the 32D movie vectorbase into 2D for visualization purposes. The results can be found on [this Streamlit app](https://tsne-movie-recommendations.streamlit.app/), which allows users to select movies and see where they lie in relation to the others. As a massive drop in dimensionality, this will not be used in our future analysis, although it provides an interesting look at (a) the efficacy of our Two Towers system, and (b) the versatility of t-SNE. One particularly interesting cluster found in basic searches yielded *The Godfather* (parts 1 and 2) and *Star Wars* (episodes 4 and 5) relatively closely to each other. These movies are older, but fan favorites that have persisted since their releases. It makes sense that fans of one would enjoy the other.

In addition, this app presents a fun exploratory form of searching for movie recommendations. Users can look up their favorite movie, then search the plot for nearby movies that appear interesting.

### Conclusion

The Two Towers model we have built provides an effective foundation for a supervised model. By forming a spatial relationship between movies, we may make our supervised predictions more effective. The key insight we derive from this model is the trends in behavior that cannot be written easily in the data.
