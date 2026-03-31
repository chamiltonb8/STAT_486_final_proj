import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import pandas as pd
from keras import ops
from sklearn.model_selection import train_test_split
import numpy as np

df_final = pd.read_csv("data/joined_df.csv")


## Define Input
user_input = layers.Input(shape=(1,),name="user_id")

## Create embedding layer for users
user_embedding = layers.Embedding(input_dim=611, output_dim=32)(user_input)

user_vec = layers.Flatten()(user_embedding)

## Create embedding for movie tower

vectorize_layer = TextVectorization(
    max_tokens = 10000,
    output_mode = 'int',
    output_sequence_length=10
)

vectorize_layer.adapt(df_final['title'].values)

integer_sequences = vectorize_layer(df_final['title'].values)

## Build the towers

def build_tower(input_dim, embedding_dim):
    inputs = layers.Input(shape=(1,))
    x = layers.Embedding(input_dim, embedding_dim)(inputs)
    x = layers.Flatten()(x)
    
    # First Dense Layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x) 
    
    # Second Dense Layer
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)  
    
    # Final Normalization 
    output = x / ops.sqrt(ops.maximum(ops.sum(ops.square(x), axis=1, keepdims=True), 1e-12))
    
    return tf.keras.Model(inputs, output)

## Initialize and build towers
num_users = df_final["userId"].max() + 1
num_movies = df_final["movieId"].max() + 1

user_tower = build_tower(num_users, 32)
movie_tower = build_tower(num_movies, 32)


## Define scores for the unsupervised method
dot_product = layers.Dot(axes=1)([user_tower.output, movie_tower.output])

score = layers.Lambda(lambda x: (x + 1.0) * 2.0 + 1.0)(dot_product)

## Full model

model = tf.keras.Model(inputs=[user_tower.input, movie_tower.input], outputs=score)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                                       loss='mean_squared_error',
                                       metrics=['mae'])

## Preprocessing

user_ids = df_final['userId'].values
movie_ids = df_final['movieId'].values
target_ratings = df_final['rating'].values

## Make train/test split with 50% of the data. When training a supervised method, only use part of the test set!

u_train, u_test, m_train, m_test, r_train, r_test = train_test_split(
    user_ids, movie_ids, target_ratings, test_size=0.5, random_state=42
)

## Fit the model

history = model.fit(
    x=[u_train, m_train],
    y=r_train,
    batch_size=64,
    epochs=10,
    validation_data=([u_test, m_test], r_test)
)

## Set up the 'null' user

user_tower.save("user_tower.keras")
movie_tower.save("movie_tower.keras")

all_ids = df_final['movieId'].unique()

all_embeddings = movie_tower.predict(all_ids)

embedding_cols = [f'v{i}' for i in range(all_embeddings.shape[1])]
export_df = pd.DataFrame(all_embeddings, columns=embedding_cols)

export_df.insert(0, 'movieId', all_ids)

export_df.to_csv('data/movie_vectors.csv', index=False)


df_final['two_tower_score'] = model.predict([df_final['userId'], df_final['movieId']])
df_final.to_csv("joined_with_tt_scores.csv")

print(df_final[['userId', 'movieId', 'rating', 'two_tower_score']].head())