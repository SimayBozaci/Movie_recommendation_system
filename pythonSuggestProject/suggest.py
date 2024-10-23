
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot
from sklearn.model_selection import train_test_split


def load_data():
    movies = pd.read_csv('/Users/simaybozaci/PycharmProjects/pythonSuggestProject/ml-latest-small/movies.csv')
    ratings = pd.read_csv('/Users/simaybozaci/PycharmProjects/pythonSuggestProject/ml-latest-small/ratings.csv')
    return movies, ratings


def create_user_movie_matrix(ratings, movies):
    movie_ratings = pd.merge(ratings, movies, on='movieId')
    user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')
    user_movie_matrix.fillna(0, inplace=True)
    return user_movie_matrix

def calculate_similarity(user_movie_matrix):
    return cosine_similarity(user_movie_matrix.T)

# Öneri Fonksiyonu (Temel Yöntem)
def recommend_movies_basic(user_id, user_movie_matrix, movie_similarity_df, num_recommendations=5):
    user_ratings = user_movie_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index.tolist()

    similar_movies = pd.Series(dtype=float)
    for movie in watched_movies:
        similar_scores = movie_similarity_df[movie].sort_values(ascending=False)
        similar_movies = pd.concat([similar_movies, similar_scores])

    similar_movies = similar_movies.groupby(similar_movies.index).mean()
    similar_movies = similar_movies[~similar_movies.index.isin(watched_movies)]

    return similar_movies.sort_values(ascending=False).head(num_recommendations)

# Öneri Fonksiyonu (Derin Öğrenme Yöntemi)
def recommend_movies_deep(user_id, model, user_to_id, movie_ids, num_recommendations=5):
    user_index = user_to_id[user_id]
    movie_indices = np.array(list(range(len(movie_ids))))

    user_array = np.full((len(movie_ids),), user_index)
    ratings_predicted = model.predict([user_array, movie_indices])

    recommended_indices = np.argsort(ratings_predicted.flatten())[-num_recommendations:][::-1]
    recommended_movies = [movie_ids[i] for i in recommended_indices]

    return recommended_movies

# Modeli Oluşturma
def build_model(num_users, num_movies):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, 50)(user_input)
    movie_embedding = Embedding(num_movies, 50)(movie_input)

    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)

    similarity = Dot(axes=1)([user_vector, movie_vector])
    model = Model(inputs=[user_input, movie_input], outputs=similarity)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Ana Fonksiyon
def main():
    
    movies, ratings = load_data()

    user_movie_matrix = create_user_movie_matrix(ratings, movies)

    
    movie_similarity = calculate_similarity(user_movie_matrix)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    # Öneri Yapma (Temel Yöntem)
    recommendations_basic = recommend_movies_basic(user_id=1, user_movie_matrix=user_movie_matrix, movie_similarity_df=movie_similarity_df)
    print("Temel Öneriler:\n", recommendations_basic)

    user_ids = ratings['userId'].unique().tolist()
    movie_ids = ratings['movieId'].unique().tolist()
    user_to_id = {user: i for i, user in enumerate(user_ids)}
    movie_to_id = {movie: i for i, movie in enumerate(movie_ids)}
    ratings['user'] = ratings['userId'].map(user_to_id)
    ratings['movie'] = ratings['movieId'].map(movie_to_id)

    
    X = ratings[['user', 'movie']].values
    y = ratings['rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  
    model = build_model(num_users=len(user_to_id), num_movies=len(movie_to_id))

    
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, verbose=1)

    # Öneri Yapma 
    recommended_movies = recommend_movies_deep(user_id=1, model=model, user_to_id=user_to_id, movie_ids=movie_ids)
    print("Derin Öğrenme Önerileri:", recommended_movies)

if __name__ == "__main__":
    main()
