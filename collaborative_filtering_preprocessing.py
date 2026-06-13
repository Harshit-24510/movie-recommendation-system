"""
Collaborative Filtering Recommender - Preprocessing Script
------------------------------------------------------------
This script builds an ITEM-BASED collaborative filtering model using
real user ratings from the MovieLens dataset.

Idea:
- Build a User x Movie ratings matrix
- Compute Cosine Similarity BETWEEN MOVIES based on how users rated them
- Movies that tend to receive similar ratings from the same users
  are considered "similar"

Download dataset from GroupLens (ml-latest-small, ~1MB, recommended for testing):
https://grouplens.org/datasets/movielens/latest/

You need two files in this folder:
- ratings.csv   (columns: userId, movieId, rating, timestamp)
- movies.csv    (columns: movieId, title, genres)

Run:
    python collaborative_filtering_preprocessing.py
"""

import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def main():
    print("Loading MovieLens data...")
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")

    print(f"Ratings: {len(ratings)} rows | Movies: {len(movies)} rows")

    print("Building User-Movie ratings matrix...")
    user_movie_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    print("Computing item-item Cosine Similarity (this may take a moment)...")
    item_similarity = cosine_similarity(user_movie_matrix.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_movie_matrix.columns,
        columns=user_movie_matrix.columns,
    )

    print("Saving pickle files: cf_similarity.pkl, cf_movies.pkl")
    pickle.dump(item_similarity_df, open("cf_similarity.pkl", "wb"))
    pickle.dump(movies, open("cf_movies.pkl", "wb"))

    print("Done! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
