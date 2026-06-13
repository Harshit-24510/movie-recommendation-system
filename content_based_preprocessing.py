"""
Content-Based Recommender - Preprocessing Script
--------------------------------------------------
This script:
1. Loads the TMDB 5000 Movie Dataset (movies + credits)
2. Cleans and merges relevant metadata (genres, keywords, cast, crew, overview)
3. Builds a "tags" column combining all metadata
4. Vectorizes the tags using CountVectorizer (Bag of Words)
5. Computes Cosine Similarity between all movies
6. Saves the processed dataframe + similarity matrix as pickle files

Download dataset from Kaggle:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

You need two files in this folder:
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

Run:
    python content_based_preprocessing.py
"""

import pandas as pd
import numpy as np
import ast
import pickle


def convert_list(text):
    """Convert stringified list of dicts -> list of 'name' values."""
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []


def convert_cast(text, top_n=3):
    """Get top N cast member names."""
    try:
        people = ast.literal_eval(text)
        return [p["name"] for p in people[:top_n]]
    except (ValueError, SyntaxError):
        return []


def fetch_director(text):
    """Extract the director's name from the crew column."""
    try:
        for p in ast.literal_eval(text):
            if p.get("job") == "Director":
                return [p["name"]]
    except (ValueError, SyntaxError):
        pass
    return []


def remove_spaces(items):
    """Remove spaces inside multi-word names so they act as single tokens
    e.g. 'Tom Hanks' -> 'TomHanks' (avoids mixing first/last names of different people)."""
    return [str(i).replace(" ", "") for i in items]


def main():
    print("Loading datasets...")
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    print(f"Movies: {len(movies)} rows | Credits: {len(credits)} rows")

    # Merge on title
    movies = movies.merge(credits, on="title")

    # Keep only useful columns
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies.dropna(inplace=True)

    print("Parsing genres, keywords, cast, crew...")
    movies["genres"] = movies["genres"].apply(convert_list)
    movies["keywords"] = movies["keywords"].apply(convert_list)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    # Remove spaces from multi-word tokens
    movies["genres"] = movies["genres"].apply(remove_spaces)
    movies["keywords"] = movies["keywords"].apply(remove_spaces)
    movies["cast"] = movies["cast"].apply(remove_spaces)
    movies["crew"] = movies["crew"].apply(remove_spaces)

    # Build combined "tags" column
    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

    print("Vectorizing with CountVectorizer (Bag of Words, max_features=5000)...")
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(new_df["tags"]).toarray()

    print("Computing Cosine Similarity matrix...")
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(vectors)

    print("Saving pickle files: movie_list.pkl, similarity.pkl")
    pickle.dump(new_df, open("movie_list.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))

    print("Done! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
