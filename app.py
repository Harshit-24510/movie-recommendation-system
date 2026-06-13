"""
Movie Recommendation System - Streamlit App
----------------------------------------------
Combines two recommendation approaches:

1. Content-Based Filtering (TMDB 5000 dataset)
   - Uses CountVectorizer + Cosine Similarity on genres, cast, crew, keywords & overview

2. Collaborative Filtering (MovieLens dataset)
   - Item-based Cosine Similarity using real user rating patterns

Run:
    streamlit run app.py

Make sure you have already generated the pickle files by running:
    python content_based_preprocessing.py
    python collaborative_filtering_preprocessing.py
"""

import pickle
import streamlit as st

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Content-Based & Collaborative Filtering recommendations powered by Pandas, "
         "Scikit-Learn (CountVectorizer + Cosine Similarity) and Streamlit.")

tab1, tab2 = st.tabs(["📚 Content-Based (TMDB 5000)", "👥 Collaborative Filtering (MovieLens)"])

# -------------------------------------------------------------------
# TAB 1: Content-Based Recommendations
# -------------------------------------------------------------------
with tab1:
    st.subheader("Content-Based Recommendations")
    st.caption("Finds movies with similar genres, cast, crew, keywords & plot.")

    try:
        movies = pickle.load(open("movie_list.pkl", "rb"))
        similarity = pickle.load(open("similarity.pkl", "rb"))

        def recommend(movie_title, n=5):
            index = movies[movies["title"] == movie_title].index[0]
            distances = sorted(
                list(enumerate(similarity[index])),
                reverse=True,
                key=lambda x: x[1],
            )
            return [movies.iloc[i[0]].title for i in distances[1:n + 1]]

        selected_movie = st.selectbox(
            "Pick a movie you like:", movies["title"].values, key="content_select"
        )

        if st.button("Recommend", key="content_btn"):
            recommendations = recommend(selected_movie)
            st.success(f"Movies similar to **{selected_movie}**:")
            cols = st.columns(len(recommendations))
            for col, title in zip(cols, recommendations):
                with col:
                    st.write(f"🎥 **{title}**")

    except FileNotFoundError:
        st.error(
            "Pickle files not found! Run `python content_based_preprocessing.py` "
            "first (after downloading the TMDB 5000 dataset CSVs)."
        )

# -------------------------------------------------------------------
# TAB 2: Collaborative Filtering Recommendations
# -------------------------------------------------------------------
with tab2:
    st.subheader("Collaborative Filtering Recommendations")
    st.caption("Finds movies that users with similar taste also rated highly.")

    try:
        cf_similarity = pickle.load(open("cf_similarity.pkl", "rb"))
        cf_movies = pickle.load(open("cf_movies.pkl", "rb"))

        movie_query = st.text_input(
            "Type a movie name (partial match works, e.g. 'Toy Story')",
            key="cf_input",
        )

        if st.button("Recommend", key="cf_btn"):
            if not movie_query.strip():
                st.warning("Please enter a movie name.")
            else:
                matched = cf_movies[
                    cf_movies["title"].str.contains(movie_query, case=False, na=False)
                ]
                if matched.empty:
                    st.warning("Movie not found in the MovieLens dataset.")
                else:
                    movie_id = matched["movieId"].values[0]
                    matched_title = matched["title"].values[0]

                    if movie_id in cf_similarity.columns:
                        scores = cf_similarity[movie_id].sort_values(ascending=False)[1:6]
                        rec_titles = cf_movies[
                            cf_movies["movieId"].isin(scores.index)
                        ]["title"].tolist()

                        st.success(f"Movies similar to **{matched_title}**:")
                        for r in rec_titles:
                            st.write(f"🎥 {r}")
                    else:
                        st.warning("Not enough rating data available for this movie.")

    except FileNotFoundError:
        st.error(
            "Pickle files not found! Run `python collaborative_filtering_preprocessing.py` "
            "first (after downloading the MovieLens ml-latest-small dataset CSVs)."
        )

st.markdown("---")
st.caption("Built with Pandas, NumPy, Scikit-Learn & Streamlit")
