# 🎬 Movie Recommendation System

A movie recommendation engine using **Content-Based Filtering** and **Collaborative Filtering**,
built with Python, Pandas, NumPy, Scikit-Learn and Streamlit.

## Features

- **Content-Based Filtering**: Recommends movies similar in genre, cast, crew, keywords and
  plot using `CountVectorizer` (Bag of Words) + `Cosine Similarity` over 5,000+ titles
  (TMDB 5000 dataset).
- **Collaborative Filtering**: Recommends movies based on real user rating patterns
  (item-based cosine similarity) using the MovieLens dataset.
- **Streamlit Dashboard**: Interactive web UI with both recommenders in separate tabs.

## Project Structure

```
movie-recommendation-system/
├── app.py                                    # Streamlit app (main entry point)
├── content_based_preprocessing.py            # Builds content-based model (run once)
├── collaborative_filtering_preprocessing.py  # Builds collaborative filtering model (run once)
├── requirements.txt
├── .gitignore
└── README.md
```

## Datasets

### 1. TMDB 5000 Movie Dataset (for Content-Based Filtering)
- Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Download and place these files in the project folder:
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`

### 2. MovieLens ml-latest-small (for Collaborative Filtering)
- Source: https://grouplens.org/datasets/movielens/latest/
- Download the "ml-latest-small" zip, extract, and place these files in the project folder:
  - `ratings.csv`
  - `movies.csv`

> Both CSVs are excluded from git via `.gitignore` — download them locally instead of
> committing them (they're large and licensed by their providers).

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset CSVs (see "Datasets" section above) in this folder

# 5. Generate the model files (run once)
python content_based_preprocessing.py
python collaborative_filtering_preprocessing.py

# 6. Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-Learn (`CountVectorizer`, `cosine_similarity`)
- Streamlit

## How It Works

**Content-Based:**
1. Merge movie metadata (genres, keywords, cast, crew, overview).
2. Combine everything into a single "tags" string per movie.
3. Convert tags into vectors using `CountVectorizer`.
4. Compute pairwise `cosine_similarity` between all movie vectors.
5. For a selected movie, return the top 5 most similar movies.

**Collaborative Filtering:**
1. Build a User × Movie ratings matrix from MovieLens ratings.
2. Compute `cosine_similarity` between movie columns (item-based CF).
3. For a selected movie, return the top 5 movies most similar in
   how users rated them.

## Future Improvements

- Add poster images via the TMDB API
- Hybrid scoring (weighted combination of both methods)
- Deploy on Streamlit Community Cloud
