import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

from sklearn.metrics.pairwise import linear_kernel

# Paths
MODEL_PATH = 'artifacts/svd_model.pkl'
MOVIES_PATH = 'artifacts/movies.csv'
RATINGS_PATH = 'artifacts/ratings.csv'
TFIDF_PATH = 'artifacts/tfidf_genre.pkl'
GENRE_MATRIX_PATH = 'artifacts/genre_matrix.pkl'


@st.cache_data
def load_artifacts():
    algo = joblib.load(MODEL_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    with open(GENRE_MATRIX_PATH, 'rb') as f:
        genre_matrix = pickle.load(f)
    return algo, movies, ratings, tfidf, genre_matrix


algo, movies_df, ratings_df, tfidf, genre_matrix = load_artifacts()

st.title('Movie Recommender — MovieLens 100k')

mode = st.sidebar.selectbox('Recommendation mode', [
                            'By user id (collaborative)', 'By movie (content-based)'])

if mode == 'By user id (collaborative)':
    st.write('Get top recommendations for an existing user id')
    user_id = st.number_input('User id', min_value=int(ratings_df.user_id.min(
    )), max_value=int(ratings_df.user_id.max()), value=int(ratings_df.user_id.min()))
    n = st.slider('Number of recommendations', 1, 20, 10)
    if st.button('Recommend'):
        # compute predictions for all unseen movies
        with st.spinner('Generating recommendations...'):
            seen = set(ratings_df[ratings_df.user_id == user_id].movie_id)
            all_movie_ids = movies_df['movie_id'].unique()
            preds = []
            for mid in all_movie_ids:
                if mid in seen:
                    continue
                try:
                    pred = algo.predict(user_id, int(mid))
                    preds.append((mid, pred.est))
                except Exception:
                    continue
            preds.sort(key=lambda x: x[1], reverse=True)
            top = preds[:n]
            result = movies_df[movies_df.movie_id.isin([m for m, _ in top])].merge(pd.DataFrame(
                top, columns=['movie_id', 'est']), on='movie_id').sort_values('est', ascending=False)
            st.write(result[['movie_id', 'title', 'est']
                            ].reset_index(drop=True))

else:
    st.write('Find movies similar to a chosen movie (content-based by genres)')
    movie_title = st.text_input(
        'Type part of a movie title to search (e.g. Toy Story)')
    n = st.slider('Number of similar movies', 1, 20, 10)
    if movie_title:
        matches = movies_df[movies_df.title.str.lower(
        ).str.contains(movie_title.lower())]
        if matches.empty:
            st.write('No matching movie found')
        else:
            choice = st.selectbox('Choose a movie', matches['title'].tolist())
            if st.button('Find similar'):
                chosen = matches[matches.title == choice].iloc[0]
                idx = movies_df.index[movies_df.movie_id == chosen.movie_id].tolist()[
                    0]
                cosine_similarities = linear_kernel(
                    genre_matrix[idx], genre_matrix).flatten()
                related_indices = cosine_similarities.argsort()[::-1]
                related_indices = [i for i in related_indices if i != idx][:n]
                similar = movies_df.iloc[related_indices].assign(
                    score=cosine_similarities[related_indices])
                st.write(similar[['movie_id', 'title', 'score']
                                 ].reset_index(drop=True))

st.sidebar.markdown('---')
st.sidebar.write('Artifacts loaded from `artifacts/`')
