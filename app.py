import streamlit as st
import pandas as pd
import joblib

# --- Load model and data ---
clf = joblib.load('movie_recommender_model.pkl')  # Load your trained ML model
# Load preprocessed movie data
movies = pd.read_csv('your_preprocessed_movies.csv')

# --- Genre columns used in training ---
genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


# --- App Title ---
st.title("🎬 Bosy Movie Recommender")
st.write("Answer a few questions and Bosy will recommend movies you'll probably love!")

# --- Quiz Questions ---
selected_genres = st.multiselect("🎭 What genres do you like?", genre_cols)
decade = st.selectbox("📅 Which decade do you prefer?",
                      [None, 1980, 1990])
popularity = st.radio("📈 Do you prefer popular or underrated movies?", [
                      'Popular', 'Underrated'])
top_n = st.slider("🎯 How many recommendations do you want?",
                  min_value=5, max_value=20, value=10)

# --- Helper Function to Prepare Input Data ---


def build_features_from_quiz(movies_df, selected_genres, decade, popularity):
    df = movies_df.copy()

    # Filter by decade
    if decade:
        df = df[(df['year'] >= decade) & (df['year'] < decade + 10)]

    # Filter by popularity
    if popularity == 'Popular':
        df = df[df['rating_count'] >= 100]
    else:
        df = df[df['rating_count'] < 100]

    # Apply genre flags
    for genre in genre_cols:
        df[genre] = df[genre].astype(int)
        if genre in selected_genres:
            df[genre] = 1
        else:
            df[genre] = 0

    return df


# --- When Button Clicked, Predict & Recommend ---
if st.button("🎥 Show Recommendations"):
    input_df = build_features_from_quiz(
        movies, selected_genres, decade, popularity)

    # Feature columns used in training
    model_features = ['year', 'avg_rating', 'rating_count'] + genre_cols
    X_input = input_df[model_features]

    # Predict "like" probability
    input_df['like_prob'] = clf.predict_proba(X_input)[:, 1]

    # Top N Recommendations
    top_movies = input_df.sort_values(
        by='like_prob', ascending=False).head(top_n)

    st.subheader("🎯 Recommended Movies:")
    st.table(
        top_movies[['title', 'year', 'avg_rating', 'rating_count', 'like_prob']])
