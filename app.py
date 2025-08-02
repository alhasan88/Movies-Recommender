import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# Set page configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load data


@st.cache_data
def load_data():
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                         usecols=[0, 1] + list(range(5, 24)),
                         names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_cols)

    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])

    movie_stats = ratings.groupby('movie_id').agg(
        {'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'rating_count']
    movie_stats = movie_stats.reset_index()

    movies = movies.merge(movie_stats, on='movie_id')
    return movies, genre_cols

# Load model


@st.cache_resource
def load_model():
    return joblib.load('movie_recommender_model.pkl')

# Preprocess features


def preprocess_features(movies, genre_cols):
    features = movies[genre_cols + ['avg_rating', 'rating_count']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

# Main app


def main():
    st.title("Movie Recommendation System")
    st.markdown(
        "Explore movie clusters and get personalized movie recommendations using the MovieLens dataset.")

    # Load data and model
    movies, genre_cols = load_data()
    kmeans = load_model()

    # Assign clusters
    features = preprocess_features(movies, genre_cols)
    movies['cluster'] = kmeans.predict(features)

    # Sidebar for filters
    st.sidebar.header("Filter Options")
    selected_genres = st.sidebar.multiselect(
        "Select Genres", genre_cols, default=['Comedy', 'Drama'])
    min_rating = st.sidebar.slider(
        "Minimum Average Rating", 1.0, 5.0, 3.0, step=0.1)
    min_ratings = st.sidebar.slider("Minimum Number of Ratings", 1, 500, 50)

    # Filter movies
    filtered_movies = movies[movies['avg_rating'] >= min_rating]
    filtered_movies = filtered_movies[filtered_movies['rating_count'] >= min_ratings]
    if selected_genres:
        genre_filter = filtered_movies[selected_genres].sum(axis=1) > 0
        filtered_movies = filtered_movies[genre_filter]

    # Display filtered movies
    st.header("Filtered Movies")
    st.dataframe(filtered_movies[[
                 'title', 'avg_rating', 'rating_count', 'cluster'] + selected_genres].head(50))

    # PCA Visualization
    st.header("Movie Clusters Visualization")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1],
                    hue=movies['cluster'], palette='Set2', s=60, ax=ax)
    ax.set_title('Movie Clusters Visualized using PCA')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(title='Cluster', loc='best')
    ax.grid(True)
    st.pyplot(fig)

    # Cluster Profiles
    st.header("Cluster Profiles")
    cluster_profiles = movies.groupby(
        'cluster')[genre_cols + ['avg_rating', 'rating_count']].mean()
    cluster_profiles['num_movies'] = movies['cluster'].value_counts(
    ).sort_index()
    st.dataframe(cluster_profiles.round(2))

    # Movie Recommendation
    st.header("Get Movie Recommendations")
    selected_movie = st.selectbox(
        "Select a Movie", movies['title'].sort_values())
    if selected_movie:
        selected_movie_id = movies[movies['title']
                                   == selected_movie]['movie_id'].iloc[0]
        selected_cluster = movies[movies['title']
                                  == selected_movie]['cluster'].iloc[0]

        # Recommend movies from the same cluster
        recommendations = movies[movies['cluster'] == selected_cluster][[
            'title', 'avg_rating', 'rating_count']]
        recommendations = recommendations[recommendations['title'] != selected_movie].head(
            5)

        st.subheader(f"Recommended Movies (Cluster {selected_cluster})")
        st.dataframe(recommendations)


if __name__ == "__main__":
    main()
