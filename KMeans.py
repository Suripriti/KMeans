import streamlit as st
import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# --------------------------------
# LOAD DATASET
# --------------------------------
movies = pd.read_csv("tmdb_5000_movies.csv")

# --------------------------------
# SELECT REQUIRED COLUMNS
# --------------------------------
movies = movies[['title', 'overview', 'genres']]

# --------------------------------
# HANDLE MISSING VALUES
# --------------------------------
movies.dropna(inplace=True)

# --------------------------------
# CREATE TAGS COLUMN
# --------------------------------
movies['tags'] = movies['overview'] + " " + movies['genres']

# --------------------------------
# TEXT VECTORIZATION
# --------------------------------
cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(movies['tags']).toarray()

# --------------------------------
# TRAIN KMEANS MODEL
# --------------------------------
model = KMeans(
    n_clusters=10,
    random_state=42
)

movies['cluster'] = model.fit_predict(vectors)

# --------------------------------
# STREAMLIT UI
# --------------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    "Get movie recommendations using KMeans Clustering."
)

# --------------------------------
# MOVIE SELECTION
# --------------------------------
movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Select a Movie",
    movie_list
)

# --------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------
def recommend(movie_name):

    movie_index = movies[
        movies['title'] == movie_name
    ].index[0]

    movie_cluster = movies.iloc[movie_index]['cluster']

    recommended_movies = movies[
        movies['cluster'] == movie_cluster
    ]['title'].tolist()

    recommended_movies = [
        movie for movie in recommended_movies
        if movie != movie_name
    ]

    return recommended_movies[:10]

# --------------------------------
# BUTTON
# --------------------------------
if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write(f"✅ {movie}")