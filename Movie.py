
import streamlit as st 
import numpy as np
import requests
import pickle


with open('movie_dict.pkl','rb') as file:
    obj = pickle.load(file)

# Support two possible pickle formats:
# 1) (movies_dataframe, cosine_sim_matrix)
# 2) dict-like data representing movies (e.g., keys: movie_id, title, tags, etc.)
if isinstance(obj, tuple) and len(obj) == 2:
    movies, cosine_sim = obj
else:
    import pandas as _pd

    if isinstance(obj, dict):
        movies = _pd.DataFrame(obj)
    elif isinstance(obj, _pd.DataFrame):
        movies = obj
    else:
        raise RuntimeError(
            "Unsupported format in movie_dict.pkl: expected tuple (movies, cosine_sim) or a dict/DataFrame representing movies"
        )

    # Compute cosine similarity from text tags (requires scikit-learn)
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required to compute similarity when only movie data is present.\n"
            "Install it with: python -m pip install scikit-learn"
        ) from exc

    movies['tags'] = movies.get('tags', '').fillna('')
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags'])
    # Build a NearestNeighbors model to find similar movies efficiently (avoids creating a full dense matrix)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(metric='cosine', algorithm='brute')
    nbrs.fit(vectors)
    cosine_sim = None  # kept for backward compatibility

def get_recommendations(title, n_recs=10):
    idx = movies[movies['title'] == title].index[0]

    # Preferred: use the precomputed NearestNeighbors model (memory-efficient)
    try:
        if nbrs is not None:
            distances, indices = nbrs.kneighbors(vectors[idx], n_neighbors=n_recs + 1)
            neighbor_idxs = indices[0][1:n_recs + 1]
            return movies['title'].iloc[neighbor_idxs]
    except NameError:
        pass

    # Fallback: if a cosine similarity matrix exists in the pickle, use it
    if 'cosine_sim' in globals() and cosine_sim is not None:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        neighbor_idxs = [i[0] for i in sim_scores[1:n_recs + 1]]
        return movies['title'].iloc[neighbor_idxs]

    raise RuntimeError('No similarity model available. Ensure the pickle contains either a movie dataset with tags or a precomputed similarity matrix.')

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, timeout=5, retries=3, backoff=0.3):
    """Fetch poster URL for a TMDB movie id with retries and fallback placeholder.

    Returns a full poster URL or a placeholder image URL on error.
    """
    api_key = 'fe586bb2edfd9813f24a76872c1ee756'  # Replace with your actual TMDB API key
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'
    placeholder = "https://via.placeholder.com/500x750?text=No+Image"

    try:
        # configure a session with retries to be robust to transient network errors
        session = requests.Session()
        from requests.adapters import HTTPAdapter
        try:
            # urllib3 >=1.26: use Retry from urllib3.util.retry
            from urllib3.util import Retry
        except Exception:
            from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return placeholder
        return "https://image.tmdb.org/t/p/w500" + poster_path

    except requests.exceptions.RequestException:
        # Don't crash the app for network issues — show a placeholder and a brief warning
        try:
            st.warning('Could not fetch poster images (network error); showing placeholder.')
        except Exception:
            pass
        return placeholder
    except Exception:
        # Catch-all to be defensive
        return placeholder

st.title('Movie Recommender System')
selected_movie = st.selectbox(
    'Select a movie:',
    movies['title'].values
)


if st.button('Recommend'):
    recommendations = get_recommendations(selected_movie)
    st.write('Top 10 movie recommendations for you:')

    #create a 2x5 grid layout
    for i in range(0, 10, 5): # Loop over rows(2 rows and 5 movies each)
        cols = st.columns(5)  # Create 5 columns for each row
        for col, j in zip(cols, range(i, i+5)):
            if j < len(recommendations):
                movie_title = recommendations.iloc[j]
                movie_id = movies[movies['title'] == movie_title]['movie_id'].values[0]
                poster_url = fetch_poster(movie_id)
                with col:
                    st.text(movie_title)
                    st.image(poster_url)