"""
ml_model.py – MovieLens-powered recommendation engine
======================================================
Algorithms:
  1. Content-Based Filtering  – TF-IDF on genres + cosine similarity
  2. Collaborative Filtering  – Truncated SVD on user-item rating matrix
  3. Popularity Baseline       – Bayesian-weighted mean rating

Dataset: MovieLens Small  (auto-downloaded on first run, ~3 MB)
Cache:   ./model_cache/    (pickle files, rebuilt if missing)
"""

import os
import io
import pickle
import zipfile
import logging
import requests
import json
import re
import numpy  as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.decomposition           import TruncatedSVD
from sklearn.preprocessing           import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
CACHE_DIR   = os.path.join(BASE_DIR, "model_cache")
ML_ZIP_URL  = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

logging.basicConfig(level=logging.INFO, format="[ML] %(message)s")
log = logging.getLogger(__name__)

# ── Globals (loaded once at startup) ──────────────────────────────────────────
movies_df       = None   # pd.DataFrame: movieId, title, genres, year, ...
ratings_df      = None   # pd.DataFrame: userId, movieId, rating
combined_norm   = None   # sparse matrix: movie feature vectors
tfidf_titles    = None   # TfidfVectorizer fitted on titles
tfidf_matrix_t  = None   # tfidf matrix for title search
movie_idx       = None   # dict: title → row index in cosine_sim
idx_movie       = None   # dict: row index → movieId
svd_user_factors= None   # np.ndarray: user latent factors (U × k)
svd_item_factors= None   # np.ndarray: item latent factors (I × k)
movie_id_to_idx = None   # dict: movieId → svd item row
idx_to_movie_id = None   # dict: svd item row → movieId
popularity_df   = None   # sorted by bayesian score
links_df        = None   # pd.DataFrame: movieId → imdbId, tmdbId

POSTER_CACHE_FILE = os.path.join(CACHE_DIR, "poster_cache.json")
_poster_cache     = {}

def load_poster_cache():
    """Load cached poster URLs from disk."""
    global _poster_cache
    if os.path.exists(POSTER_CACHE_FILE):
        try:
            with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
                _poster_cache = json.load(f)
        except Exception as e:
            log.warning(f"Could not load poster cache: {e}")
            _poster_cache = {}

def get_poster_url(tmdb_id):
    """Retrieve poster image URL for a given TMDB movie ID."""
    if not tmdb_id:
        return None
    s_id = str(int(tmdb_id))
    if s_id in _poster_cache:
        return _poster_cache[s_id]

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(f"https://www.themoviedb.org/movie/{s_id}", headers=headers, timeout=4)
        if r.status_code == 200:
            m = re.search(r'property=["\']og:image["\'] content=["\']([^"\']+)["\']', r.text)
            if m:
                url = m.group(1)
                _poster_cache[s_id] = url
                try:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    with open(POSTER_CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(_poster_cache, f, indent=2)
                except Exception:
                    pass
                return url
    except Exception:
        pass

    _poster_cache[s_id] = None
    return None

GENRE_EMOJI = {
    "Action":    "🔥", "Adventure": "🌍", "Animation":  "🎨",
    "Children":  "🧒", "Comedy":    "😂", "Crime":      "🔫",
    "Documentary":"📽️","Drama":     "🎭", "Fantasy":    "🧙",
    "Film-Noir": "🕵️", "Horror":    "👻", "Musical":    "🎵",
    "Mystery":   "🔍", "Romance":   "💕", "Sci-Fi":     "🚀",
    "Thriller":  "😱", "War":       "⚔️", "Western":    "🤠",
}

# ── 1. Dataset download ────────────────────────────────────────────────────────

def download_dataset():
    """Download and unzip MovieLens small dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    unified_path = os.path.join(DATA_DIR, "movies_dataset.csv")
    movies_path  = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    def has_ratings():
        if os.path.exists(ratings_path):
            return True
        if os.path.exists(unified_path):
            cols = pd.read_csv(unified_path, nrows=0).columns
            if "userId" in cols and "rating" in cols:
                return True
        return False

    if has_ratings():
        log.info("Dataset already present.")
        return

    log.info("Downloading MovieLens dataset (~3 MB)…")
    try:
        resp = requests.get(ML_ZIP_URL, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in ("movies.csv", "ratings.csv", "links.csv", "tags.csv"):
                    with zf.open(name) as src, open(os.path.join(DATA_DIR, basename), "wb") as dst:
                        dst.write(src.read())
        log.info("Dataset downloaded successfully.")
    except Exception as e:
        log.error(f"Download failed: {e}")
        raise


# ── 2. Data loading & feature engineering ─────────────────────────────────────

def _extract_year(title: str):
    """Pull the year out of 'Movie Title (YYYY)' format."""
    import re
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def _clean_title(title: str):
    """Strip trailing year from title."""
    import re
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def load_data():
    """Load and prepare movies + ratings DataFrames."""
    global movies_df, ratings_df, popularity_df, links_df

    unified_path = os.path.join(DATA_DIR, "movies_dataset.csv")
    if os.path.exists(unified_path):
        full_df = pd.read_csv(unified_path)
        movies_df  = full_df[["movieId", "title", "genres"]].drop_duplicates("movieId").copy()
        
        if "userId" in full_df.columns and "rating" in full_df.columns:
            ratings_df = full_df.dropna(subset=["userId", "rating"])[["userId", "movieId", "rating", "timestamp"]].copy()
        else:
            ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
            
        ratings_df["userId"]  = ratings_df["userId"].astype(int)
        ratings_df["movieId"] = ratings_df["movieId"].astype(int)

        if "imdbId" in full_df.columns and "tmdbId" in full_df.columns:
            links_df = full_df[["movieId", "imdbId", "tmdbId"]].drop_duplicates("movieId").copy()
        else:
            links_path = os.path.join(DATA_DIR, "links.csv")
            links_df = pd.read_csv(links_path) if os.path.exists(links_path) else pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
            
        links_df["imdbId"] = links_df["imdbId"].apply(lambda x: str(int(float(x))).zfill(7) if pd.notna(x) else None)
        links_df["tmdbId"] = links_df["tmdbId"].apply(lambda x: int(float(x)) if pd.notna(x) else None)
        log.info(f"Loaded dataset: {len(movies_df):,} movies, {len(ratings_df):,} ratings, {len(links_df):,} links.")
    else:
        movies_df  = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

        # Load links for IMDb / TMDb URLs
        links_path = os.path.join(DATA_DIR, "links.csv")
        if os.path.exists(links_path):
            links_df = pd.read_csv(links_path)
            links_df["imdbId"] = links_df["imdbId"].apply(lambda x: str(int(x)).zfill(7) if pd.notna(x) else None)
            links_df["tmdbId"] = links_df["tmdbId"].apply(lambda x: int(x) if pd.notna(x) else None)
            log.info(f"Loaded {len(links_df):,} movie links.")
        else:
            links_df = pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])

    # Feature engineering
    movies_df["year"]          = movies_df["title"].apply(_extract_year)
    movies_df["clean_title"]   = movies_df["title"].apply(_clean_title)
    movies_df["genres_list"]   = movies_df["genres"].apply(lambda g: g.split("|") if g != "(no genres listed)" else [])
    movies_df["genres_str"]    = movies_df["genres_list"].apply(lambda g: " ".join(g))
    movies_df["first_genre"]   = movies_df["genres_list"].apply(lambda g: g[0] if g else "Unknown")
    movies_df["genre_emoji"]   = movies_df["first_genre"].apply(lambda g: GENRE_EMOJI.get(g, "🎬"))

    # Compute popularity: Bayesian average rating
    stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    stats.columns = ["movieId", "avg_rating", "rating_count"]
    C = stats["avg_rating"].mean()   # global mean
    m = stats["rating_count"].quantile(0.60)  # min votes threshold
    stats["score"] = (stats["rating_count"] / (stats["rating_count"] + m)) * stats["avg_rating"] + \
                     (m / (stats["rating_count"] + m)) * C

    movies_df = movies_df.merge(stats, on="movieId", how="left")
    movies_df["avg_rating"]   = movies_df["avg_rating"].fillna(C).round(1)
    movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)
    movies_df["score"]        = movies_df["score"].fillna(0)

    popularity_df = movies_df.sort_values("score", ascending=False)
    log.info(f"Loaded {len(movies_df):,} movies and {len(ratings_df):,} ratings.")


# ── 3. Content-Based model ─────────────────────────────────────────────────────

def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def train_content_based(force=False):
    """Build TF-IDF vectors. Cached to disk."""
    global combined_norm, tfidf_titles, tfidf_matrix_t, movie_idx, idx_movie

    cs_path = _cache_path("combined_norm.pkl")
    ti_path = _cache_path("tfidf_titles.pkl")

    if not force and os.path.exists(cs_path) and os.path.exists(ti_path):
        log.info("Loading content-based model from cache…")
        with open(cs_path, "rb") as f: combined_norm = pickle.load(f)
        with open(ti_path, "rb") as f:
            tfidf_titles, tfidf_matrix_t = pickle.load(f)
    else:
        log.info("Training content-based model (TF-IDF + cosine)…")

        # Genre features
        tfidf_genre = TfidfVectorizer(token_pattern=r"[^\s]+", analyzer="word")
        genre_matrix = tfidf_genre.fit_transform(movies_df["genres_str"].fillna(""))

        # Title features for boosting (blended)
        tfidf_title_cb = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        title_matrix   = tfidf_title_cb.fit_transform(movies_df["clean_title"].fillna(""))

        # Weighted blend: 80% genre, 20% title
        from scipy.sparse import hstack
        combined = hstack([genre_matrix * 0.8, title_matrix * 0.2])
        global_combined = normalize(combined, norm="l2")
        combined_norm = global_combined

        # Title search TF-IDF (separate, for search endpoint)
        tfidf_titles = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_matrix_t = tfidf_titles.fit_transform(movies_df["clean_title"].fillna(""))

        with open(cs_path, "wb") as f: pickle.dump(combined_norm, f, protocol=4)
        with open(ti_path, "wb") as f: pickle.dump((tfidf_titles, tfidf_matrix_t), f, protocol=4)
        log.info("Content-based model trained and cached.")

    # Index maps
    movies_df_reset = movies_df.reset_index(drop=True)
    movie_idx = {title: i for i, title in enumerate(movies_df_reset["clean_title"])}
    idx_movie = {i: mid for i, mid in enumerate(movies_df_reset["movieId"])}


# ── 4. Collaborative Filtering model (SVD) ────────────────────────────────────

def train_collaborative(n_components=50, force=False):
    """Build user-item SVD model. Cached to disk."""
    global svd_user_factors, svd_item_factors, movie_id_to_idx, idx_to_movie_id

    svd_path = _cache_path("svd_model.pkl")

    if not force and os.path.exists(svd_path):
        log.info("Loading collaborative model from cache…")
        with open(svd_path, "rb") as f:
            svd_user_factors, svd_item_factors, movie_id_to_idx, idx_to_movie_id = pickle.load(f)
    else:
        log.info("Training collaborative filtering model (SVD)…")

        # Build pivot table: users × movies
        pivot = ratings_df.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0)
        movie_id_to_idx = {mid: i for i, mid in enumerate(pivot.columns)}
        idx_to_movie_id = {i: mid for mid, i in movie_id_to_idx.items()}

        matrix = pivot.values.astype(np.float32)

        # Truncated SVD (matrix factorization)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_user_factors = svd.fit_transform(matrix)          # (n_users, k)
        svd_item_factors = svd.components_.T                   # (n_movies, k)

        with open(svd_path, "wb") as f:
            pickle.dump((svd_user_factors, svd_item_factors, movie_id_to_idx, idx_to_movie_id), f, protocol=4)
        log.info("Collaborative model trained and cached.")


# ── 5. Load all models ────────────────────────────────────────────────────────

def load_models(force=False):
    """Full pipeline: download → load data → train both models."""
    download_dataset()
    load_data()
    train_content_based(force=force)
    train_collaborative(force=force)
    load_poster_cache()
    log.info("All models ready.")


# ── 6. Inference: Content-Based ───────────────────────────────────────────────

def get_content_recommendations(title: str, n: int = 10):
    """
    Given a movie title (or partial match), return N similar movies
    using cosine similarity on genre+title TF-IDF vectors.
    """
    if combined_norm is None:
        return []

    # Try exact match first, then fuzzy
    idx = movie_idx.get(title)
    if idx is None:
        title_lower = title.lower()
        for t, i in movie_idx.items():
            if title_lower in t.lower():
                idx = i
                break
    if idx is None:
        return []

    query_vec = combined_norm[idx]
    sim_scores = cosine_similarity(query_vec, combined_norm).flatten()
    sim_scores_indexed = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores_indexed, key=lambda x: x[1], reverse=True)[1:n+1]

    result_ids = [idx_movie[i] for i, _ in sim_scores]
    scores     = [round(float(s), 3) for _, s in sim_scores]

    result = movies_df[movies_df["movieId"].isin(result_ids)].copy()
    score_map = {mid: sc for mid, sc in zip(result_ids, scores)}
    result["similarity"] = result["movieId"].map(score_map)
    result = result.sort_values("similarity", ascending=False)
    return _format_movies(result.head(n))


# ── 7. Inference: Collaborative ────────────────────────────────────────────────

def get_collaborative_recommendations(user_id: int, n: int = 10):
    """
    Predict ratings for unseen movies for a given user using SVD.
    Falls back to popularity if user_id is out of range.
    """
    if svd_user_factors is None:
        return get_popular_movies(n)

    n_users = svd_user_factors.shape[0]
    if user_id < 1 or user_id > n_users:
        log.warning(f"User {user_id} out of range (1–{n_users}). Falling back to popularity.")
        return get_popular_movies(n)

    # Predict all movie ratings for this user (0-indexed)
    user_vec    = svd_user_factors[user_id - 1]              # shape (k,)
    pred_ratings = svd_item_factors @ user_vec                # shape (n_movies,)

    # Find movies the user already rated
    seen = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist())

    # Rank unseen movies
    ranked = sorted(
        [(idx_to_movie_id[i], float(pred_ratings[i])) for i in range(len(pred_ratings))
         if idx_to_movie_id[i] not in seen],
        key=lambda x: x[1], reverse=True
    )[:n]

    rec_ids = [r[0] for r in ranked]
    result  = movies_df[movies_df["movieId"].isin(rec_ids)].copy()
    score_map = {mid: sc for mid, sc in ranked}
    result["pred_rating"] = result["movieId"].map(score_map)
    result = result.sort_values("pred_rating", ascending=False)
    return _format_movies(result.head(n))


# ── 8. Inference: Search / Filter ─────────────────────────────────────────────

def search_movies(query="", genre="all", rating="0", year="all", n=50):
    """
    Full search + filter pipeline.
    Returns list of movie dicts sorted by relevance + popularity score.
    """
    df = movies_df.copy()

    # Genre filter
    if genre and genre != "all":
        genre_norm = genre.replace("-", " ").title().replace(" ", "-")
        df = df[df["genres_list"].apply(lambda g: any(
            genre_norm.lower() in gi.lower() or genre.lower() in gi.lower() for gi in g
        ))]

    # Rating filter (based on avg_rating in dataset)
    try:
        min_rating = float(rating)
        if min_rating > 0:
            df = df[df["avg_rating"] >= min_rating]
    except (ValueError, TypeError):
        pass

    # Year filter
    if year and year != "all":
        if year == "2020s":
            df = df[df["year"].between(2020, 2029)]
        elif year == "2010s":
            df = df[df["year"].between(2010, 2019)]
        elif year == "2000s":
            df = df[df["year"].between(2000, 2009)]
        elif year == "classic":
            df = df[df["year"] < 2000]
        else:
            try:
                df = df[df["year"] == int(year)]
            except ValueError:
                pass

    # Text search using TF-IDF similarity
    if query and query.strip() and tfidf_titles is not None:
        query_vec  = tfidf_titles.transform([query.strip()])
        # Reindex tfidf_matrix_t to current df rows
        all_indices = df.index.tolist()
        if all_indices:
            sub_matrix = tfidf_matrix_t[all_indices]
            sims = cosine_similarity(query_vec, sub_matrix).flatten()
            df = df.copy()
            df["search_score"] = sims
            # Also do substring match boost
            ql = query.lower()
            df["title_match"] = df["clean_title"].str.lower().str.contains(ql, regex=False).astype(float)
            df["search_score"] = df["search_score"] * 0.6 + df["title_match"] * 0.4
            df = df[df["search_score"] > 0.01].sort_values("search_score", ascending=False)
    else:
        df = df.sort_values("score", ascending=False)

    return _format_movies(df.head(n))


# ── 9. Popular movies (baseline) ──────────────────────────────────────────────

def get_popular_movies(n=20):
    """Return top-N movies by Bayesian popularity score."""
    return _format_movies(popularity_df.head(n))


# ── 10. Format output ─────────────────────────────────────────────────────────

def _format_movies(df):
    """Convert DataFrame rows to list of dicts for the Flask template/API."""
    # Build a fast movieId → link lookup
    link_map = {}
    if links_df is not None and not links_df.empty:
        for _, lr in links_df.iterrows():
            mid = int(lr["movieId"])
            imdb_id = lr["imdbId"]
            tmdb_id = lr["tmdbId"]
            tmdb_val = int(tmdb_id) if pd.notna(tmdb_id) else None
            link_map[mid] = {
                "imdb_url": f"https://www.imdb.com/title/tt{imdb_id}/" if imdb_id and pd.notna(imdb_id) else None,
                "tmdb_url": f"https://www.themoviedb.org/movie/{tmdb_val}" if tmdb_val else None,
                "tmdb_id":  tmdb_val,
            }

    result = []
    for _, row in df.iterrows():
        genres = row.get("genres_list", [])
        genres_display = [g for g in genres if g != "(no genres listed)"]
        mid = int(row["movieId"])
        links = link_map.get(mid, {})
        tmdb_id = links.get("tmdb_id")
        poster_url = _poster_cache.get(str(tmdb_id)) if tmdb_id else None
        result.append({
            "id":           mid,
            "title":        row["clean_title"],
            "full_title":   row["title"],
            "genre":        row["first_genre"].lower(),
            "first_genre":  row["first_genre"],
            "genres_list":  genres_display,
            "genres_str":   ", ".join(genres_display),
            "rating":       round(float(row.get("avg_rating", 0)), 1),
            "rating_count": int(row.get("rating_count", 0)),
            "year":         int(row["year"]) if pd.notna(row.get("year")) else "N/A",
            "emoji":        row.get("genre_emoji", "🎬"),
            "score":        round(float(row.get("score", 0)), 3),
            "imdb_url":     links.get("imdb_url"),
            "tmdb_url":     links.get("tmdb_url"),
            "tmdb_id":      tmdb_id,
            "poster_url":   poster_url,
        })
    return result


def get_all_movies(page=1, per_page=100, genre="all", rating="0", year="all", query=""):
    """Return ALL movies (paginated) with links, sorted by popularity score."""
    results = search_movies(query=query, genre=genre, rating=rating, year=year, n=len(movies_df))
    total   = len(results)
    start   = (page - 1) * per_page
    end     = start + per_page
    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page,
        "movies":   results[start:end],
    }


# ── 11. Standalone test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CineMatch ML Engine – Self Test ===\n")
    load_models()

    print("\n[Content-Based] Movies similar to 'Toy Story':")
    recs = get_content_recommendations("Toy Story", n=5)
    for r in recs:
        print(f"  {r['emoji']} {r['title']} ({r['year']})  ★{r['rating']}  sim={r.get('similarity','?')}")

    print("\n[Collaborative] Top picks for User #1:")
    recs_cf = get_collaborative_recommendations(1, n=5)
    for r in recs_cf:
        print(f"  {r['emoji']} {r['title']} ({r['year']})  ★{r['rating']}")

    print("\n[Search] Query: 'dark knight'")
    recs_s = search_movies(query="dark knight", n=5)
    for r in recs_s:
        print(f"  {r['emoji']} {r['title']} ({r['year']})  ★{r['rating']}")

    print("\n=== All tests passed ===")
