"""
app.py – Movies Flask backend (ML-powered)
==============================================
Routes:
  GET  /                               → home: top popular movies
  GET  /search                         → search + filter
  GET  /all-movies                     → full dataset browser with IMDb/TMDb links
  GET  /api/recommend/<title>           → content-based similar movies (JSON)
  GET  /api/collaborative/<user_id>     → collaborative recommendations (JSON)
  GET  /api/popular                     → top popular movies (JSON)
  GET  /api/movies                      → filtered movies (JSON)
  GET  /api/genres                      → list of all genres
  GET  /api/all-movies                  → all movies paginated with links (JSON)
"""

import logging
import os
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
import ml_model as ml

logging.basicConfig(level=logging.INFO, format="[App] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── CORS: allow Vite dev server (port 5173) ──────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response

# Custom Jinja2 filter
@app.template_filter('format_number')
def format_number(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)

# ── Boot: load dataset + train models ────────────────────────────────────────
log.info("Starting Movies – loading ML models…")
ml.load_models()
log.info("Ready!")

# ── Helper ────────────────────────────────────────────────────────────────────

def _get_filter_params():
    """Extract common filter params from request.args."""
    return {
        "query":  request.args.get("query",  "").strip(),
        "genre":  request.args.get("genre",  "all"),
        "rating": request.args.get("rating", "0"),
        "year":   request.args.get("year",   "all"),
    }


# ── HTML Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Home page – prompt user to pick filters before showing movies."""
    return render_template(
        "index.html",
        movies=None,
        applied_filters=False,
        genre_emoji=ml.GENRE_EMOJI,
        page_title="Movie Recommendation",
        total_movies=len(ml.movies_df) if ml.movies_df is not None else 0,
    )


@app.route("/search")
def search():
    """Search & filter page."""
    p = _get_filter_params()
    results = ml.search_movies(
        query=p["query"],
        genre=p["genre"],
        rating=p["rating"],
        year=p["year"],
        n=48,
    )
    genre_display = f"{p['genre']} Movies" if p["genre"] != "all" else "All Filtered Movies"
    return render_template(
        "index.html",
        movies=results,
        applied_filters=True,
        genre_emoji=ml.GENRE_EMOJI,
        query=p["query"],
        selected_genre=p["genre"],
        selected_rating=p["rating"],
        selected_year=p["year"],
        result_count=len(results),
        page_title=genre_display,
        total_movies=len(ml.movies_df) if ml.movies_df is not None else 0,
    )

@app.route("/all-movies")
def all_movies():
    """Full dataset browser with IMDb/TMDb links."""
    p = _get_filter_params()
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 100))
    data = ml.get_all_movies(
        page=page,
        per_page=per_page,
        query=p["query"],
        genre=p["genre"],
        rating=p["rating"],
        year=p["year"],
    )
    return render_template(
        "all_movies.html",
        movies=data["movies"],
        total=data["total"],
        page=page,
        per_page=per_page,
        pages=data["pages"],
        genre_emoji=ml.GENRE_EMOJI,
        query=p["query"],
        selected_genre=p["genre"],
        selected_rating=p["rating"],
        selected_year=p["year"],
        total_movies=len(ml.movies_df) if ml.movies_df is not None else 0,
    )



# ── JSON API Routes ───────────────────────────────────────────────────────────

@app.route("/api/recommend/<path:title>")
def api_recommend(title):
    """Content-based similar movies for a given title."""
    n = int(request.args.get("n", 10))
    recs = ml.get_content_recommendations(title, n=n)
    return jsonify({
        "algorithm":    "content-based (TF-IDF + cosine similarity)",
        "based_on":     title,
        "count":        len(recs),
        "movies":       recs,
    })


@app.route("/api/collaborative/<int:user_id>")
def api_collaborative(user_id):
    """SVD collaborative filtering recommendations for a user ID."""
    n = int(request.args.get("n", 10))
    recs = ml.get_collaborative_recommendations(user_id, n=n)
    return jsonify({
        "algorithm":  "collaborative filtering (Truncated SVD)",
        "user_id":    user_id,
        "count":      len(recs),
        "movies":     recs,
    })


@app.route("/api/popular")
def api_popular():
    """Top popular movies by Bayesian score."""
    n = int(request.args.get("n", 20))
    movies = ml.get_popular_movies(n=n)
    return jsonify({"count": len(movies), "movies": movies})


@app.route("/api/movies")
def api_movies():
    """Filtered movie list (JSON version of /search)."""
    p = _get_filter_params()
    n = int(request.args.get("n", 50))
    results = ml.search_movies(
        query=p["query"],
        genre=p["genre"],
        rating=p["rating"],
        year=p["year"],
        n=n,
    )
    return jsonify({"count": len(results), "movies": results})


@app.route("/api/all-movies")
def api_all_movies():
    """All movies paginated, with IMDb + TMDb links."""
    p        = _get_filter_params()
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 100))
    data = ml.get_all_movies(
        page=page,
        per_page=per_page,
        query=p["query"],
        genre=p["genre"],
        rating=p["rating"],
        year=p["year"],
    )
    return jsonify(data)


@app.route("/api/genres")
def api_genres():
    """List of all unique genres in the dataset."""
    if ml.movies_df is None:
        return jsonify({"genres": []})
    all_genres = set()
    for g_list in ml.movies_df["genres_list"]:
        all_genres.update(g_list)
    all_genres.discard("(no genres listed)")
    sorted_genres = sorted(all_genres)
    return jsonify({
        "count":  len(sorted_genres),
        "genres": [{"name": g, "emoji": ml.GENRE_EMOJI.get(g, "🎬")} for g in sorted_genres],
    })


@app.route("/api/poster/<int:tmdb_id>")
def api_poster(tmdb_id):
    """Serve/redirect to TMDb movie poster image."""
    url = ml.get_poster_url(tmdb_id)
    if url:
        return redirect(url, code=302)
    return "", 404


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[Movies] Server starting...")
    print("[Movies] Open your browser at:  http://localhost:5000\n")
    app.run(debug=True, port=5000, use_reloader=False)
