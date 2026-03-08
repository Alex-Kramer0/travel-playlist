# Travel Playlist

Personalized travel playlists that pair Airbnb stays with emotion- and location-aware Spotify tracks. Users will authenticate with Spotify, drop in an Airbnb listing (URL or description), and receive a curated playlist they can save directly to their account via our Streamlit frontend.

## Team
- Alex Kramer
- Kreena Totala

## Problem Statement & Objectives
Travelers often browse Airbnb listings that convey a mood or local vibe, but translating that feeling into music is manual and subjective. We aim to:
1. Understand a traveler’s desired vibe from an Airbnb listing description (or URL-derived data).
2. Match that vibe to 550K+ Spotify tracks with lyrics, audio features, mood labels, and popularity signals.
3. Let authenticated Spotify users preview and save the generated playlist instantly.

## Datasets
- **Spotify Lyrics & Audio Dataset** (`spotify/dataset/spotify_dataset_lyrics.csv`, 551k rows, 39 raw columns → 21 cleaned). Key columns: lyrics (text), emotion, genre, popularity, standard audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, explicit, etc.). Loudness converted to floats and track length parsed into seconds.
- **Airbnb Listings**: Consolidated CSVs (e.g., `Airbnb/Airbnb City Extracts/test-city-1.csv` for Bozeman, MT). Retained fields: id, name, description, neighborhood, property_type, room_type, accommodates, amenities. Companion NLP notebook extracts TF-IDF keywords from listing descriptions.

## Methods & Models
- **Data Preparation** (`spotify/data_loader.py`): column cleanup, audio feature selection, outlier removal, scaling (`StandardScaler`).
- **Clustering & Embeddings** (`spotify/clustering.py`, `recommendation/keyword_embedder.py`): k-means clustering on scaled audio features; PCA for visualization; sentence-transformer embeddings (all-MiniLM-L6-v2) to map free-form keywords to emotions, locations, and audio targets.
- **Recommendation Pipeline** (`recommendation/recommender.py`): multi-layer scoring (lyrics keyword match, emotion cosine similarity, audio similarity, cluster boost) with tunable weights to produce ranked playlists.
- **Spotify API Integrations** (`spotify-api-integrations/`):
  - `auth.py` PKCE helper for Streamlit-friendly login + token exchange.
  - `playlists.py` utilities to create playlists and add tracks.
  - `listening_history.py` (top-artist/genre helper) to derive user genre preferences via `/v1/me/top/artists`.

## How to Run
1. **Clone & Environment**
   ```bash
   git clone <repo-url>
   cd travel-playlist
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install pandas scikit-learn matplotlib seaborn nltk sentence-transformers tqdm ipykernel streamlit requests python-dotenv
   ```
2. **Register a Spotify App** (https://developer.spotify.com/dashboard) and add your redirect URI (e.g., `http://localhost:8501/callback`).
3. **Create a `.env` file** (or export variables) with:
   ```
   SPOTIFY_CLIENT_ID=...
   SPOTIFY_CLIENT_SECRET=...
   SPOTIFY_REDIRECT_URI=http://localhost:8501/callback
   ```
4. **Run Data/Model Notebooks**
   - `spotify/spotify_clustering.ipynb` for feature prep & clustering diagnostics.
   - `recommendation/playlist_generation.ipynb` for keyword resolution + recommendation demos.
5. **Streamlit Frontend (coming online)**
   - `streamlit run app.py` (placeholder; will orchestrate user login, Airbnb input, playlist preview, and save-to-Spotify actions).

## Assumptions & Limitations
- Spotify dataset emotions/genres come from provided labels; quality varies across tracks.
- Airbnb NLP currently tuned on a single test city; broader generalization requires more listings.
- Tokens are stored in-session for demos; long-term deployments need encrypted storage.
- Playlist personalization emphasizes lyrics/emotion alignment; live audio analysis is out-of-scope.

## Current Progress & Next Steps
**Progress**
- Cleaned and scaled Spotify corpus; established clustering + PCA basis.
- Built keyword embedding resolver and four-layer recommendation engine.
- Added Spotify auth, playlist, and top-genre helper modules for frontend integration.

**Next Steps**
1. Implement Streamlit UI: Spotify login, Airbnb input form, playlist preview/download.
2. Integrate genre + Airbnb keyword signals into recommendation weights.
3. Dockerize the full stack and deploy to a public endpoint (AWS or similar).
4. Add automated tests + linting to lock down pipelines before deployment.