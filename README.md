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

### 1. Clone & set up the environment
```bash
git clone <repo-url>
cd travel-playlist
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Register a Spotify App
Go to https://developer.spotify.com/dashboard and create an app. Under **Settings → Redirect URIs**, add:
```
http://127.0.0.1:8501
```
> **Note:** Spotify does not allow `localhost` — use `127.0.0.1` explicitly.

### 3. Configure environment variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8501
```
> `.env` is gitignored and must never be committed.

### 4. (Optional) Run data/model notebooks
- `spotify-clustering/spotify_clustering.ipynb` — feature prep, k-means clustering, PCA diagnostics.
- `recommendation/playlist_generation.ipynb` — keyword resolution + end-to-end recommendation demo.

### 5. Start the Streamlit app
```bash
.venv/bin/streamlit run app.py
```
The app will be available at **http://127.0.0.1:8501**.

From the homepage you can:
- **Connect to Spotify** — OAuth flow (PKCE); grants permission to create playlists in your account.
- **Continue without Spotify** — generate playlists and download as CSV without authenticating.

Then navigate to **Generate Playlist**, paste an Airbnb listing URL, and click **Generate**.

## Assumptions & Limitations
- Spotify dataset emotions/genres come from provided labels; quality varies across tracks.
- Airbnb NLP currently tuned on a single test city; broader generalization requires more listings.
- Tokens are stored in-session only; long-term deployments need encrypted persistent storage.
- Playlist personalization emphasizes lyrics/emotion alignment; live audio analysis is out-of-scope.
- The Spotify app must be in **Development Mode** with your account email added as a registered user (up to 25 users supported without going through the Spotify quota extension process).

## Current Progress & Next Steps
**Progress**
- Cleaned and scaled Spotify corpus (551k tracks); established clustering + PCA basis.
- Built keyword embedding resolver and four-layer recommendation engine (lyrics, emotion, audio, cluster).
- Implemented Spotify OAuth 2.0 with PKCE, persistent token exchange, and playlist save/export.
- Streamlit multi-page app live: homepage auth flow + Generate Playlist page.

**Next Steps**
1. Pre-compute lyric embeddings to speed up cold-start recommendations.
2. Integrate user listening history (top genres/artists) into recommendation weights.
3. Deploy to a public endpoint (Streamlit Community Cloud or similar).