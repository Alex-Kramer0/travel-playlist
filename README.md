# Travel Playlist

Personalized travel playlists that pair Airbnb stays with emotion- and location-aware Spotify tracks. Users authenticate with Spotify, drop in an Airbnb listing (URL or description), and receive a curated playlist they can save directly to their account via our React + FastAPI web application.

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

## Architecture

### Backend (FastAPI)
- **Data Preparation** (`spotify-clustering/data_loader.py`): column cleanup, audio feature selection, outlier removal, scaling (`StandardScaler`).
- **Clustering & Embeddings** (`spotify-clustering/clustering.py`, `recommendation/keyword_embedder.py`): k-means clustering on scaled audio features; sentence-transformer embeddings (all-MiniLM-L6-v2) to map keywords to emotions and audio targets.
- **Recommendation Pipeline** (`recommendation/recommender.py`): 4-layer scoring (lyrics keyword match, emotion cosine similarity, audio similarity, cluster boost) with tunable weights.
- **Spotify API Integrations** (`spotify-api-integrations/`):
  - `auth.py` PKCE OAuth2 flow for secure authentication
  - `playlists.py` utilities to create playlists and add tracks
  - `listening_history.py` top-artist/genre helper for user preferences
- **FastAPI Server** (`backend/`): REST API with routers for auth, recommendations, user data, and playlist management

### Frontend (React + TypeScript)
- **Vite + React 18**: Modern build tooling and component framework
- **TailwindCSS + shadcn/ui**: Beautiful, accessible UI components
- **Zustand**: Lightweight state management for auth and playlist data
- **React Router**: Client-side routing with protected routes
- **Axios**: HTTP client for API communication

## How to Run

### Prerequisites
- Python 3.11+
- Node.js 18+
- Spotify Developer Account

### Backend Setup
1. **Create Python Virtual Environment**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Create `backend/.env` file (see `backend/.env.example`):
   ```
   SPOTIFY_DATASET_PATH=../spotify-clustering/dataset/spotify_dataset.csv
   NRC_PATH=../Airbnb/emolex/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
   AIRBNB_DATASET_PATH=../Airbnb/data/Output.csv
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   SPOTIFY_REDIRECT_URI=http://localhost:5173/callback
   K_CLUSTERS=8
   ```

3. **Unzip Airbnb Dataset**
   ```bash
   cd Airbnb/data
   unzip Output.csv.zip
   ```

4. **Start Backend Server**
   
   **Option A: Using the startup script (recommended)**
   ```bash
   ./start_backend.sh
   ```
   
   **Option B: Manual command from project root**
   ```bash
   source backend/venv/bin/activate
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   uvicorn backend.main:app --reload --port 8000
   ```
   
   The backend will load the dataset and build indexes on startup (may take 1-2 minutes).

### Frontend Setup
1. **Install Dependencies**
   ```bash
   cd frontend
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

### Spotify App Configuration
1. Go to https://developer.spotify.com/dashboard
2. Create a new app
3. Add redirect URI: `http://localhost:5173/callback`
4. Copy Client ID and Client Secret to backend `.env` file

## Assumptions & Limitations
- Spotify dataset emotions/genres come from provided labels; quality varies across tracks.
- Airbnb NLP currently tuned on a single test city; broader generalization requires more listings.
- Tokens are stored in-session for demos; long-term deployments need encrypted storage.
- Playlist personalization emphasizes lyrics/emotion alignment; live audio analysis is out-of-scope.

## Current Progress & Next Steps
**Completed**
- Cleaned and scaled Spotify corpus; established clustering + PCA basis
- Built keyword embedding resolver and four-layer recommendation engine
- Implemented FastAPI backend with all endpoints (auth, recommend, user, playlist)
- Created React frontend with modern UI (TailwindCSS + shadcn/ui)
- Spotify PKCE OAuth2 authentication flow
- Full playlist generation and save-to-Spotify functionality

**Next Steps**
1. Test end-to-end flow with real Spotify credentials
2. Add error handling and loading states
3. Implement responsive design for mobile devices
4. Add automated tests for backend endpoints
5. Dockerize the full stack for deployment
6. Deploy to cloud platform (AWS, Vercel, or similar)