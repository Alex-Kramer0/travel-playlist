# Travel Playlist — React + FastAPI Integration Plan

## Overview

This document describes the plan to evolve the project from a notebook/script-based prototype into a deployable full-stack web application. The backend will be a **FastAPI** server that wraps all existing Python model logic and Spotify API integrations. The frontend will be a **React (Vite + TypeScript)** application styled with **TailwindCSS** and **shadcn/ui**.

---

## 1. Current State

| Module | Location | Purpose |
|--------|----------|---------|
| Airbnb NLP pipeline | `Airbnb/nlp_pipeline.py` | Listing URL → keywords + NRC emotion scores |
| Data loader | `spotify-clustering/data_loader.py` | Load + clean + scale Spotify dataset |
| Clustering | `spotify-clustering/clustering.py` | K-means / DBSCAN fit, PCA basis |
| Keyword embedder | `recommendation/keyword_embedder.py` | Keywords → emotions + audio target (sentence-transformers + zero-shot NLI) |
| Recommender | `recommendation/recommender.py` | 4-layer scoring → ranked playlist DataFrame |
| Spotify auth | `spotify-api-integrations/auth.py` | PKCE OAuth2 flow helpers |
| Playlists | `spotify-api-integrations/playlists.py` | Create playlist, add tracks |
| Listening history | `spotify-api-integrations/listening_history.py` | Top artists/genres |

All existing modules are **reused as-is** — no model code is rewritten. FastAPI simply wraps them in HTTP endpoints.

---

## 2. Target Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    React Frontend                         │
│  (Vite + TypeScript + TailwindCSS + shadcn/ui)           │
│  localhost:5173  (dev)  /  CDN or static host (prod)     │
└───────────────────────┬──────────────────────────────────┘
                        │  HTTP (REST JSON)
┌───────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                         │
│  localhost:8000  (dev)                                    │
│                                                           │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌────────┐ │
│  │   auth   │  │  recommend │  │ playlist │  │  user  │ │
│  │  router  │  │   router   │  │  router  │  │ router │ │
│  └──────────┘  └────────────┘  └──────────┘  └────────┘ │
│                                                           │
│  ┌───────────────────────────────────────────────────┐   │
│  │              App State (loaded at startup)        │   │
│  │  filtered_df · scaled_df · kmeans · lyric_index   │   │
│  │  nrc_lexicon                                      │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                        │
           External: Spotify Web API
```

---

## 3. FastAPI Backend

### 3.1 Directory Layout

```
backend/
  main.py            # FastAPI app; lifespan startup hook
  state.py           # Module-level shared state (loaded data + models)
  models.py          # Pydantic request/response schemas
  routers/
    auth.py          # /api/auth/* — Spotify PKCE flow
    recommend.py     # /api/recommend/* — playlist generation
    playlist.py      # /api/playlist/* — save to Spotify
    user.py          # /api/user/* — profile + top genres
```

### 3.2 Startup Sequence (`main.py` lifespan)

Heavy objects are loaded once at startup and shared across all requests via `state.py`.

```
1. data_loader.build_dataset(SPOTIFY_DATASET_PATH)
       → raw_df, filtered_df, feature_df, scaled_df, scaler

2. clustering.fit_kmeans(scaled_df, k=K_CLUSTERS)
       → kmeans_model, cluster_labels

3. filtered_df["cluster"] = cluster_labels

4. keyword_embedder.build_lyric_index(filtered_df, scaled_df, AUDIO_FEATURE_COLS)
       → lyric embedding index cached in keyword_embedder module globals

5. nlp_pipeline.load_nrc_lexicon(NRC_PATH)
       → cat_to_words dict stored in state
```

Environment variables required:
```
SPOTIFY_DATASET_PATH   # path to spotify_dataset.csv
NRC_PATH               # path to NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
AIRBNB_DATASET_PATH    # path to Airbnb/data/Output.csv
SPOTIFY_CLIENT_ID
SPOTIFY_CLIENT_SECRET
SPOTIFY_REDIRECT_URI   # must match Spotify Dashboard (e.g. http://localhost:5173/callback)
K_CLUSTERS             # int, default 8
```

### 3.3 API Endpoint Specifications

#### Auth Router — `/api/auth`

---

**`GET /api/auth/start`**

Initiates the Spotify PKCE OAuth2 flow.

Calls `auth.start_spotify_auth(scopes=[...])` which generates a PKCE pair and authorization URL.

Response:
```json
{
  "authorization_url": "https://accounts.spotify.com/authorize?...",
  "state": "<random 32-char string>",
  "code_verifier": "<64-char PKCE verifier>"
}
```

The frontend **must** persist `state` and `code_verifier` in `sessionStorage` before redirecting the user to `authorization_url`.

Required Spotify scopes:
- `user-read-email`
- `user-read-private`
- `user-top-read`
- `playlist-modify-private`
- `playlist-modify-public`

---

**`POST /api/auth/callback`**

Exchanges the authorization code returned by Spotify for tokens.

Calls `auth.complete_spotify_auth(code, expected_state, provided_state, code_verifier)`.

Request body:
```json
{
  "code": "<auth code from Spotify>",
  "state": "<state returned by Spotify>",
  "expected_state": "<state stored by frontend>",
  "code_verifier": "<verifier stored by frontend>"
}
```

Response:
```json
{
  "access_token": "...",
  "refresh_token": "...",
  "expires_at": 1700000000.0,
  "scope": "user-read-email ...",
  "token_type": "Bearer"
}
```

---

#### Recommend Router — `/api/recommend`

All recommendation endpoints return the same response shape.

**Common response schema:**
```json
{
  "tracks": [
    {
      "track_name": "...",
      "artist": "...",
      "genre": "...",
      "emotion": "...",
      "cluster": 3,
      "score": 0.82,
      "score_lyrics": 0.0,
      "score_emotion": 1.0,
      "score_audio": 0.74,
      "score_cluster": 1.0,
      "danceability": 0.71,
      "energy": 0.65,
      "valence": 0.55
      // ... other audio features
    }
  ],
  "resolved": {
    "emotions": ["joy", "surprise"],
    "emotion_weights": { "joy": 0.61, "surprise": 0.18, ... },
    "audio_target": { "danceability": 0.4, "energy": 0.2, ... },
    "location_terms": ["paris"]
  },
  "keywords_used": ["vibrant", "cozy", "paris"]
}
```

---

**`POST /api/recommend/from-keywords`**

Accepts a pre-formed keyword list.

Request body:
```json
{
  "keywords": ["vibrant", "cozy", "mountain view"],
  "top_n": 20,
  "weights": {
    "lyrics": 0.35,
    "emotion": 0.25,
    "audio": 0.25,
    "cluster": 0.15
  }
}
```

Calls `recommender.recommend(keywords, filtered_df, scaled_df, top_n, weights)`.

---

**`POST /api/recommend/from-description`**

Accepts raw listing description text. Extracts keywords via the Airbnb NLP pipeline, then passes them to the recommender.

Request body:
```json
{
  "description": "Stunning mountain retreat with cozy fireplace...",
  "top_n": 20,
  "weights": { ... }
}
```

Pipeline:
1. `nlp_pipeline.extract_vibe_keywords(description, top_n=10)` → keywords list
2. `nlp_pipeline.extract_emotions(description, cat_to_words)` → NRC emotion scores (surfaced in response)
3. `recommender.recommend(keywords, ...)` → tracks

---

**`POST /api/recommend/from-url`**

Accepts an Airbnb listing URL. Looks up the listing in the local dataset, extracts keywords, recommends.

Request body:
```json
{
  "url": "https://www.airbnb.com/rooms/2536175",
  "top_n": 20,
  "weights": { ... }
}
```

Pipeline:
1. `nlp_pipeline.analyze_listing_from_url(url, AIRBNB_DATASET_PATH, NRC_PATH)` → `(keyword_json, emotion_json)`
2. `recommender.recommend(keyword_json["keywords"], ...)` → tracks

Returns a 404 error if the URL is not found in the local dataset.

---

#### User Router — `/api/user`

---

**`GET /api/user/profile`**

Returns the authenticated user's Spotify profile. Calls `GET https://api.spotify.com/v1/me`.

Headers: `Authorization: Bearer <access_token>`

Response:
```json
{
  "id": "spotifyuser123",
  "display_name": "Alex",
  "email": "alex@example.com",
  "images": [{ "url": "https://..." }]
}
```

---

**`GET /api/user/genres`**

Returns the user's top genre tags derived from their top artists.

Headers: `Authorization: Bearer <access_token>`
Query params: `time_range=medium_term` (short_term | medium_term | long_term)

Calls `listening_history.get_top_genres(access_token, ...)`.

Response:
```json
{
  "genres": [["indie pop", 5], ["alternative rock", 3], ...]
}
```

---

#### Playlist Router — `/api/playlist`

---

**`POST /api/playlist/save`**

Creates a new Spotify playlist and populates it with the recommended tracks.

Request body:
```json
{
  "access_token": "...",
  "name": "Mountain Retreat Vibes",
  "description": "Generated from Airbnb listing: ...",
  "track_uris": ["spotify:track:abc123", ...],
  "public": false
}
```

Pipeline:
1. `playlists.create_playlist(access_token, name, description=description, public=public)`
2. `playlists.add_tracks_to_playlist(access_token, playlist_id, track_uris)`

**Note:** Current Spotify dataset tracks include `track_name` and `artist` but not Spotify URIs. A track search step using `GET /v1/search` must be added to resolve `(track_name, artist)` → `spotify:track:ID` before adding to the playlist.

Response:
```json
{
  "playlist_id": "37i9dQZF1DXcBWIGoYBM5M",
  "playlist_url": "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
  "tracks_added": 18
}
```

---

### 3.4 CORS Configuration

Enable CORS for the React dev server origin (`http://localhost:5173`) during development. Restrict to the deployed frontend domain in production.

---

## 4. React Frontend

### 4.1 Tech Stack

| Tool | Purpose |
|------|---------|
| React 18 + Vite | App framework and build tooling |
| TypeScript | Type safety |
| React Router v6 | Client-side routing |
| TailwindCSS | Utility-first styling |
| shadcn/ui | Accessible component primitives |
| Zustand | Lightweight global state (auth + playlist) |
| Axios | HTTP client with interceptors |
| Lucide React | Icon library |
| Recharts | Radar chart for audio feature visualization |

### 4.2 Directory Layout

```
frontend/
  index.html
  vite.config.ts
  tailwind.config.ts
  tsconfig.json
  package.json
  src/
    main.tsx
    App.tsx                    # Router setup, auth rehydration
    types/
      index.ts                 # Track, RecommendResponse, SpotifyToken, etc.
    api/
      client.ts                # Axios instance (baseURL, error handling)
      auth.ts                  # start, callback
      recommend.ts             # fromKeywords, fromDescription, fromUrl
      playlist.ts              # save
      user.ts                  # profile, genres
    store/
      authStore.ts             # Zustand: tokens, userProfile
      playlistStore.ts         # Zustand: tracks, resolved, isLoading, savedUrl
    pages/
      Home.tsx                 # Landing / login
      Callback.tsx             # OAuth callback handler
      Generate.tsx             # Input form + weight sliders
      Results.tsx              # Playlist display
    components/
      SpotifyLoginButton.tsx   # "Connect with Spotify" button
      AirbnbInputForm.tsx      # Tabbed input (URL / description / keywords)
      WeightSliders.tsx        # lyrics/emotion/audio/cluster weight controls
      TrackCard.tsx            # Single track result card
      ScoreBreakdownBar.tsx    # Mini bar showing 4-layer score breakdown
      EmotionBadge.tsx         # Colored pill for emotion label
      AudioRadarChart.tsx      # Recharts radar for audio feature target
      PlaylistSaveModal.tsx    # Name/description entry + save action
      LoadingSkeleton.tsx      # Skeleton cards during fetch
      Navbar.tsx               # Top nav with user avatar + logout
```

### 4.3 Routing

```
/                 Home           — unauthenticated landing page
/callback         Callback       — Spotify redirect target
/generate         Generate       — protected, requires auth token
/results          Results        — protected, requires tracks in store
```

Protected routes redirect unauthenticated users to `/`.

### 4.4 Page Specifications

---

#### `/` — Home

- **Hero:** app name "Travel Playlist", tagline ("Your Airbnb stay deserves a soundtrack"), brief description.
- **SpotifyLoginButton:** clicking it calls `GET /api/auth/start`, stores `state` + `code_verifier` in `sessionStorage`, then redirects `window.location.href` to `authorization_url`.
- **Background:** atmospheric travel/music imagery.

---

#### `/callback` — Callback

Runs on mount:
1. Parse `code` and `state` from `window.location.search`.
2. Read `expected_state` and `code_verifier` from `sessionStorage`.
3. Call `POST /api/auth/callback`.
4. On success: store tokens in `authStore` and `sessionStorage`; fetch user profile; redirect to `/generate`.
5. On error: display error message with link back to `/`.

---

#### `/generate` — Generate

Two-panel layout:

**Left panel — Input:**
- Tabbed input with three modes:
  - **Airbnb URL** — text input for a listing URL (`POST /api/recommend/from-url`)
  - **Paste Description** — textarea for raw listing text (`POST /api/recommend/from-description`)
  - **Keywords** — comma-separated keyword input (`POST /api/recommend/from-keywords`)
- `top_n` slider: 5–50 (default 20)
- Expandable "Advanced: Tune Weights" section with `WeightSliders` (lyrics / emotion / audio / cluster, must sum to 1.0)
- "Generate Playlist" button (disabled if input is empty)

**Right panel — Context:**
- User's top genres pulled from `GET /api/user/genres` displayed as badge chips
- Short copy explaining how the recommendation works

On submit: sets `playlistStore.isLoading = true`, calls appropriate endpoint, stores results, navigates to `/results`.

---

#### `/results` — Results

Three-section layout:

**Section 1 — Vibe Summary:**
- Detected keywords / location terms as badges
- Emotion breakdown horizontal bar chart (from `resolved.emotion_weights`)
- `AudioRadarChart`: radar chart comparing the audio target vector to the overall dataset average across the 11 audio features

**Section 2 — Track List:**
- Scrollable list of `TrackCard` components (one per recommended track)
- Each card: track name (bold), artist, genre tag, `EmotionBadge`, total score, `ScoreBreakdownBar` showing the four layer scores
- Track cards link out to Spotify search for the track

**Section 3 — Actions:**
- "Save to Spotify" button → opens `PlaylistSaveModal`
  - Pre-filled playlist name (e.g., "Mountain Retreat Vibes")
  - Editable description field
  - Calls `POST /api/playlist/save` with resolved Spotify track URIs
  - On success: shows confirmation with link to open the playlist in Spotify
- "Start Over" button → navigates back to `/generate`

---

### 4.5 State Management

**`authStore` (Zustand + sessionStorage persistence):**
```ts
interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  expiresAt: number | null;
  userProfile: SpotifyUserProfile | null;
  setTokens: (tokens: SpotifyToken) => void;
  setUserProfile: (profile: SpotifyUserProfile) => void;
  logout: () => void;
  isAuthenticated: () => boolean;
}
```

**`playlistStore` (Zustand, in-memory only):**
```ts
interface PlaylistState {
  tracks: Track[];
  resolved: ResolvedKeywords | null;
  keywordsUsed: string[];
  isLoading: boolean;
  savedPlaylistUrl: string | null;
  setResult: (result: RecommendResponse) => void;
  setLoading: (v: boolean) => void;
  setSavedUrl: (url: string) => void;
  reset: () => void;
}
```

On `App.tsx` mount: rehydrate `authStore` from `sessionStorage`.

---

### 4.6 TypeScript Types (`src/types/index.ts`)

```ts
interface Track {
  track_name: string;
  artist: string;
  genre: string;
  emotion: string;
  cluster: number;
  score: number;
  score_lyrics: number;
  score_emotion: number;
  score_audio: number;
  score_cluster: number;
  danceability: number;
  energy: number;
  loudness: number;
  speechiness: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  valence: number;
  tempo: number;
  duration_s: number;
  popularity: number;
}

interface ResolvedKeywords {
  emotions: string[];
  emotion_weights: Record<string, number>;
  audio_target: Record<string, number>;
  location_terms: string[];
}

interface RecommendResponse {
  tracks: Track[];
  resolved: ResolvedKeywords;
  keywords_used: string[];
}

interface SpotifyToken {
  access_token: string;
  refresh_token: string | null;
  expires_at: number;
  scope: string;
  token_type: string;
}

interface SpotifyUserProfile {
  id: string;
  display_name: string;
  email: string;
  images: { url: string }[];
}
```

---

## 5. End-to-End Data Flow

```
User pastes Airbnb URL
        │
        ▼
POST /api/recommend/from-url
        │
        ├─ nlp_pipeline.analyze_listing_from_url()
        │       └─ extract_vibe_keywords() → ["cozy", "mountain view", "fireplace"]
        │       └─ extract_emotions()       → NRC scores
        │
        ├─ keyword_embedder.resolve_keywords()
        │       └─ zero-shot NLI           → emotions: ["joy", "surprise"]
        │       └─ retrieve-then-aggregate → audio_target: {energy: -0.3, valence: +0.4, ...}
        │       └─ location detection      → location_terms: []
        │
        ├─ recommender.recommend()
        │       ├─ Layer 1: lyrics score   (location term search in lyrics)
        │       ├─ Layer 2: emotion score  (emotion column match)
        │       ├─ Layer 3: audio cosine   (scaled feature cosine similarity)
        │       └─ Layer 4: cluster boost  (closest K-means centroid)
        │
        └─ Return top-N tracks + resolved metadata
                │
                ▼
        React Results page renders track cards, radar chart, emotion bars
                │
        User clicks "Save to Spotify"
                │
                ▼
POST /api/playlist/save
        ├─ Spotify search: resolve (track_name, artist) → spotify:track:URI
        ├─ playlists.create_playlist()
        └─ playlists.add_tracks_to_playlist()
```

---

## 6. Implementation Phases

### Phase 1 — FastAPI Backend (no frontend changes needed)
- [x] Create `backend/` directory structure
- [x] Implement `backend/state.py` with shared data + model objects
- [x] Write `backend/main.py` with lifespan startup sequence
- [x] Add Pydantic schemas to `backend/models.py`
- [x] Implement `routers/auth.py` (start + callback endpoints)
- [x] Implement `routers/recommend.py` (3 input-mode endpoints)
- [x] Implement `routers/user.py` (profile + genres endpoints)
- [x] Implement `routers/playlist.py` (save endpoint including track URI resolution via Spotify Search API)
- [x] Configure CORS middleware
- [x] Add `backend/requirements.txt` (fastapi, uvicorn, python-dotenv, sentence-transformers, transformers, scikit-learn, pandas, nltk, requests)
- [ ] Validate all endpoints with manual curl / Postman tests

### Phase 2 — React Frontend Scaffold
- [x] Scaffold project: `npm create vite@latest frontend -- --template react-ts`
- [x] Install dependencies: tailwindcss, shadcn/ui, react-router-dom, zustand, axios, lucide-react, recharts
- [x] Configure TailwindCSS + shadcn/ui
- [x] Create `src/types/index.ts`
- [x] Implement `src/api/client.ts` (Axios instance pointing at backend)
- [x] Implement all API modules (`auth.ts`, `recommend.ts`, `playlist.ts`, `user.ts`)
- [x] Implement `authStore.ts` and `playlistStore.ts`

### Phase 3 — Frontend Pages
- [x] Build `Home.tsx` with hero + `SpotifyLoginButton`
- [x] Build `Callback.tsx` with token exchange + redirect
- [x] Build `Generate.tsx` with tabbed input + weight sliders
- [x] Build `Results.tsx` with track list + save modal (radar chart can be added later)
- [x] Wire routing in `App.tsx` with auth guards
- [x] Build `Navbar.tsx` with user avatar + logout (integrated into Generate and Results pages)

### Phase 4 — Integration Testing & Polish
- [ ] Test full flow: Spotify login → generate from URL → save playlist
- [ ] Test error states: URL not found, Spotify API failures, expired token
- [x] Add token expiry detection and logout on 401
- [x] Add loading skeletons and empty-state messaging
- [ ] Responsive design (mobile-friendly breakpoints)
- [x] Update `README.md` with new run instructions

---

## 7. Open Questions / Future Considerations

- **Track URI resolution:** The Spotify dataset contains track names and artists, but not Spotify track URIs. The save-to-Spotify flow requires a search step (`GET /v1/search?q=track:...+artist:...&type=track`). Some tracks may not be found — the frontend should show a warning if fewer tracks were added than recommended.
- **Token refresh:** The PKCE flow does not always return a `refresh_token`. A manual re-authentication prompt should appear when the token expires (or is absent) rather than silently failing.
- **Dataset availability:** The Spotify dataset CSV is large (~551K rows). Startup may take 30–90 seconds to load, embed, and index. A readiness endpoint (`GET /api/health`) should return `{"status": "ready"}` only after all startup steps complete, and the frontend should show a loading state while waiting.
- **Airbnb URL lookup:** The current implementation matches listing URLs against a local CSV. Expanding to a real web scraper or Airbnb API in the future would remove the dependency on pre-loaded city CSVs.
- **Deployment:** Backend can be containerized with Docker. Frontend can be built to static files (`npm run build`) and served from a CDN (Netlify, Vercel) or the same container via a static file route.
