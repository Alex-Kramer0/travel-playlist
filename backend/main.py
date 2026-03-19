"""
FastAPI main application.

Startup sequence:
1. Load Spotify dataset
2. Fit K-means clustering
3. Build lyric embedding index
4. Load NRC emotion lexicon
"""
from __future__ import annotations

import os
import sys

# Add project root to sys.path to enable imports of parent modules
# This must be done before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend import state
from backend.routers import auth, recommend, user, playlist

# Import existing modules (using importlib for hyphenated directory names)
import importlib
data_loader = importlib.import_module('spotify-clustering.data_loader')
clustering = importlib.import_module('spotify-clustering.clustering')
from Airbnb import nlp_pipeline
from recommendation import keyword_embedder

# Load .env file from backend directory
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle handler.
    Loads all heavy objects at startup and stores them in the state module.
    """
    print("=" * 80)
    print("Starting Travel Playlist Backend")
    print("=" * 80)
    
    # Get environment variables
    spotify_dataset_path = os.getenv("SPOTIFY_DATASET_PATH")
    nrc_path = os.getenv("NRC_PATH")
    k_clusters = int(os.getenv("K_CLUSTERS", "8"))
    
    if not spotify_dataset_path:
        raise ValueError("SPOTIFY_DATASET_PATH environment variable is required")
    if not nrc_path:
        raise ValueError("NRC_PATH environment variable is required")
    
    print(f"\n[1/4] Loading Spotify dataset from {spotify_dataset_path}")
    dataset = data_loader.build_dataset(spotify_dataset_path)
    state.raw_df = dataset["raw_df"]
    state.filtered_df = dataset["filtered_df"]
    state.feature_df = dataset["feature_df"]
    state.scaled_df = dataset["scaled_df"]
    state.scaler = dataset["scaler"]
    print(f"✓ Loaded {len(state.filtered_df):,} tracks")
    
    print(f"\n[2/4] Fitting K-means clustering with k={k_clusters}")
    kmeans, labels = clustering.fit_kmeans(state.scaled_df, k=k_clusters)
    state.kmeans_model = kmeans
    state.cluster_labels = labels
    state.filtered_df["cluster"] = labels
    print(f"✓ Clustering complete")
    
    print(f"\n[3/4] Building lyric embedding index")
    # Check if lyrics column exists
    if "lyrics" in state.filtered_df.columns:
        keyword_embedder.build_lyric_index(
            state.filtered_df,
            state.scaled_df,
            state.AUDIO_FEATURE_COLS
        )
        print(f"✓ Lyric index ready")
    else:
        print(f"⚠ Skipping lyric index (no lyrics column in dataset)")
        print(f"  Lyric-based recommendations will be disabled")
    
    print(f"\n[4/4] Loading NRC emotion lexicon from {nrc_path}")
    state.nrc_lexicon = nlp_pipeline.load_nrc_lexicon(nrc_path)
    print(f"✓ NRC lexicon loaded ({len(state.nrc_lexicon)} categories)")
    
    print("\n" + "=" * 80)
    print("Backend ready to serve requests")
    print("=" * 80 + "\n")
    
    yield
    
    # Cleanup (if needed)
    print("\nShutting down backend...")


app = FastAPI(
    title="Travel Playlist API",
    description="Generate Spotify playlists from Airbnb listings",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
origins = [
    "http://localhost:5173",  # React dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(recommend.router, prefix="/api/recommend", tags=["recommend"])
app.include_router(user.router, prefix="/api/user", tags=["user"])
app.include_router(playlist.router, prefix="/api/playlist", tags=["playlist"])


@app.get("/")
async def root():
    return {
        "message": "Travel Playlist API",
        "status": "ready",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health():
    """Health check endpoint that confirms all startup steps completed."""
    ready = all([
        state.filtered_df is not None,
        state.scaled_df is not None,
        state.kmeans_model is not None,
        state.nrc_lexicon is not None,
    ])
    return {
        "status": "ready" if ready else "loading",
        "tracks_loaded": len(state.filtered_df) if state.filtered_df is not None else 0,
    }
