"""
Pydantic request/response schemas for the FastAPI backend.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Auth schemas ──────────────────────────────────────────────────────────────

class AuthStartResponse(BaseModel):
    authorization_url: str
    state: str
    code_verifier: str


class AuthCallbackRequest(BaseModel):
    code: str
    state: str
    expected_state: str
    code_verifier: str


class SpotifyTokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str]
    expires_at: float
    scope: str
    token_type: str


# ── Recommend schemas ─────────────────────────────────────────────────────────

class RecommendWeights(BaseModel):
    lyrics: float = 0.35
    emotion: float = 0.25
    audio: float = 0.25
    cluster: float = 0.15


class RecommendFromKeywordsRequest(BaseModel):
    keywords: list[str]
    top_n: int = 20
    weights: Optional[RecommendWeights] = None


class RecommendFromDescriptionRequest(BaseModel):
    description: str
    top_n: int = 20
    weights: Optional[RecommendWeights] = None


class RecommendFromUrlRequest(BaseModel):
    url: str
    top_n: int = 20
    weights: Optional[RecommendWeights] = None


class Track(BaseModel):
    track_name: str
    artist: str
    genre: str
    emotion: str
    cluster: int
    score: float
    score_lyrics: float
    score_emotion: float
    score_audio: float
    score_cluster: float
    danceability: float
    energy: float
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_s: float
    popularity: float


class ResolvedKeywords(BaseModel):
    emotions: list[str]
    emotion_weights: dict[str, float]
    audio_target: dict[str, float]
    location_terms: list[str]


class RecommendResponse(BaseModel):
    tracks: list[Track]
    resolved: ResolvedKeywords
    keywords_used: list[str]


# ── User schemas ──────────────────────────────────────────────────────────────

class SpotifyUserProfile(BaseModel):
    id: str
    display_name: str
    email: Optional[str] = None
    images: list[dict] = []


class UserGenresResponse(BaseModel):
    genres: list[list]  # list of [genre_name, count] pairs


# ── Playlist schemas ──────────────────────────────────────────────────────────

class PlaylistSaveRequest(BaseModel):
    access_token: str
    name: str
    description: str = ""
    track_uris: list[str]
    public: bool = False


class PlaylistSaveResponse(BaseModel):
    playlist_id: str
    playlist_url: str
    tracks_added: int
