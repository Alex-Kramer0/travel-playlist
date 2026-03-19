"""
Auth router - Spotify PKCE OAuth2 flow endpoints.
"""
from __future__ import annotations

import importlib

from fastapi import APIRouter, HTTPException

# Import from hyphenated directory using importlib
spotify_auth = importlib.import_module('spotify-api-integrations.auth')
from backend.models import AuthStartResponse, AuthCallbackRequest, SpotifyTokenResponse

router = APIRouter()

REQUIRED_SCOPES = [
    "user-read-email",
    "user-read-private",
    "user-top-read",
    "playlist-modify-private",
    "playlist-modify-public",
]


@router.get("/start", response_model=AuthStartResponse)
async def start_auth():
    """
    Initiates the Spotify PKCE OAuth2 flow.
    
    Returns authorization URL, state, and code_verifier that the frontend
    must persist before redirecting the user.
    """
    try:
        flow = spotify_auth.start_spotify_auth(scopes=REQUIRED_SCOPES)
        return AuthStartResponse(
            authorization_url=flow.authorization_url,
            state=flow.state,
            code_verifier=flow.code_verifier,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/callback", response_model=SpotifyTokenResponse)
async def auth_callback(request: AuthCallbackRequest):
    """
    Exchanges the authorization code for Spotify tokens.
    
    Validates the state parameter and completes the PKCE flow.
    """
    try:
        token = spotify_auth.complete_spotify_auth(
            code=request.code,
            expected_state=request.expected_state,
            provided_state=request.state,
            code_verifier=request.code_verifier,
        )
        return SpotifyTokenResponse(
            access_token=token.access_token,
            refresh_token=token.refresh_token,
            expires_at=token.expires_at,
            scope=token.scope,
            token_type=token.token_type,
        )
    except spotify_auth.SpotifyAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
