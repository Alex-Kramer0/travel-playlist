"""
User router - Spotify user profile and preferences endpoints.
"""
from __future__ import annotations

import importlib
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import requests

# Import from hyphenated directory using importlib
listening_history = importlib.import_module('spotify-api-integrations.listening_history')
from backend.models import SpotifyUserProfile, UserGenresResponse, UserTopArtistsResponse

router = APIRouter()


@router.get("/profile", response_model=SpotifyUserProfile)
async def get_user_profile(authorization: str = Header(...)):
    """
    Returns the authenticated user's Spotify profile.
    
    Requires Authorization header with Bearer token.
    """
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        
        # Call Spotify API
        response = requests.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid or expired access token")
        elif response.status_code != 200:
            # Log the full error for debugging
            print(f"Spotify API error - Status: {response.status_code}")
            print(f"Response: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Spotify API error: {response.text}"
            )
        
        data = response.json()
        return SpotifyUserProfile(
            id=data["id"],
            display_name=data.get("display_name", ""),
            email=data.get("email"),
            images=data.get("images", []),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/genres", response_model=UserGenresResponse)
async def get_user_genres(
    authorization: str = Header(...),
    time_range: str = "medium_term",
):
    """
    Returns the user's top genre tags derived from their top artists.
    
    Query params:
    - time_range: short_term | medium_term | long_term
    """
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        
        # Validate time_range
        if time_range not in ["short_term", "medium_term", "long_term"]:
            raise HTTPException(
                status_code=400,
                detail="time_range must be one of: short_term, medium_term, long_term"
            )
        
        # Get top genres
        genres = listening_history.get_top_genres(
            access_token=access_token,
            time_range=time_range,
            limit_artists=20,
            top_n=10,
        )
        
        return UserGenresResponse(genres=genres)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-artists", response_model=UserTopArtistsResponse)
async def get_user_top_artists(
    authorization: str = Header(...),
    time_range: str = "medium_term",
):
    """
    Returns the user's top artist names derived from their listening history.
    
    Query params:
    - time_range: short_term | medium_term | long_term
    """
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        
        if time_range not in ["short_term", "medium_term", "long_term"]:
            raise HTTPException(
                status_code=400,
                detail="time_range must be one of: short_term, medium_term, long_term"
            )
        
        artists = listening_history.get_top_artist_names(
            access_token=access_token,
            time_range=time_range,
            limit_artists=20,
        )
        
        return UserTopArtistsResponse(artists=artists)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
