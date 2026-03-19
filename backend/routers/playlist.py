"""
Playlist router - Save playlist to Spotify endpoint.
"""
from __future__ import annotations

import importlib
from fastapi import APIRouter, HTTPException
import requests

# Import from hyphenated directory using importlib
playlists = importlib.import_module('spotify-api-integrations.playlists')
from backend.models import PlaylistSaveRequest, PlaylistSaveResponse

router = APIRouter()


def _search_track_uri(track_name: str, artist: str, access_token: str) -> str | None:
    """
    Search for a track on Spotify and return its URI.
    
    Returns None if track not found.
    """
    try:
        query = f"track:{track_name} artist:{artist}"
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "q": query,
                "type": "track",
                "limit": 1,
            },
            timeout=10,
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        tracks = data.get("tracks", {}).get("items", [])
        
        if not tracks:
            return None
        
        return tracks[0]["uri"]
    
    except Exception:
        return None


@router.post("/save", response_model=PlaylistSaveResponse)
async def save_playlist(request: PlaylistSaveRequest):
    """
    Creates a new Spotify playlist and populates it with the recommended tracks.
    
    Note: The request includes track_uris, but if they are in the format
    "track_name|artist", we need to resolve them via Spotify search first.
    """
    try:
        if not request.name or not request.name.strip():
            raise HTTPException(status_code=400, detail="Playlist name is required")
        
        if not request.track_uris:
            raise HTTPException(status_code=400, detail="At least one track is required")
        
        # Resolve track URIs if needed
        # If track_uris contain "|", they're in "track_name|artist" format
        resolved_uris = []
        for uri in request.track_uris:
            if "|" in uri:
                # Parse track_name|artist format
                parts = uri.split("|", 1)
                if len(parts) == 2:
                    track_name, artist = parts
                    spotify_uri = _search_track_uri(track_name, artist, request.access_token)
                    if spotify_uri:
                        resolved_uris.append(spotify_uri)
            elif uri.startswith("spotify:track:"):
                # Already a valid Spotify URI
                resolved_uris.append(uri)
        
        if not resolved_uris:
            raise HTTPException(
                status_code=400,
                detail="Could not resolve any tracks to Spotify URIs"
            )
        
        # Create playlist
        playlist_data = playlists.create_playlist(
            access_token=request.access_token,
            name=request.name,
            description=request.description,
            public=request.public,
        )
        
        playlist_id = playlist_data["id"]
        playlist_url = playlist_data["external_urls"]["spotify"]
        
        # Add tracks to playlist
        playlists.add_tracks_to_playlist(
            access_token=request.access_token,
            playlist_id=playlist_id,
            track_uris=resolved_uris,
        )
        
        return PlaylistSaveResponse(
            playlist_id=playlist_id,
            playlist_url=playlist_url,
            tracks_added=len(resolved_uris),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
