"""Spotify playlist helpers.

Provides functions to create a playlist for the current user and to add tracks
once an access token (with appropriate scopes) is available.
Scopes required:
- ``playlist-modify-private`` and/or ``playlist-modify-public`` depending on the
  playlist visibility you need.
"""
from __future__ import annotations

from typing import Iterable, Optional

import requests

from auth import SpotifyAuthError

ME_ENDPOINT = "https://api.spotify.com/v1/me"
CREATE_PLAYLIST_URL_TEMPLATE = "https://api.spotify.com/v1/users/{user_id}/playlists"
ADD_TRACKS_URL_TEMPLATE = "https://api.spotify.com/v1/playlists/{playlist_id}/tracks"


def _get_current_user(access_token: str, timeout: int = 10) -> dict:
    response = requests.get(
        ME_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise SpotifyAuthError(
            f"Failed to fetch current user profile: {response.status_code} {response.text}"
        )
    return response.json()


def create_playlist(
    access_token: str,
    name: str,
    *,
    public: bool = False,
    description: str = "",
    timeout: int = 10,
) -> dict:
    """Create a Spotify playlist for the authenticated user."""

    if not name:
        raise ValueError("Playlist name is required.")

    user_profile = _get_current_user(access_token, timeout=timeout)
    user_id = user_profile.get("id")
    if not user_id:
        raise SpotifyAuthError("Spotify user profile response missing 'id'.")

    payload = {
        "name": name,
        "public": public,
        "description": description,
    }

    response = requests.post(
        CREATE_PLAYLIST_URL_TEMPLATE.format(user_id=user_id),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code not in (200, 201):
        raise SpotifyAuthError(
            f"Failed to create playlist: {response.status_code} {response.text}"
        )
    return response.json()


def add_tracks_to_playlist(
    access_token: str,
    playlist_id: str,
    track_uris: Iterable[str],
    *,
    position: Optional[int] = None,
    timeout: int = 10,
) -> dict:
    """Add tracks to an existing playlist."""

    uris = [uri for uri in track_uris if uri]
    if not uris:
        raise ValueError("Provide at least one Spotify track URI to add.")

    payload = {"uris": uris}
    if position is not None:
        payload["position"] = position

    response = requests.post(
        ADD_TRACKS_URL_TEMPLATE.format(playlist_id=playlist_id),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code not in (200, 201):
        raise SpotifyAuthError(
            f"Failed to add tracks: {response.status_code} {response.text}"
        )
    return response.json()
