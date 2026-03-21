"""Spotify playlist helpers.

Provides functions to create a playlist for the current user and to add tracks
once an access token (with appropriate scopes) is available.
Scopes required:
- ``playlist-modify-private`` and/or ``playlist-modify-public`` depending on the
  playlist visibility you need.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import requests

from auth import SpotifyAuthError

log = logging.getLogger(__name__)

ME_ENDPOINT = "https://api.spotify.com/v1/me"
CREATE_PLAYLIST_URL = "https://api.spotify.com/v1/me/playlists"
ADD_TRACKS_URL_TEMPLATE = "https://api.spotify.com/v1/playlists/{playlist_id}/items"
SEARCH_ENDPOINT = "https://api.spotify.com/v1/search"


def _get_current_user(access_token: str, timeout: int = 10) -> dict:
    response = requests.get(
        ME_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout,
    )
    if response.status_code != 200:
        log.error("GET /v1/me failed — HTTP %s: %s", response.status_code, response.text)
        raise SpotifyAuthError(
            f"Failed to fetch current user profile: {response.status_code} {response.text}"
        )
    log.info("GET /v1/me succeeded — HTTP %s", response.status_code)
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
    user_email = user_profile.get("email", "unknown")
    user_product = user_profile.get("product", "unknown")
    log.info("Authenticated user — id=%s, email=%s, product=%s", user_id, user_email, user_product)
    if not user_id:
        raise SpotifyAuthError("Spotify user profile response missing 'id'.")

    payload = {
        "name": name,
        "public": public,
        "description": description,
    }

    url = CREATE_PLAYLIST_URL
    log.info("POST %s — payload=%s", url, payload)
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code not in (200, 201):
        log.error("POST create-playlist failed — HTTP %s: %s", response.status_code, response.text)
        log.error("Response headers: %s", dict(response.headers))
        raise SpotifyAuthError(
            f"Failed to create playlist: {response.status_code} {response.text}"
        )
    log.info("POST create-playlist succeeded — HTTP %s — name=%r", response.status_code, name)
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

    url = ADD_TRACKS_URL_TEMPLATE.format(playlist_id=playlist_id)
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code not in (200, 201):
        log.error("POST add-tracks failed — HTTP %s: %s", response.status_code, response.text)
        raise SpotifyAuthError(
            f"Failed to add tracks: {response.status_code} {response.text}"
        )
    log.info("POST add-tracks succeeded — HTTP %s — %d URIs added to %s", response.status_code, len(uris), playlist_id)
    return response.json()


def search_track(
    access_token: str,
    track_name: str,
    artist: str,
    *,
    timeout: int = 10,
) -> Optional[str]:
    """Search Spotify for a track and return its URI, or None if not found."""

    query = f"track:{track_name} artist:{artist}"
    response = requests.get(
        SEARCH_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"q": query, "type": "track", "limit": 1},
        timeout=timeout,
    )
    if response.status_code != 200:
        log.warning("GET /v1/search failed — HTTP %s for query=%r", response.status_code, query)
        return None

    items = response.json().get("tracks", {}).get("items", [])
    if items:
        uri = items[0]["uri"]
        log.info("GET /v1/search — HTTP %s — found %r → %s", response.status_code, track_name, uri)
        return uri
    log.info("GET /v1/search — HTTP %s — no results for %r by %r", response.status_code, track_name, artist)
    return None


def resolve_track_uris(
    access_token: str,
    playlist_df,
    track_col: str = "track_name",
    artist_col: str = "artist",
) -> list[tuple[str, Optional[str]]]:
    """Look up Spotify URIs for each track in a playlist DataFrame.

    Returns a list of (track_name, uri_or_None) tuples.
    """
    results = []
    for _, row in playlist_df.iterrows():
        uri = search_track(access_token, row[track_col], row[artist_col])
        results.append((row[track_col], uri))
    return results
