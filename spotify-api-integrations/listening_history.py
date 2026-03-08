"""Spotify top-artist and genre helpers.

Provides wrappers around ``GET /v1/me/top/artists`` to quickly understand a
user's listening preferences and derive their dominant genres with minimal
post-processing.

Scopes required: ``user-top-read``.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import requests

from auth import SpotifyAuthError

TOP_ARTISTS_URL = "https://api.spotify.com/v1/me/top/artists"


def get_top_artists(
    access_token: str,
    *,
    limit: int = 10,
    time_range: str = "medium_term",
    offset: int = 0,
    timeout: int = 10,
) -> dict:
    """Fetch the user's top artists in the selected listening window."""

    if not access_token:
        raise SpotifyAuthError("Access token is required to fetch top artists.")
    if not 1 <= limit <= 50:
        raise ValueError("Spotify only allows 1-50 top artists per request.")
    if time_range not in {"short_term", "medium_term", "long_term"}:
        raise ValueError("time_range must be one of short_term, medium_term, long_term")

    params = {
        "limit": limit,
        "time_range": time_range,
        "offset": offset,
    }

    response = requests.get(
        TOP_ARTISTS_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise SpotifyAuthError(
            f"Failed to fetch top artists: {response.status_code} {response.text}"
        )
    return response.json()


def get_top_genres(
    access_token: str,
    *,
    limit_artists: int = 10,
    top_n: int = 5,
    time_range: str = "medium_term",
    timeout: int = 10,
) -> list[tuple[str, int]]:
    """Return the user's dominant genres based on their top artists."""

    data = get_top_artists(
        access_token,
        limit=limit_artists,
        time_range=time_range,
        timeout=timeout,
    )
    items: Iterable[dict] = data.get("items", [])
    counter: Counter[str] = Counter()
    for artist in items:
        for genre in artist.get("genres", []):
            if genre:
                counter[genre.lower()] += 1

    return counter.most_common(top_n)
