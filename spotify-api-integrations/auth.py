"""Spotify Authentication Helpers.

This module exposes two functions that work well in a Streamlit (or any web)
frontend:

1. ``start_spotify_auth`` builds a PKCE-enabled authorization URL plus the
   generated ``state`` and ``code_verifier`` that the frontend should cache
   (e.g., ``st.session_state``) while redirecting the user to Spotify.
2. ``complete_spotify_auth`` validates the state that Spotify returns and then
   exchanges the authorization ``code`` for access/refresh tokens. The returned
   ``SpotifyToken`` can be stored securely (session memory for demos, encrypted
   persistence for production deployments).

Environment variables expected:
- ``SPOTIFY_CLIENT_ID`` (required)
- ``SPOTIFY_CLIENT_SECRET`` (optional when using PKCE; still required for
  server-to-server flows)
- ``SPOTIFY_REDIRECT_URI`` (optional if supplied programmatically)

References:
https://developer.spotify.com/documentation/web-api/concepts/authorization
"""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlencode

import requests

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
DEFAULT_SCOPES = (
    "user-read-email",
    "user-read-private",
)


@dataclass(slots=True)
class SpotifyAuthFlow:
    """Container for the data a frontend needs to start Spotify auth."""

    authorization_url: str
    state: str
    code_verifier: str
    code_challenge: str


@dataclass(slots=True)
class SpotifyToken:
    """Normalized token payload returned from Spotify."""

    access_token: str
    refresh_token: Optional[str]
    expires_at: float
    scope: str
    token_type: str
    raw_response: dict


class SpotifyAuthError(RuntimeError):
    """Raised when Spotify returns an error or validation fails."""


def _get_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value:
        return value
    raise SpotifyAuthError(
        f"Missing required Spotify configuration: set the '{name}' environment variable"
    )


def _generate_state(length: int = 32) -> str:
    return secrets.token_urlsafe(length)


def _build_pkce_pair(length: int = 64) -> tuple[str, str]:
    verifier = secrets.token_urlsafe(length)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return verifier, challenge


def start_spotify_auth(
    scopes: Optional[Iterable[str]] = None,
    *,
    redirect_uri: Optional[str] = None,
    client_id: Optional[str] = None,
    show_dialog: bool = False,
) -> SpotifyAuthFlow:
    """Create an authorization URL plus PKCE helpers.

    Parameters
    ----------
    scopes:
        Iterable of Spotify scopes. Falls back to ``DEFAULT_SCOPES`` if empty.
    redirect_uri:
        Overrides ``SPOTIFY_REDIRECT_URI`` env var.
    client_id:
        Overrides ``SPOTIFY_CLIENT_ID`` env var.
    show_dialog:
        If True, Spotify will always prompt even if the user already granted access.
    """

    scope_param = " ".join(scopes or DEFAULT_SCOPES)
    redirect = redirect_uri or _get_env("SPOTIFY_REDIRECT_URI")
    cid = client_id or _get_env("SPOTIFY_CLIENT_ID")
    state = _generate_state()
    code_verifier, code_challenge = _build_pkce_pair()

    query = urlencode(
        {
            "client_id": cid,
            "response_type": "code",
            "redirect_uri": redirect,
            "scope": scope_param,
            "state": state,
            "code_challenge_method": "S256",
            "code_challenge": code_challenge,
            "show_dialog": str(show_dialog).lower(),
        }
    )

    return SpotifyAuthFlow(
        authorization_url=f"{SPOTIFY_AUTH_URL}?{query}",
        state=state,
        code_verifier=code_verifier,
        code_challenge=code_challenge,
    )


def complete_spotify_auth(
    code: str,
    *,
    expected_state: str,
    provided_state: str,
    code_verifier: str,
    redirect_uri: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    timeout: int = 10,
) -> SpotifyToken:
    """Exchange an authorization code for Spotify tokens.

    Raises ``SpotifyAuthError`` when state validation fails or Spotify returns an
    error response.
    """

    if not code:
        raise SpotifyAuthError("Authorization code is required.")
    if expected_state != provided_state:
        raise SpotifyAuthError("State mismatch. Possible CSRF attack detected.")

    redirect = redirect_uri or _get_env("SPOTIFY_REDIRECT_URI")
    cid = client_id or _get_env("SPOTIFY_CLIENT_ID")
    secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")

    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect,
        "client_id": cid,
        "code_verifier": code_verifier,
    }
    if secret:
        payload["client_secret"] = secret

    response = requests.post(SPOTIFY_TOKEN_URL, data=payload, timeout=timeout)
    if response.status_code != 200:
        raise SpotifyAuthError(
            f"Spotify token endpoint returned {response.status_code}: {response.text}"
        )

    token_data = response.json()
    expires_in = token_data.get("expires_in")
    expires_at = time.time() + int(expires_in or 0)

    return SpotifyToken(
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        expires_at=expires_at,
        scope=token_data.get("scope", ""),
        token_type=token_data.get("token_type", "Bearer"),
        raw_response=token_data,
    )
