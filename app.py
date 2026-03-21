import sys
import os
import time
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SPOTIFY_API_DIR = str(PROJECT_ROOT / "spotify-api-integrations")
if SPOTIFY_API_DIR not in sys.path:
    sys.path.insert(0, SPOTIFY_API_DIR)

load_dotenv(PROJECT_ROOT / ".env")

# On Streamlit Community Cloud there is no .env file — secrets are provided via
# st.secrets. Sync them into os.environ so auth.py's os.getenv() calls work
# identically in both local and Cloud environments.
_SPOTIFY_KEYS = ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REDIRECT_URI")
for _key in _SPOTIFY_KEYS:
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except (KeyError, FileNotFoundError):
            pass

from auth import start_spotify_auth, complete_spotify_auth, SpotifyAuthError

SPOTIFY_SCOPES = [
    "user-read-email",
    "user-read-private",
    "playlist-modify-private",
    "playlist-modify-public",
]

# ── Server-side store for PKCE verifiers ─────────────────────────────────────
# st.session_state is lost when the browser redirects away (session drops).
# This dict lives in the server process and survives across reruns/sessions.
# @st.cache_resource ensures it's created once and never re-initialized.
@st.cache_resource
def _get_pending_auth() -> dict:
    return {}

_PENDING_AUTH: dict[str, str] = _get_pending_auth()  # {state: code_verifier}

st.set_page_config(
    page_title="Travel Playlist Generator",
    page_icon="🎵",
    layout="wide",
)

# ── Handle Spotify OAuth callback (redirect lands here) ─────────────────────
params = st.query_params
code = params.get("code")
state = params.get("state")

if code and state and state in _PENDING_AUTH:
    log.info("OAuth callback received — state=%s, code=%s…", state, code[:10])
    verifier = _PENDING_AUTH.pop(state)
    try:
        token = complete_spotify_auth(
            code=code,
            expected_state=state,
            provided_state=state,
            code_verifier=verifier,
        )
        st.session_state["spotify_token"] = token
        log.info("OAuth token exchange succeeded — expires_at=%s, scope=%r", token.expires_at, token.scope)
        st.query_params.clear()
        st.rerun()
    except SpotifyAuthError as e:
        log.error("OAuth token exchange failed: %s", e)
        st.error(f"Spotify auth failed: {e}")
        st.query_params.clear()
elif code and state:
    log.warning("OAuth callback received but no matching pending auth — state=%s (known states: %s)", state, list(_PENDING_AUTH.keys()))
    st.query_params.clear()

st.title("Travel Playlist Generator")
st.markdown("#### Turn your Airbnb stay into a soundtrack")

# ── Spotify connection status + auth button ──────────────────────────────────
has_token = (
    "spotify_token" in st.session_state
    and st.session_state["spotify_token"].expires_at > time.time()
)

if has_token:
    st.success("Spotify account connected. Navigate to **Generate Playlist** to get started.", icon="🟢")
else:
    col_auth, col_skip = st.columns(2)
    with col_auth:
        st.info("Connect your Spotify account to save playlists directly.", icon="🎧")
        if st.button("Connect to Spotify", type="primary", use_container_width=True):
            try:
                flow = start_spotify_auth(
                    scopes=SPOTIFY_SCOPES,
                    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
                )
                _PENDING_AUTH[flow.state] = flow.code_verifier
                log.info("Starting Spotify auth — state=%s, redirect=%s", flow.state, os.getenv("SPOTIFY_REDIRECT_URI"))
                st.markdown(
                    f'<meta http-equiv="refresh" content="0;url={flow.authorization_url}">',
                    unsafe_allow_html=True,
                )
                st.stop()
            except SpotifyAuthError as e:
                st.error(f"Could not start Spotify auth: {e}")
    with col_skip:
        st.info("You can also use the app without Spotify and download playlists as CSV.", icon="📄")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ### How It Works

        1. **Paste an Airbnb listing URL** — we extract the vibe
           keywords and emotions from the listing description.
        2. **Keywords drive the recommendation engine** — a 4-layer
           scoring pipeline matches keywords to 530k Spotify tracks.
        3. **Get a personalized playlist** — ranked by lyrics match,
           emotion, audio similarity, and cluster affinity.
        """
    )

with col2:
    st.markdown(
        """
        ### Recommendation Layers

        | Layer | Signal | Weight |
        |---|---|---|
        | Lyrics match | Location terms in song lyrics | 35% |
        | Emotion match | NLI-inferred emotion vs track emotion | 25% |
        | Audio cosine | Keyword-derived audio target vs track features | 25% |
        | Cluster boost | Nearest K-means cluster to target vector | 15% |
        """
    )

st.divider()

st.markdown("👈 **Navigate to Generate Playlist** in the sidebar to get started.")
