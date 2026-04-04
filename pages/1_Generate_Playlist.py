import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ── Path setup so we can import project modules ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPOTIFY_DIR = str(PROJECT_ROOT / "spotify-clustering")
RECO_DIR = str(PROJECT_ROOT / "recommendation")
SPOTIFY_API_DIR = str(PROJECT_ROOT / "spotify-api-integrations")

for p in [SPOTIFY_DIR, RECO_DIR, SPOTIFY_API_DIR, str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(PROJECT_ROOT / ".env")

import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from data_loader import AUDIO_FEATURE_COLS, CLUSTER_FEATURE_COLS, load_spotify, select_features, remove_outliers, scale_features, quantile_transform, pca_reduce
from clustering import fit_kmeans
from Airbnb.nlp_pipeline import (
    load_listing_dataset,
    load_nrc_lexicon,
    get_listing_by_url,
    extract_vibe_keywords,
    extract_emotions,
    build_keyword_json,
    build_emotion_json,
)
from auth import SpotifyAuthError
from playlists import create_playlist, add_tracks_to_playlist, resolve_track_uris

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Generate Playlist", page_icon="🎶", layout="wide")

st.title("Generate Playlist")

# ── Paths ─────────────────────────────────────────────────────────────────────
SPOTIFY_CSV = PROJECT_ROOT / "spotify-clustering" / "dataset" / "spotify_dataset_lyrics_random50k.csv"
AIRBNB_DATASET = PROJECT_ROOT / "Airbnb" / "data" / "Output.csv.zip"
NRC_PATH = PROJECT_ROOT / "Airbnb" / "emolex" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
K_FINAL = 5
N_PCA = 2


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Spotify dataset and clustering...")
def load_spotify_data():
    """Load, clean, cluster the Spotify dataset. Runs once per process."""
    df_spotify = load_spotify(str(SPOTIFY_CSV))
    filtered_df, feature_df = select_features(df_spotify, AUDIO_FEATURE_COLS)
    filtered_df, feature_df = remove_outliers(filtered_df, feature_df, AUDIO_FEATURE_COLS)
    scaled_df, scaler = scale_features(feature_df, AUDIO_FEATURE_COLS)
    qt_df, _ = quantile_transform(feature_df, CLUSTER_FEATURE_COLS)
    pca_df, _ = pca_reduce(qt_df, n_components=N_PCA)
    kmeans, clusters = fit_kmeans(pca_df, k=K_FINAL)
    clustered_df = filtered_df.copy()
    clustered_df["cluster"] = clusters
    return clustered_df, scaled_df


@st.cache_resource(show_spinner="Loading Airbnb listings...")
def load_airbnb_data():
    """Load the Airbnb listing dataset."""
    path = str(AIRBNB_DATASET)
    if path.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return load_listing_dataset(path)


@st.cache_resource(show_spinner="Loading emotion lexicon...")
def load_nrc():
    """Load the NRC emotion lexicon."""
    return load_nrc_lexicon(str(NRC_PATH))


# ── Load data ─────────────────────────────────────────────────────────────────
missing_files = []
if not SPOTIFY_CSV.exists():
    missing_files.append(f"Spotify CSV: `{SPOTIFY_CSV}`")
airbnb_path = AIRBNB_DATASET if AIRBNB_DATASET.exists() else AIRBNB_DATASET.with_suffix("")
if not airbnb_path.exists():
    missing_files.append(f"Airbnb dataset: `{AIRBNB_DATASET}`")
else:
    AIRBNB_DATASET = airbnb_path
if not NRC_PATH.exists():
    missing_files.append(f"NRC lexicon: `{NRC_PATH}`")

if missing_files:
    st.error("Missing required data files:\n\n" + "\n".join(f"- {f}" for f in missing_files))
    st.stop()

clustered_df, scaled_df = load_spotify_data()
airbnb_df = load_airbnb_data()
cat_to_words = load_nrc()

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.metric("Spotify tracks", f"{len(clustered_df):,}")
    st.metric("Airbnb listings", f"{len(airbnb_df):,}")

# ── User input ────────────────────────────────────────────────────────────────
st.markdown("### Enter an Airbnb Listing URL")
st.caption("Paste the full URL of an Airbnb listing from the dataset.")

url_input = st.text_input(
    "Airbnb listing URL",
    placeholder="https://www.airbnb.com/rooms/2536175",
    label_visibility="collapsed",
)

top_n = st.slider("Number of tracks", min_value=5, max_value=50, value=20)

generate_clicked = st.button("Generate Playlist", type="primary", use_container_width=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
if generate_clicked:
    if not url_input.strip():
        st.warning("Please enter an Airbnb listing URL.")
        st.stop()

    # Step 1: Look up listing
    try:
        listing_data = get_listing_by_url(url_input.strip(), airbnb_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    description = listing_data.get("description", "") or ""

    st.markdown("---")
    st.markdown("### Listing Details")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Name:** {listing_data.get('name', 'N/A')}")
        st.markdown(f"**Neighborhood:** {listing_data.get('neighbourhood_cleansed', 'N/A')}")
    with col_b:
        with st.expander("Full description", expanded=False):
            st.write(description if description else "_No description available._")

    # Lazy imports — torch/sentence-transformers only loaded when Generate is clicked
    from keyword_embedder import resolve_keywords, build_lyric_index  # noqa: F401
    from recommender import recommend

    # Step 2: Extract keywords & emotions
    with st.spinner("Extracting keywords and emotions..."):
        keywords = extract_vibe_keywords(description, top_n=5)
        emotion_scores = extract_emotions(description, cat_to_words)

    st.markdown("### Extracted Keywords")
    if keywords:
        st.markdown(" ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("No keywords could be extracted from this listing description.")
        st.stop()

    dominant = emotion_scores.get("dominant_emotion")
    if dominant:
        st.markdown(f"**Dominant emotion:** {dominant}")

    # Step 3: Recommend
    with st.spinner("Generating playlist..."):
        playlist = recommend(
            keywords=keywords,
            df=clustered_df,
            scaled_df=scaled_df,
            top_n=top_n,
        )

    # Step 4: Display playlist
    st.markdown("---")
    st.markdown(f"### Your Playlist ({len(playlist)} tracks)")

    display_cols = [
        "track_name", "artist", "genre", "emotion",
        "score", "score_lyrics", "score_emotion", "score_audio", "score_cluster",
    ]
    available = [c for c in display_cols if c in playlist.columns]
    st.dataframe(
        playlist[available],
        width="stretch",
        hide_index=True,
        column_config={
            "track_name": st.column_config.TextColumn("Track", width="large"),
            "artist": st.column_config.TextColumn("Artist", width="medium"),
            "genre": st.column_config.TextColumn("Genre"),
            "emotion": st.column_config.TextColumn("Emotion"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.3f"),
            "score_lyrics": st.column_config.NumberColumn("Lyrics", format="%.2f"),
            "score_emotion": st.column_config.NumberColumn("Emotion", format="%.2f"),
            "score_audio": st.column_config.NumberColumn("Audio", format="%.2f"),
            "score_cluster": st.column_config.NumberColumn("Cluster", format="%.2f"),
        },
    )

    # Store playlist in session state so it persists across reruns
    st.session_state["current_playlist"] = playlist
    st.session_state["current_listing_name"] = listing_data.get("name", "Airbnb Listing")

    # Step 5: Score breakdown chart
    st.markdown("### Score Breakdown — Top 10")
    top10 = playlist.head(10).copy()
    top10["label"] = top10["track_name"].str[:30] + " — " + top10["artist"].str[:20]

    fig, ax = plt.subplots(figsize=(11, 6))
    layers = ["score_lyrics", "score_emotion", "score_audio", "score_cluster"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    labels = ["Lyrics (0.35)", "Emotion (0.25)", "Audio (0.25)", "Cluster (0.15)"]
    weights = [0.35, 0.25, 0.25, 0.15]

    bottoms = np.zeros(len(top10))
    for layer, color, label, w in zip(layers, colors, labels, weights):
        vals = top10[layer].values * w
        ax.barh(top10["label"], vals, left=bottoms, color=color, label=label)
        bottoms += vals

    ax.set_xlabel("Weighted score contribution")
    ax.set_title(f"Score breakdown — {listing_data.get('name', 'Listing')}")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)


# ── Save to Spotify section ──────────────────────────────────────────────────
if "current_playlist" in st.session_state:
    st.markdown("---")
    st.markdown("### Export Options")

    has_token = (
        "spotify_token" in st.session_state
        and st.session_state["spotify_token"].expires_at > time.time()
    )

    col_spotify, col_csv = st.columns(2)

    with col_csv:
        st.markdown("#### Continue without Spotify")
        st.caption("Download the playlist as a CSV file.")
        csv_data = st.session_state["current_playlist"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv_data,
            file_name="travel_playlist.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_spotify:
        st.markdown("#### Save to Spotify")
        if not has_token:
            st.caption("Connect your Spotify account on the **Home** page first.")
        else:
            listing_name = st.session_state.get("current_listing_name", "Travel Playlist")
            playlist_name = st.text_input(
                "Playlist name",
                value=f"Travel Vibes — {listing_name}",
            )

            if st.button("Save to Spotify", type="primary", use_container_width=True):
                token = st.session_state["spotify_token"]
                playlist_df = st.session_state["current_playlist"]

                with st.spinner("Searching Spotify for tracks..."):
                    track_results = resolve_track_uris(token.access_token, playlist_df)
                    found_uris = [uri for _, uri in track_results if uri]
                    not_found = [name for name, uri in track_results if not uri]

                if not found_uris:
                    st.error("Could not find any of the recommended tracks on Spotify.")
                else:
                    with st.spinner("Creating playlist..."):
                        try:
                            new_playlist = create_playlist(
                                token.access_token,
                                playlist_name,
                                public=True,
                                description=(
                                    f"Generated by Travel Playlist Generator for: {listing_name}"
                                ),
                            )
                            add_tracks_to_playlist(
                                token.access_token,
                                new_playlist["id"],
                                found_uris,
                            )
                            playlist_url = new_playlist.get("external_urls", {}).get("spotify", "")
                            st.success(
                                f"Playlist created with {len(found_uris)} tracks!"
                            )
                            if playlist_url:
                                st.markdown(f"[Open in Spotify]({playlist_url})")
                            if not_found:
                                st.caption(
                                    f"{len(not_found)} track(s) not found on Spotify: "
                                    + ", ".join(not_found[:5])
                                    + ("..." if len(not_found) > 5 else "")
                                )
                        except SpotifyAuthError as e:
                            st.error(f"Spotify API error: {e}")
