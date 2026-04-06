import re
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import csv
from datetime import datetime, timezone

# ── Path setup so we can import project modules ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPOTIFY_DIR = str(PROJECT_ROOT / "spotify-clustering")
RECO_DIR = str(PROJECT_ROOT / "recommendation")
SPOTIFY_API_DIR = str(PROJECT_ROOT / "spotify-api-integrations")
FEEDBACK_CSV = PROJECT_ROOT / "user-feedback" / "playlist_feedback.csv"

for p in [SPOTIFY_DIR, RECO_DIR, SPOTIFY_API_DIR, str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(PROJECT_ROOT / ".env")

import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

from data_loader import AUDIO_FEATURE_COLS, load_spotify, select_features, remove_outliers, scale_features
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
from evaluation import evaluate_playlist, top_k_popular_baseline

# ── Emotion Themes Set up ───────────────────────────────────────────────────────────────
EMOTION_THEMES = {
    "joy": { #done
        "primary": "#43b3bd",
        "secondary": "#b3f08c",
        "accent": "#f9f37e",
        "text": "#FAFAFA",
    },
    "trust": {
        "primary": "#36476f",
        "secondary": "#3e868e",
        "accent": "#d1b896",
        "text": "#FAFAFA",
    },
    "anticipation": {
        "primary": "#b55937",
        "secondary": "#a1b099",
        "accent": "#f18e30",
        "text": "#030303",
    },
    "surprise": {
        "primary": "#9D4EDD",
        "secondary": "#240046",
        "accent": "#C77DFF",
        "text": "#FAFAFA",
    },
    "sadness": { # blues
        "primary": "#13273e",
        "secondary": "#39546d",
        "accent": "#567b89",
        "text": "#FAFAFA",
    },
    "fear": {
        "primary": "#6A4C93",
        "secondary": "#2A2A35",
        "accent": "#9A8C98",
        "text": "#FAFAFA",
    },
    "anger": {
        "primary": "#D62828",
        "secondary": "#3A0CA3",
        "accent": "#F77F00",
        "text": "#FAFAFA",
    },
    "disgust": {
        "primary": "#6A994E",
        "secondary": "#283618",
        "accent": "#A7C957",
        "text": "#FAFAFA",
    },
    "default": {
        "primary": "#0e1117",
        "secondary": "#ff4b4b",
        "accent": "#262730",
        "text": "#fafafa",
    },
}

def apply_emotion_theme(emotion: str | None):
    if not emotion or emotion.lower() == "default":
        return

    theme = EMOTION_THEMES.get((emotion or "").lower(), EMOTION_THEMES["default"])
    # apply color changes based on dominat emotion once generated
    theme = EMOTION_THEMES.get((emotion or "").lower(), EMOTION_THEMES["default"])

    primary = theme["primary"]
    secondary = theme["secondary"]
    accent = theme["accent"]
    text = theme["text"]

    st.markdown(
        f"""
        <style>
        /* Main page background blocks */
        html, body, [data-testid="stAppViewContainer"], .stApp {{
        background: linear-gradient(180deg, {primary} 0%, {secondary} 100%);
        color: {text};
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stMain"] {{
            background: transparent;
        }}

        /* Input areas, dataframe containers, expanders, tabs, etc. */
        div[data-testid="stTextInputRootElement"] > div,
        div[data-testid="stTextAreaRootElement"] > div,
        div[data-testid="stNumberInputRootElement"] > div,
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stSlider"] {{
            border-radius: 12px;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {primary};
            color: #0E1117;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }}

        .stButton > button:hover {{
            background-color: {accent};
            color: #0E1117;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            border-radius: 10px 10px 0 0;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {primary};
            border-bottom: 2px solid {primary};
        }}

        /* Metric cards / generic blocks */
        div[data-testid="stMetric"] {{
            background-color: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px;
            border-radius: 12px;
        }}

        /* Dataframe container */
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            overflow: hidden;
        }}

        /* Expanders */
        div[data-testid="stExpander"] {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            background-color: rgba(255,255,255,0.03);
        }}

        /* Accent text helpers */
        .emotion-accent {{
            color: {primary};
            font-weight: 700;
        }}
        /* Feedback stars / sentiment buttons */
        div[data-testid="stFeedback"] button,
        [data-testid="stFeedback"] button,
        .stFeedback button {{
            background-color: rgba(255,255,255,0.06);
            border: 1px solid {accent};
            border-radius: 10px;
            color: {text};
        }}

        /* Hover */
        div[data-testid="stFeedback"] button:hover,
        [data-testid="stFeedback"] button:hover,
        .stFeedback button:hover {{
            background-color: {secondary};
            border-color: {accent};
            color: {text};
        }}

        /* Selected / active */
        div[data-testid="stFeedback"] button[aria-pressed="true"],
        [data-testid="stFeedback"] button[aria-pressed="true"],
        .stFeedback button[aria-pressed="true"] {{
            background-color: {accent};
            border-color: {accent};
            color: #0E1117;
            box-shadow: 0 0 0 2px rgba(255,255,255,0.08);
        }}
        /* Text input + text area backgrounds */
        div[data-testid="stTextInputRootElement"] > div,
        div[data-testid="stTextAreaRootElement"] > div {{
            background-color: {primary};
            border: 1px solid {accent};
            border-radius: 12px;
        }}

        div[data-testid="stTextAreaRootElement"] textarea,
        div[data-testid="stTextInputRootElement"] input {{
            background-color: {primary} !important;
            color: {text} !important;
            caret-color: {text};
        }}

        div[data-testid="stTextAreaRootElement"] textarea::placeholder,
        div[data-testid="stTextInputRootElement"] input::placeholder {{
            color: rgba(255,255,255,0.75);
        }}

        /* Main buttons */
        .stButton > button,
        .stDownloadButton > button {{
            background-color: {primary};
            color: #0E1117;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }}

        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background-color: {accent};
            color: #0E1117;
        }}

        /* Expander */
        div[data-testid="stExpander"] {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            background-color: rgba(255,255,255,0.03);
        }}

        div[data-testid="stExpander"] summary {{
            background-color: {primary};
            color: #0E1117;
            border-radius: 10px;
            padding: 0.4rem 0.75rem;
            font-weight: 600;
        }}

        div[data-testid="stExpander"] summary:hover {{
            background-color: {accent};
            color: #0E1117;
        }}

        div[data-testid="stExpanderDetails"] {{
            background-color: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 0 0 10px 10px;
            padding: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Generate Playlist", page_icon="🎶", layout="wide")

st.title("Generate Playlist")
#apply_emotion_theme("default")
current_emotion = st.session_state.get("current_emotion", "default")
apply_emotion_theme(current_emotion)

# ── Paths ─────────────────────────────────────────────────────────────────────
SPOTIFY_CSV = PROJECT_ROOT / "spotify-clustering" / "dataset" / "spotify_dataset_lyrics_top50k.csv"
AIRBNB_DATASET = PROJECT_ROOT / "Airbnb" / "data" / "Output.csv.zip"
NRC_PATH = PROJECT_ROOT / "Airbnb" / "emolex" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
K_FINAL = 5


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Spotify dataset and clustering...")
def load_spotify_data():
    """Load, clean, cluster the Spotify dataset. Runs once per process."""
    df_spotify = load_spotify(str(SPOTIFY_CSV))
    filtered_df, feature_df = select_features(df_spotify, AUDIO_FEATURE_COLS)
    filtered_df, feature_df = remove_outliers(filtered_df, feature_df, AUDIO_FEATURE_COLS)
    scaled_df, scaler = scale_features(feature_df, AUDIO_FEATURE_COLS)
    kmeans, clusters = fit_kmeans(scaled_df, k=K_FINAL)
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

# Save playlist feedback to CSV for later analysis. Appends a new row with timestamp, listing info, and user rating.
def save_playlist_feedback(
    rating: int,
    listing_name: str,
    input_mode: str,
    listing_url: str,
    track_count: int,
):
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)

    file_exists = FEEDBACK_CSV.exists()

    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "listing_name",
                "input_mode",
                "listing_url",
                "track_count",
                "rating",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "listing_name": listing_name,
                "input_mode": input_mode,
                "listing_url": listing_url,
                "track_count": track_count,
                "rating": rating,
            }
        )


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
#st.markdown("### Enter an Airbnb Listing URL")
#st.caption("Paste the full URL of an Airbnb listing from the dataset.")

#url_input = st.text_input(
#    "Airbnb listing URL",
#    placeholder="https://www.airbnb.com/rooms/2536175",
#    label_visibility="collapsed",
#)

#top_n = st.slider("Number of tracks", min_value=5, max_value=50, value=20)

#generate_clicked = st.button("Generate Playlist", type="primary", use_container_width=True)

# ── User input ────────────────────────────────────────────────────────────────
st.markdown("### Start with an Airbnb listing")
st.caption("Use a listing URL from the dataset, or paste a listing description manually.")

tab_url, tab_desc = st.tabs(["From Airbnb URL", "Paste Description"])

with tab_url:
    url_input = st.text_input(
        "Airbnb listing URL",
        placeholder="https://www.airbnb.com/rooms/2536175",
        help="Use a full Airbnb listing URL that exists in the dataset.",
    )

with tab_desc:
    pasted_description = st.text_area(
        "Airbnb listing description",
        placeholder="Paste the Airbnb description here...",
        height=180,
        help="Use this if you want to generate a playlist from listing text directly.",
    )

top_n = st.slider("Number of tracks", min_value=5, max_value=50, value=20)

with st.expander("Advanced Settings — Layer Weights"):
    st.caption(
        "Adjust how much each recommendation layer contributes to the final score. "
        "Values are automatically normalized so they sum to 100%."
    )
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        w_lyrics = st.slider("Lyrics keyword match", 0.0, 1.0, 0.35, 0.05, key="w_lyrics")
        w_emotion = st.slider("Emotion match", 0.0, 1.0, 0.25, 0.05, key="w_emotion")
    with col_w2:
        w_audio = st.slider("Audio similarity", 0.0, 1.0, 0.25, 0.05, key="w_audio")
        w_cluster = st.slider("Cluster boost", 0.0, 1.0, 0.15, 0.05, key="w_cluster")

    raw_total = w_lyrics + w_emotion + w_audio + w_cluster
    if raw_total > 0:
        nw = {
            "lyrics": w_lyrics / raw_total,
            "emotion": w_emotion / raw_total,
            "audio": w_audio / raw_total,
            "cluster": w_cluster / raw_total,
        }
    else:
        nw = {"lyrics": 0.25, "emotion": 0.25, "audio": 0.25, "cluster": 0.25}

    st.markdown(
        f"**Normalized:** Lyrics {nw['lyrics']:.0%} · Emotion {nw['emotion']:.0%} · "
        f"Audio {nw['audio']:.0%} · Cluster {nw['cluster']:.0%}"
    )

generate_clicked = st.button("Generate Playlist", type="primary", use_container_width=True)

# ── Pipeline ──────────────────────────────────────────────────────────────────
#if generate_clicked:
#    if not url_input.strip():
#        st.warning("Please enter an Airbnb listing URL.")
#        st.stop()

    # Step 1: Look up listing
#    try:
#        listing_data = get_listing_by_url(url_input.strip(), airbnb_df)
#    except ValueError as e:
#        st.error(str(e))
#        st.stop()

#    description = listing_data.get("description", "") or ""
if generate_clicked:
    url_value = url_input.strip() if "url_input" in locals() and url_input else ""
    desc_value = pasted_description.strip() if "pasted_description" in locals() and pasted_description else ""

    if not url_value and not desc_value:
        st.warning("Please enter either an Airbnb listing URL or paste a listing description.")
        st.stop()

    # Step 1: Resolve listing source
    if url_value:
        try:
            listing_data = get_listing_by_url(url_value, airbnb_df)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        description = listing_data.get("description", "") or ""
        listing_name = listing_data.get("name", "Airbnb Listing")
        neighbourhood = listing_data.get("neighbourhood_cleansed", "N/A")

    else:
        description = desc_value
        listing_data = {
            "name": "Custom Airbnb Listing",
            "neighbourhood_cleansed": "N/A",
            "description": description,
        }
        listing_name = "Custom Airbnb Listing"
        neighbourhood = "N/A"

    if not description.strip():
        st.error("The listing description is empty.")
        st.stop()
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

    #dominant = emotion_scores.get("dominant_emotion")
    #apply_emotion_theme(dominant)
    dominant = emotion_scores.get("dominant_emotion") or "default"
    st.session_state["current_emotion"] = dominant
    apply_emotion_theme(dominant)
    if dominant:
        emotion_key = dominant.lower()

        EMOTION_EMOJIS = {
            "joy": ["✨", "😊"],
            "trust": ["🤝", "💚"],
            "anticipation": ["🌅", "🚀"],
            "surprise": ["🎉", "⚡"],
            "sadness": ["🌧️", "💙"],
            "fear": ["🌫️", "👁️"],
            "anger": ["🔥", "😤"],
            "disgust": ["⚠️", "🫥"],
            "default": ["🎶"],
        }

        emotion_emojis = EMOTION_EMOJIS.get(emotion_key, EMOTION_EMOJIS["default"])
        emoji_string = " ".join(emotion_emojis)

        st.markdown(
            f'''
            <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
                <span><strong>Dominant emotion:</strong></span>
                <span class="emotion-chip">{dominant.title()}</span>
                <span class="emotion-emoji">{emoji_string}</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # --- Additional emotions ---
        emotion_display = {
            k: v for k, v in emotion_scores.items()
            if k != "dominant_emotion" and isinstance(v, (int, float))
        }

        top_emotions = sorted(
            emotion_display.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]

        if top_emotions:
            chips_html = " ".join(
                [
                    f'<span class="emotion-secondary-chip">{emotion.title()}</span>'
                    for emotion, _ in top_emotions
                ]
            )


    # Step 3: Recommend
    with st.spinner("Generating playlist..."):
        playlist = recommend(
            keywords=keywords,
            df=clustered_df,
            scaled_df=scaled_df,
            top_n=top_n,
            weights=nw,
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
    #st.session_state["current_playlist"] = playlist
    #st.session_state["current_listing_name"] = listing_data.get("name", "Airbnb Listing")
    st.session_state["current_playlist"] = playlist
    st.session_state["current_listing_name"] = listing_data.get("name", "Airbnb Listing")
    st.session_state["current_listing_url"] = url_input.strip()
    st.session_state["current_input_mode"] = "url"
    st.session_state["playlist_feedback_saved"] = False

    # Reset the rating widget for a newly generated playlist
    if "playlist_rating" in st.session_state:
        del st.session_state["playlist_rating"]

    # Step 5: Keyword match heatmap + score breakdown
    top10 = playlist.head(10).copy()
    top10["label"] = top10["track_name"].str[:30] + " — " + top10["artist"].str[:20]

    # ── Keyword Matches in Lyrics (expandable per track) ────────────────────
    lyrics_matches = top10[top10["matched_keywords"].astype(bool)]
    if not lyrics_matches.empty:
        st.markdown("### Keyword Matches in Lyrics")
        st.caption("Expand a track to see which keywords were found and the surrounding lyrics context.")

        for _, row in lyrics_matches.iterrows():
            matched_list = [k.strip() for k in row["matched_keywords"].split(",") if k.strip()]
            track_label = f"{row['track_name']} — {row['artist']}  ({len(matched_list)} keyword{'s' if len(matched_list) != 1 else ''})"
            with st.expander(track_label):
                st.markdown("**Matched keywords:** " + " ".join(f"`{kw}`" for kw in matched_list))
                lyrics_text = row.get("lyrics", "") or ""
                if lyrics_text:
                    # Highlight matched keywords in the lyrics
                    highlighted = lyrics_text
                    for kw in matched_list:
                        pattern = re.compile(r"(\b" + re.escape(kw) + r"\b)", re.IGNORECASE)
                        highlighted = pattern.sub(r"**\1**", highlighted)
                    st.markdown(highlighted)
                else:
                    st.info("Lyrics not available for this track.")

    # ── Score Breakdown Stacked Bar ──────────────────────────────────────────
    st.markdown("### Score Breakdown — Top 10")

    fig, ax = plt.subplots(figsize=(11, 6))
    layers = ["score_lyrics", "score_emotion", "score_audio", "score_cluster"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    labels = [
        f"Lyrics ({nw['lyrics']:.0%})",
        f"Emotion ({nw['emotion']:.0%})",
        f"Audio ({nw['audio']:.0%})",
        f"Cluster ({nw['cluster']:.0%})",
    ]
    weights = [nw["lyrics"], nw["emotion"], nw["audio"], nw["cluster"]]

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

    # Step 6: Playlist evaluation vs popularity baseline
    st.markdown("---")
    st.markdown("### Playlist Evaluation")
    st.caption(
        "Comparing your generated playlist against a baseline of the "
        f"top {top_n} most popular tracks in the dataset."
    )

    baseline = top_k_popular_baseline(clustered_df, k=top_n)
    metrics_ours = evaluate_playlist(playlist)
    metrics_base = evaluate_playlist(baseline)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        delta_ad = metrics_ours["audio_diversity"] - metrics_base["audio_diversity"]
        st.metric(
            label="Intra-Playlist Audio Diversity",
            value=f"{metrics_ours['audio_diversity']:.4f}",
            delta=f"{delta_ad:+.4f} vs baseline",
            delta_color="normal",
            help="Mean pairwise cosine distance (1 − similarity) between all track audio vectors. Higher = more diverse.",
        )

    with col_m2:
        delta_ge = metrics_ours["genre_entropy"] - metrics_base["genre_entropy"]
        st.metric(
            label="Genre Entropy",
            value=f"{metrics_ours['genre_entropy']:.4f}",
            delta=f"{delta_ge:+.4f} vs baseline",
            delta_color="normal",
            help="Shannon entropy over genre distribution. Higher = broader genre coverage.",
        )

    with st.expander("Baseline details (Top-K Popular)", expanded=False):
        base_cols = ["track_name", "artist", "genre", "popularity"]
        base_available = [c for c in base_cols if c in baseline.columns]
        st.dataframe(baseline[base_available], hide_index=True)

    # Side-by-side bar charts (separate y-axes)
    fig_eval, (ax_ad, ax_ge) = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Your Playlist", "Top-K Popular"]
    colors = ["#1DB954", "#B3B3B3"]

    ax_ad.bar(labels, [metrics_ours["audio_diversity"], metrics_base["audio_diversity"]], color=colors)
    ax_ad.set_ylabel("Mean Pairwise Cosine Distance")
    ax_ad.set_title("Intra-Playlist Audio Diversity")

    ax_ge.bar(labels, [metrics_ours["genre_entropy"], metrics_base["genre_entropy"]], color=colors)
    ax_ge.set_ylabel("Shannon Entropy (nats)")
    ax_ge.set_title("Genre Entropy")

    fig_eval.tight_layout()
    st.pyplot(fig_eval)


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
# ── Playlist feedback ─────────────────────────────────────────────────────────
if "current_playlist" in st.session_state:
    st.markdown("---")
    st.markdown("### Rate your playlist")
    st.caption("How well did this playlist match the Airbnb vibe?")

    rating_value = st.feedback("stars", key="playlist_rating")

    if rating_value is not None and not st.session_state.get("playlist_feedback_saved", False):
        # st.feedback("stars") returns 0–4, so convert to 1–5
        stars = rating_value + 1

        save_playlist_feedback(
            rating=stars,
            listing_name=st.session_state.get("current_listing_name", "Unknown Listing"),
            input_mode=st.session_state.get("current_input_mode", "unknown"),
            listing_url=st.session_state.get("current_listing_url", ""),
            track_count=len(st.session_state["current_playlist"]),
        )

        st.session_state["playlist_feedback_saved"] = True
        st.success(f"Thanks for rating this playlist {stars}/5.")
