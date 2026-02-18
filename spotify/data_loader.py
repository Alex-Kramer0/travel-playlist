import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_s",
    "popularity",
]

_RAW_TO_CLEAN = {
    "Artist(s)":       "artist",
    "song":            "track_name",
    "text":            "lyrics",
    "Length":          "duration_s",
    "emotion":         "emotion",
    "Genre":           "genre",
    "Album":           "album",
    "Release Date":    "release_date",
    "Key":             "key",
    "Tempo":           "tempo",
    "Loudness (db)":   "loudness",
    "Time signature":  "time_signature",
    "Explicit":        "explicit",
    "Popularity":      "popularity",
    "Energy":          "energy",
    "Danceability":    "danceability",
    "Positiveness":    "valence",
    "Speechiness":     "speechiness",
    "Liveness":        "liveness",
    "Acousticness":    "acousticness",
    "Instrumentalness": "instrumentalness",
}

_DROP_PREFIXES = ("Good for", "Similar Artist", "Similar Song", "Similarity Score")


def _parse_loudness(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("db", "", case=False).str.strip(), errors="coerce")


def _parse_length(series: pd.Series) -> pd.Series:
    def _to_seconds(val):
        try:
            parts = str(val).strip().split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return np.nan
    return series.map(_to_seconds)


def load_spotify(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    drop_cols = [c for c in df.columns if any(c.startswith(p) for p in _DROP_PREFIXES)]
    df = df.drop(columns=drop_cols)

    df = df.rename(columns={k: v for k, v in _RAW_TO_CLEAN.items() if k in df.columns})

    if "loudness" in df.columns:
        df["loudness"] = _parse_loudness(df["loudness"])
    if "duration_s" in df.columns:
        df["duration_s"] = _parse_length(df["duration_s"])

    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {path}")
    return df


def select_features(
    df: pd.DataFrame,
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    filtered_df = df.dropna(subset=feature_cols).reset_index(drop=True)
    feature_df = filtered_df[feature_cols]
    print(f"Using {feature_df.shape[0]} tracks with complete audio features")
    return filtered_df, feature_df


def diagnostic_scale(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols)
    return scaled_df, scaler


def remove_outliers(
    filtered_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
    z_threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(feature_df)
    z_df = pd.DataFrame(z_scores, columns=feature_cols)

    outlier_mask = (np.abs(z_df) > z_threshold).any(axis=1)
    print(f"Flagged {outlier_mask.sum()} tracks (>±{z_threshold} z-score on any feature)")

    clean_df = filtered_df.loc[~outlier_mask].reset_index(drop=True)
    clean_features = feature_df.loc[~outlier_mask].reset_index(drop=True)
    print(f"Retained {clean_features.shape[0]} tracks after trimming outliers")
    return clean_df, clean_features


def scale_features(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols)
    return scaled_df, scaler


def build_dataset(
    path: str,
    feature_cols: list[str] = AUDIO_FEATURE_COLS,
    z_threshold: float = 4.0,
) -> dict:
    df = load_spotify(path)
    filtered_df, feature_df = select_features(df, feature_cols)
    _, _ = diagnostic_scale(feature_df, feature_cols)
    filtered_df, feature_df = remove_outliers(filtered_df, feature_df, feature_cols, z_threshold)
    scaled_df, scaler = scale_features(feature_df, feature_cols)
    return {
        "raw_df": df,
        "filtered_df": filtered_df,
        "feature_df": feature_df,
        "scaled_df": scaled_df,
        "scaler": scaler,
    }
