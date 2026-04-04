import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPOTIFY_DIR = str(PROJECT_ROOT / "spotify-clustering")
for p in [SPOTIFY_DIR, str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import (
    AUDIO_FEATURE_COLS, CLUSTER_FEATURE_COLS, load_spotify, select_features,
    remove_outliers, scale_features, quantile_transform, pca_reduce,
)
from clustering import fit_kmeans, build_cluster_profile

st.set_page_config(page_title="Clustering Insights", page_icon="📊", layout="wide")
st.title("Clustering Insights")
st.markdown("Explore how Spotify tracks are grouped by audio features using K-Means clustering.")

SPOTIFY_CSV = PROJECT_ROOT / "spotify-clustering" / "dataset" / "spotify_dataset_lyrics_random50k.csv"
K_FINAL = 5
N_PCA = 2

@st.cache_resource(show_spinner="Loading Spotify data & clustering…")
def _load_data():
    df = load_spotify(str(SPOTIFY_CSV))
    filt, feat = select_features(df, AUDIO_FEATURE_COLS)
    filt, feat = remove_outliers(filt, feat, AUDIO_FEATURE_COLS)
    scaled_all, _ = scale_features(feat, AUDIO_FEATURE_COLS)
    cluster_scaled, _ = scale_features(feat[CLUSTER_FEATURE_COLS], CLUSTER_FEATURE_COLS)
    qt_df, _ = quantile_transform(feat, CLUSTER_FEATURE_COLS)
    pca_df, pca_model = pca_reduce(qt_df, n_components=N_PCA)
    km, labels = fit_kmeans(pca_df, k=K_FINAL)
    return filt, scaled_all, cluster_scaled, qt_df, pca_df, pca_model, labels, km

if not SPOTIFY_CSV.exists():
    st.error(f"Missing: `{SPOTIFY_CSV}`")
    st.stop()

filt_df, scaled_all, cluster_scaled, qt_df, pca_df, pca_model, clusters, kmeans_model = _load_data()
sil = silhouette_score(pca_df, clusters)
db = davies_bouldin_score(pca_df, clusters)
ch = calinski_harabasz_score(pca_df, clusters)

with st.sidebar:
    st.metric("Tracks", f"{len(filt_df):,}")
    st.metric("Cluster Features", f"{len(CLUSTER_FEATURE_COLS)} → QT → PCA{N_PCA}")
    st.metric("k", K_FINAL)
    st.metric("Silhouette", f"{sil:.3f}", help="Higher is better (range -1 to 1)")
    st.metric("Davies-Bouldin", f"{db:.3f}", help="Lower is better (0 = perfect)")
    st.metric("Calinski-Harabasz", f"{ch:,.0f}", help="Higher is better")

# --- Section 1: Elbow / Silhouette ---
st.markdown("---")
st.subheader("1 · Elbow & Silhouette Analysis")
st.caption("Computed on QuantileTransformer + PCA-reduced space (7 features → QT → 2 components). The chosen k is highlighted.")

@st.cache_data(show_spinner="Computing evaluation metrics for k = 2–10…")
def _compute_k_eval(_pca_df):
    from sklearn.cluster import KMeans as _KM
    rows = []
    for k in range(2, 11):
        km = _KM(n_clusters=k, random_state=42, n_init="auto")
        lb = km.fit_predict(_pca_df)
        np.random.seed(42)
        sub_idx = np.random.choice(len(_pca_df), min(5000, len(_pca_df)), replace=False)
        s = silhouette_score(_pca_df.iloc[sub_idx], lb[sub_idx])
        d = davies_bouldin_score(_pca_df, lb)
        c = calinski_harabasz_score(_pca_df, lb)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": s, "davies_bouldin": d, "calinski_harabasz": c})
    return pd.DataFrame(rows)

K_EVAL = _compute_k_eval(pca_df)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))

axes1[0].plot(K_EVAL["k"], K_EVAL["inertia"], marker="o", color="tab:blue")
axes1[0].axvline(K_FINAL, color="red", ls="--", lw=1)
axes1[0].set_xlabel("k"); axes1[0].set_ylabel("Inertia")
axes1[0].set_title("Elbow (Inertia) — lower is more compact")

axes1[1].plot(K_EVAL["k"], K_EVAL["silhouette"], marker="s", color="tab:orange")
axes1[1].axvline(K_FINAL, color="red", ls="--", lw=1)
axes1[1].set_xlabel("k"); axes1[1].set_ylabel("Silhouette")
axes1[1].set_title("Silhouette Score — higher is better")

axes1[2].plot(K_EVAL["k"], K_EVAL["davies_bouldin"], marker="D", color="tab:green")
axes1[2].axvline(K_FINAL, color="red", ls="--", lw=1)
axes1[2].set_xlabel("k"); axes1[2].set_ylabel("Davies-Bouldin")
axes1[2].set_title("Davies-Bouldin Index — lower is better")

for ax in axes1:
    ax.set_xticks(K_EVAL["k"])

fig1.suptitle("Clustering Evaluation Metrics by k", fontsize=14)
fig1.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

st.dataframe(
    K_EVAL.style.format({"inertia": "{:,.0f}", "silhouette": "{:.4f}", "davies_bouldin": "{:.3f}", "calinski_harabasz": "{:,.0f}"}),
    use_container_width=True, hide_index=True,
)

# --- Section 2: Cluster Heatmap ---
st.markdown("---")
st.subheader("2 · Cluster Feature Heatmap")
st.caption("Mean z-score of each audio feature per cluster. Red = above average, blue = below.")

profile = build_cluster_profile(cluster_scaled, clusters)
n_c = profile.shape[1]
fig2, ax2h = plt.subplots(figsize=(max(10, 2 + 1.5 * n_c), 6))
sns.heatmap(profile, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2h)
ax2h.set_ylabel("Feature"); ax2h.set_xlabel("Cluster")
ax2h.set_title("Cluster Mean (z-score) per Audio Feature")
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# --- Section 3: PCA Scatter ---
st.markdown("---")
st.subheader("3 · PCA Cluster Scatter")
st.caption("Clusters were fitted in this 2-component PCA space (after QuantileTransformer).")

evr = pca_model.explained_variance_ratio_
scatter_df = pca_df.copy()
scatter_df["cluster"] = clusters

fig3, ax3 = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=scatter_df, x="PC1", y="PC2", hue="cluster", palette="tab10", alpha=0.5, s=30, ax=ax3)
ax3.axhline(0, color="gray", lw=0.5); ax3.axvline(0, color="gray", lw=0.5)
ax3.set_title("K-Means Clusters in QT + PCA Space")
ax3.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
ax3.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
fig3.tight_layout()
st.pyplot(fig3)
plt.close(fig3)
st.caption(f"2 PCA components capture **{sum(evr):.1%}** of variance from the QT-transformed features.")

# --- Section 4: Cluster Distribution ---
st.markdown("---")
st.subheader("4 · Cluster Size Distribution")

counts = pd.Series(clusters).value_counts().sort_index()
fig4, ax4 = plt.subplots(figsize=(8, 4))
bars = ax4.bar(counts.index.astype(str), counts.values, color=sns.color_palette("tab10", n_colors=len(counts)))
for b, v in zip(bars, counts.values):
    ax4.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:,}", ha="center", fontsize=9)
ax4.set_xlabel("Cluster"); ax4.set_ylabel("Track Count")
ax4.set_title("Number of Tracks per Cluster")
fig4.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

# --- Section 5: Feature Violin Plots ---
st.markdown("---")
st.subheader("5 · Feature Distributions by Cluster")
st.caption("Violin plots for key audio features split by cluster.")

KEY_FEATS = CLUSTER_FEATURE_COLS
violin_df = cluster_scaled[KEY_FEATS].copy()
violin_df["cluster"] = clusters

n_rows = (len(KEY_FEATS) + 2) // 3
fig5, axes5 = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
for ax, feat in zip(axes5.flatten()[:len(KEY_FEATS)], KEY_FEATS):
    sns.violinplot(data=violin_df, x="cluster", y=feat, hue="cluster", palette="tab10", inner="quartile", ax=ax, cut=0, legend=False)
    ax.set_title(feat.capitalize())
    ax.set_xlabel(""); ax.set_ylabel("")
for ax in axes5.flatten()[len(KEY_FEATS):]:
    ax.set_visible(False)
fig5.suptitle("Scaled Feature Distributions by Cluster", y=1.01)
fig5.tight_layout()
st.pyplot(fig5)
plt.close(fig5)

# --- Section 6: Cluster Personality Cards ---
st.markdown("---")
st.subheader("6 · Cluster Personality Summaries")

_TRAIT_MAP = {
    "energy": ("High energy", "Low energy"),
    "valence": ("Positive/happy", "Dark/melancholic"),
    "danceability": ("Danceable", "Less danceable"),
    "acousticness": ("Acoustic", "Electronic/produced"),
    "speechiness": ("Vocal/spoken-word", "Instrumental-leaning"),
    "liveness": ("Live/concert feel", "Studio recording"),
    "instrumentalness": ("Instrumental", "Vocal"),
    "tempo": ("Fast tempo", "Slow tempo"),
    "loudness": ("Loud", "Quiet"),
}

def _describe(row, threshold=0.3):
    traits = []
    for feat, (high, low) in _TRAIT_MAP.items():
        if feat in row.index:
            if row[feat] > threshold:
                traits.append(f"**{high}** ({row[feat]:+.2f})")
            elif row[feat] < -threshold:
                traits.append(f"**{low}** ({row[feat]:+.2f})")
    return traits if traits else ["Average across all features"]

cols = st.columns(min(4, K_FINAL))
for i in range(K_FINAL):
    with cols[i % len(cols)]:
        traits = _describe(profile[i])
        st.markdown(f"#### Cluster {i}")
        st.markdown(f"**{counts[i]:,}** tracks")
        for t in traits:
            st.markdown(f"- {t}")

# --- Section 7: Silhouette Analysis ---
st.markdown("---")
st.subheader("7 · Silhouette Analysis")

sample_sil = silhouette_samples(pca_df, clusters)
sil_df = pd.DataFrame({"cluster": clusters, "silhouette": sample_sil})
cluster_sil = sil_df.groupby("cluster")["silhouette"].mean().sort_index()

col7a, col7b = st.columns(2)

with col7a:
    st.caption("Mean silhouette per cluster. Higher = better-separated.")
    fig7a, ax7a = plt.subplots(figsize=(6, 4))
    colors7 = sns.color_palette("tab10", n_colors=K_FINAL)
    bars7 = ax7a.bar(cluster_sil.index.astype(str), cluster_sil.values, color=colors7)
    ax7a.axhline(sil, color="red", ls="--", lw=1, label=f"Overall: {sil:.3f}")
    for b, v in zip(bars7, cluster_sil.values):
        ax7a.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax7a.set_xlabel("Cluster"); ax7a.set_ylabel("Mean Silhouette")
    ax7a.set_title("Mean Silhouette per Cluster")
    ax7a.legend()
    fig7a.tight_layout()
    st.pyplot(fig7a)
    plt.close(fig7a)

with col7b:
    st.caption("Per-sample silhouette plot. Thin slivers or negative values indicate poorly-placed samples.")
    fig7b, ax7b = plt.subplots(figsize=(6, 5))
    y_lower = 0
    for i in range(K_FINAL):
        mask = clusters == i
        vals = np.sort(sample_sil[mask])
        y_upper = y_lower + len(vals)
        ax7b.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7, color=colors7[i], label=f"Cluster {i}")
        ax7b.text(-0.05, y_lower + len(vals) / 2, str(i), fontsize=10, fontweight="bold", va="center")
        y_lower = y_upper + 50
    ax7b.axvline(sil, color="red", ls="--", lw=1, label=f"Mean: {sil:.3f}")
    ax7b.set_xlabel("Silhouette Coefficient")
    ax7b.set_ylabel("Samples (sorted)")
    ax7b.set_title("Per-Sample Silhouette Plot")
    ax7b.set_yticks([])
    ax7b.legend(loc="lower right", fontsize=8)
    fig7b.tight_layout()
    st.pyplot(fig7b)
    plt.close(fig7b)

st.markdown("---")
st.caption("Pipeline: 7 core audio features (danceability, energy, loudness, speechiness, acousticness, liveness, valence) → QuantileTransformer (Gaussian) → PCA (2 components) → K-Means (k=5). Random 50k sample from 551k tracks.")
