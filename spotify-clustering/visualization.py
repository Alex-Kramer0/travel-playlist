import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_elbow_silhouette(k_eval: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))

    color = "tab:blue"
    ax1.plot(k_eval["k"], k_eval["inertia"], marker="o", color=color)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.plot(k_eval["k"], k_eval["silhouette"], marker="s", color=color)
    ax2.set_ylabel("Silhouette Score", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Elbow vs. Silhouette Analysis (sampled data)")
    plt.tight_layout()
    plt.show()


def plot_cluster_heatmap(
    cluster_profile: pd.DataFrame,
    title: str = "Cluster Mean (z-score) per Audio Feature",
) -> None:
    n_clusters = cluster_profile.shape[1]
    plt.figure(figsize=(max(10, 2 + 1.5 * n_clusters), 6))
    sns.heatmap(cluster_profile, annot=True, cmap="coolwarm", center=0)
    plt.title(title)
    plt.ylabel("Feature")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(
    scaled_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_col: str = "cluster",
    random_state: int = 42,
) -> None:
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(scaled_df)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df[cluster_col] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=cluster_col,
        palette="tab10",
        alpha=0.6,
        s=40,
    )
    plt.title("K-means Clusters Projected via PCA")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")


def plot_kdistance_curve(kth_distances: np.ndarray, k: int) -> None:
    x_vals = np.arange(1, len(kth_distances) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, kth_distances)
    plt.xlabel("Points sorted by kth-distance")
    plt.ylabel(f"Distance to {k}th neighbor")
    plt.title("k-distance curve (PCA sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dbscan_sweep_heatmaps(dbscan_results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot_noise = dbscan_results.pivot(index="min_samples", columns="eps", values="noise_ratio")
    pivot_db = dbscan_results.pivot(index="min_samples", columns="eps", values="davies_bouldin")

    sns.heatmap(pivot_noise, ax=axes[0], annot=True, fmt=".2f", cmap="viridis")
    axes[0].set_title("Noise ratio")
    axes[0].set_ylabel("min_samples")
    axes[0].set_xlabel("eps")

    sns.heatmap(pivot_db, ax=axes[1], annot=True, fmt=".2f", cmap="magma_r")
    axes[1].set_title("Davies-Bouldin (lower better)")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("eps")

    plt.tight_layout()
    plt.show()
