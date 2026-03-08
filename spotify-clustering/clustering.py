import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from IPython.display import display
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


def evaluate_kmeans(
    scaled_df: pd.DataFrame,
    k_values: range = range(2, 11),
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    data = scaled_df
    if sample_size is not None:
        data = scaled_df.sample(n=min(sample_size, len(scaled_df)), random_state=random_state)

    inertias, silhouette_scores, progress_log = [], [], []

    for k in tqdm(k_values, desc="Evaluating k", unit="k"):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto", max_iter=300)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(data, labels)
        silhouette_scores.append(sil)
        progress_log.append({"k": k, "inertia": kmeans.inertia_, "silhouette": sil})
        display(pd.DataFrame(progress_log).tail(1))

    return pd.DataFrame({
        "k": list(k_values),
        "inertia": inertias,
        "silhouette": silhouette_scores,
    })


def fit_kmeans(
    scaled_df: pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(scaled_df)
    return kmeans, labels


def build_cluster_profile(
    scaled_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    df = scaled_df.copy()
    df[cluster_col] = labels
    profile = df.groupby(cluster_col).mean().T
    return profile


def build_pca_basis(
    scaled_df: pd.DataFrame,
    n_components: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    matrix = pca.fit_transform(scaled_df)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(matrix, columns=columns)
    explained = pca.explained_variance_ratio_.sum()
    print(f"Top {n_components} PCs capture {explained:.1%} of total variance")
    return pca_df, pca


def compute_kdistance(
    pca_df: pd.DataFrame,
    k: int = 15,
    sample_size: int = 20000,
    knee_percentile: int = 60,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    sample = pca_df.sample(n=min(sample_size, len(pca_df)), random_state=random_state)
    matrix = sample.to_numpy()

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(matrix)
    distances, _ = nn.kneighbors(matrix)

    kth_distances = np.sort(distances[:, -1])
    knee_eps = np.percentile(kth_distances, knee_percentile)

    print(
        f"{k}th-neighbor distance stats → "
        f"min: {kth_distances.min():.4f}, "
        f"median: {np.median(kth_distances):.4f}, "
        f"max: {kth_distances.max():.4f}"
    )
    print(f"Suggested eps around the {knee_percentile}th percentile: {knee_eps:.4f}")
    return kth_distances, knee_eps


def sweep_dbscan(
    pca_df: pd.DataFrame,
    knee_eps: float,
    min_samples_values: list[int] = [8, 12, 16, 20],
    eps_span: float = 0.6,
    eps_shrink: float = 0.4,
    max_attempts: int = 6,
    min_valid_clusters: int = 4,
    sweep_sample_size: int = 50000,
    random_state: int = 7,
) -> pd.DataFrame:
    sample = pca_df.sample(n=min(sweep_sample_size, len(pca_df)), random_state=random_state)
    sample_matrix = sample.to_numpy()

    def _run_grid(eps_values: np.ndarray) -> pd.DataFrame:
        rows = []
        for eps_val, min_s in product(eps_values, min_samples_values):
            model = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
            labels = model.fit_predict(sample_matrix)
            noise_mask = labels == -1
            unique_clusters = np.setdiff1d(np.unique(labels), [-1])
            cluster_count = len(unique_clusters)
            db_score = (
                davies_bouldin_score(sample_matrix[~noise_mask], labels[~noise_mask])
                if cluster_count >= 2
                else np.nan
            )
            rows.append({
                "eps": round(eps_val, 5),
                "min_samples": min_s,
                "clusters": cluster_count,
                "noise_ratio": noise_mask.mean(),
                "davies_bouldin": db_score,
            })
        return pd.DataFrame(rows)

    eps_center = knee_eps
    dbscan_results = None

    for attempt in range(max_attempts):
        eps_values = np.linspace(
            eps_center * (1 - eps_span),
            eps_center * (1 + eps_span),
            num=7,
        )
        current_results = _run_grid(eps_values)
        max_clusters = current_results["clusters"].max()
        noise_min = current_results["noise_ratio"].min()
        noise_max = current_results["noise_ratio"].max()
        print(
            f"Attempt {attempt + 1}: eps range [{eps_values.min():.5f}, {eps_values.max():.5f}] → "
            f"max clusters {max_clusters}, noise ratio span {noise_min:.1%}–{noise_max:.1%}"
        )
        if max_clusters >= min_valid_clusters:
            dbscan_results = current_results
            break
        eps_center *= eps_shrink

    if dbscan_results is None:
        raise RuntimeError(
            "DBSCAN grid never produced the desired number of clusters. "
            "Consider revisiting the PCA basis or search space."
        )

    print(f"Selected grid from attempt {attempt + 1} with eps center {eps_center:.5f}")
    display(dbscan_results.sort_values(["davies_bouldin", "noise_ratio"], na_position="last"))
    return dbscan_results


def select_best_dbscan_params(
    dbscan_results: pd.DataFrame,
    target_cluster_min: int = 4,
    target_cluster_max: int = 6,
    target_noise_max: float = 0.40,
) -> tuple[float, int]:
    ranked = dbscan_results.dropna(subset=["davies_bouldin"]).copy()
    target_mask = ranked["clusters"].between(target_cluster_min, target_cluster_max)
    noise_mask = ranked["noise_ratio"] <= target_noise_max
    selection_mask = target_mask & noise_mask

    if selection_mask.any():
        candidates = ranked[selection_mask]
        print(
            f"Selecting best grid point within {target_cluster_min}-{target_cluster_max} "
            f"clusters and ≤{target_noise_max:.0%} noise"
        )
    elif target_mask.any():
        candidates = ranked[target_mask]
        print(
            "Warning: no grid point satisfied the noise threshold; "
            "falling back to the best option in the desired cluster range."
        )
    else:
        candidates = ranked
        print(
            "Warning: no grid point achieved the target cluster range; "
            "selecting best overall result instead."
        )

    best_row = candidates.sort_values(
        ["davies_bouldin", "noise_ratio", "clusters"],
        ascending=[True, True, False],
    ).iloc[0]

    best_eps = float(best_row["eps"])
    best_min_samples = int(best_row["min_samples"])
    print(
        f"Best grid point (sample) — eps: {best_eps:.3f}, "
        f"min_samples: {best_min_samples}, "
        f"clusters: {int(best_row['clusters'])}, "
        f"noise ratio: {best_row['noise_ratio']:.2%}, "
        f"Davies-Bouldin: {best_row['davies_bouldin']:.3f}"
    )
    return best_eps, best_min_samples


def fit_dbscan_full(
    pca_df: pd.DataFrame,
    eps: float,
    min_samples: int,
) -> dict:
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(pca_df.values)
    noise_mask = labels == -1
    cluster_ids = np.setdiff1d(np.unique(labels), [-1])
    return {
        "eps": eps,
        "min_samples": min_samples,
        "labels": labels,
        "noise_mask": noise_mask,
        "cluster_count": len(cluster_ids),
        "noise_ratio": noise_mask.mean(),
    }


def fit_dbscan_with_adjustment(
    pca_df: pd.DataFrame,
    initial_eps: float,
    min_samples: int,
    target_cluster_min: int = 4,
    target_cluster_max: int = 6,
    max_adjustments: int = 8,
) -> dict:
    desired_mid = (target_cluster_min + target_cluster_max) / 2
    fit_history = []

    result = fit_dbscan_full(pca_df, initial_eps, min_samples)
    fit_history.append(result)
    current_eps = initial_eps

    for attempt in range(1, max_adjustments + 1):
        if target_cluster_min <= result["cluster_count"] <= target_cluster_max:
            break
        current_eps *= 1.25 if result["cluster_count"] > target_cluster_max else 0.8
        print(f"Adjustment {attempt}: trying eps {current_eps:.3f}")
        result = fit_dbscan_full(pca_df, current_eps, min_samples)
        fit_history.append(result)

    if not (target_cluster_min <= result["cluster_count"] <= target_cluster_max):
        print(
            "Warning: unable to land exactly in the target cluster band; "
            "selecting the closest attempt by cluster count and noise ratio."
        )
        result = min(
            fit_history,
            key=lambda r: (abs(r["cluster_count"] - desired_mid), r["noise_ratio"]),
        )

    return result
