# clustering_utils.py
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, inconsistent, fcluster


from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)
from src.metrics import (
    compute_gap_statistic,
    compute_wcss_per_cluster,
    compute_unbalanced_factor,
    compute_gmm_bic,
    compute_gmm_aic,
    compute_avg_log_likelihood_per_component,
    compute_dbscan_noise_core_border,
)
from src.plotter import plot_all_and_unique_metrics


def compute_metrics_over_k(clusterer_cls, X, k_range, **clusterer_kwargs):
    """
    For each k in k_range, compute:
      - inertia via virtual centroids
      - silhouette, Calinski–Harabasz, Davies–Bouldin
      - avg distance ± std to real centroids
      - cluster fraction ± std
    Returns a DataFrame indexed by n_clusters.
    """
    records = []
    n_samples = X.shape[0]

    for k in k_range:
        cl = clusterer_cls(n_clusters=k, **clusterer_kwargs)
        cl.fit(X)
        m = cl.get_metrics()

        # 1) inertia via virtual centroids
        pts, lbls = cl.X_, cl.labels_
        virt      = cl.get_virtual_centroids()
        labels    = [l for l in np.unique(lbls) if l >= 0]
        cmap = {lab: virt[i] for i, lab in enumerate(labels)}

        mask  = lbls >= 0
        pts0  = pts[mask]
        lbls0 = lbls[mask]
        cents0 = np.vstack([cmap[l] for l in lbls0])
        inertia = float(np.sum((pts0 - cents0) ** 2))

        # 2) distances to real centroids
        real   = cl.get_real_centroids()
        rcmap  = {lab: real[i] for i, lab in enumerate(labels)}
        dists  = [np.linalg.norm(x - rcmap[l])
                  for x, l in zip(pts, lbls) if l >= 0]
        dist_mean = float(np.mean(dists)) if dists else np.nan
        dist_std  = float(np.std(dists))  if dists else np.nan

        # 3) cluster fractions
        pops      = list(m["population"].values()) if m["population"] else []
        fracs     = [p / n_samples for p in pops]
        frac_mean = float(np.mean(fracs)) if fracs else np.nan
        frac_std  = float(np.std(fracs))  if fracs else np.nan

        records.append({
            "n_clusters":          k,
            "inertia":             inertia,
            "silhouette":          m["silhouette"],
            "calinski_harabasz":   m["calinski_harabasz"],
            "davies_bouldin":      m["davies_bouldin"],
            "avg_distance_mean":   dist_mean,
            "avg_distance_std":    dist_std,
            "cluster_frac_mean":   frac_mean,
            "cluster_frac_std":    frac_std,
        })

    df = pd.DataFrame(records).set_index("n_clusters", drop=False)
    return df


def compute_consensus_score(df):
    """
    Normalize each metric to [0,1], invert where lower-is-better,
    average, then rescale so that raw 0.5→0 and raw 1→1.
    """
    cols = [
        'inertia',
        'silhouette',
        'calinski_harabasz',
        'davies_bouldin',
        'avg_distance_mean',
        'cluster_frac_std'
    ]
    norm_df = pd.DataFrame(index=df.index)

    for c in cols:
        s = df[c].replace([np.inf, -np.inf], np.nan)
        mn, mx = s.min(skipna=True), s.max(skipna=True)

        if pd.isna(mn) or pd.isna(mx) or mn == mx:
            norm = pd.Series(0.5, index=df.index)
        else:
            # lower-is-better metrics
            if c in ['inertia', 'davies_bouldin', 'avg_distance_mean', 'cluster_frac_std']:
                norm = (mx - s) / (mx - mn)
            else:
                norm = (s - mn) / (mx - mn)

        norm_df[c] = norm.fillna(0.5)

    raw = norm_df.mean(axis=1)
    final = (raw - 0.5) * 2
    return final.clip(0, 1)


def run_and_report(name, cls, X, k_range, **kwargs):
    """
    Runs the k-range pipeline for KMeans, Agglomerative, or GMM:
      1) compute metrics over k
      2) compute & save consensus score
      3) pick best k, plot 3×3 grid, highlight best k
      4) refit final model, save centroids
    """
    print(f"\n\n=== {name.upper()} ===")

    # common metrics + consensus
    df = compute_metrics_over_k(cls, X, k_range, **kwargs)
    df['consensus_score'] = compute_consensus_score(df)
    df.to_csv(f"{name}_metrics.csv", index=False)
    print(f"Saved metrics to {name}_metrics.csv")

    # best k by consensus
    best_k = int(df.loc[df['consensus_score'].idxmax(), 'n_clusters'])
    print(f"Best k by consensus: {best_k}")

    # assemble unique‐metric arrays
    if cls is AgglomerativeClusterer:
        Z   = linkage(X, method='ward')
        inc = inconsistent(Z)
        n   = len(X)

        cut_heights, cut_incons, size_std = [], [], []
        for k in k_range:
            idx = n - 1 - k
            cut_heights.append(float(Z[idx, 2]))
            cut_incons .append(float(inc[idx, 3]))

            cl_k = cls(n_clusters=k)
            cl_k.fit(X)
            pops = list(cl_k.get_metrics()['population'].values())
            size_std.append(float(np.std(pops)) if pops else np.nan)

        unique_vals   = [cut_heights, cut_incons, size_std]
        unique_errs   = [None, None, None]
        unique_titles = [
            "Dendrogram Cut Height",
            "Inconsistency at Cut",
            "Cluster-Size Std"
        ]

    elif cls is KMeansClusterer:
        gaps, ubs, wcss_means, wcss_stds = [], [], [], []
        for k in k_range:
            km = cls(n_clusters=k, **kwargs)
            km.fit(X)

            gaps.append(compute_gap_statistic(X, cls, k, B=10, random_state=0))
            ubs.append(compute_unbalanced_factor(km.labels_))

            wcss = compute_wcss_per_cluster(
                X, km.labels_, km.get_virtual_centroids()
            )
            vals = list(wcss.values())
            wcss_means.append(float(np.mean(vals)))
            wcss_stds .append(float(np.std(vals)))

        unique_vals   = [gaps, ubs, wcss_means]
        unique_errs   = [None, None, wcss_stds]
        unique_titles = [
            "Gap Statistic",
            "Unbalanced Factor",
            "Avg WCSS per Cluster"
        ]

    elif cls is GMMClusterer:
        bics, aics, avgll_means, avgll_stds = [], [], [], []
        for k in k_range:
            gm = cls(n_clusters=k, **kwargs)
            gm.fit(X)

            bics.append(compute_gmm_bic(gm.model, X))
            aics.append(compute_gmm_aic(gm.model, X))

            comp_ll = compute_avg_log_likelihood_per_component(gm.model, X)
            vals = list(comp_ll.values())
            avgll_means.append(float(np.mean(vals)))
            avgll_stds .append(float(np.std(vals)))

        unique_vals   = [bics, aics, avgll_means]
        unique_errs   = [None, None, avgll_stds]
        unique_titles = [
            "GMM BIC",
            "GMM AIC",
            "Avg Log-Lik per Component"
        ]

    else:
        # should not happen here for k-range methods
        unique_vals   = [[np.nan] * len(k_range)] * 3
        unique_errs   = [None] * 3
        unique_titles = ["N/A"] * 3

    # plot and highlight
    fig = plot_all_and_unique_metrics(
        df,
        unique_vals,
        unique_errs,
        unique_titles,
        fontsize=12,
        use_tex=True,
        linewidth=2,
        capsize=4
    )
    for ax in fig.axes:
        ax.axvline(best_k, color='red', linestyle='--', linewidth=1)

    fig.savefig(f"{name}_3x3_metrics.png")
    plt.close(fig)

    # final fit & centroids
    final = cls(n_clusters=best_k, **kwargs)
    final.fit(X)
    virt  = final.get_virtual_centroids()
    real  = final.get_real_centroids()

    pd.DataFrame(virt).to_csv(f"{name}_virtual_centroids.csv", index=False)
    pd.DataFrame(real).to_csv( f"{name}_real_centroids.csv",    index=False)
    print(f"Saved centroids to {name}_virtual_centroids.csv and {name}_real_centroids.csv")


def run_dbscan(name, X, eps=0.7, min_samples=5):
    """
    Single-run DBSCAN: compute metrics, consensus, and save CSVs + centroids.
    """
    print(f"\n\n=== {name.upper()} ===")
    db = DBSCANClusterer(eps=eps, min_samples=min_samples)
    db.fit(X)
    mdb = db.get_metrics()
    print(mdb)

    virt = db.get_virtual_centroids()
    real = db.get_real_centroids()
    print("Virtual centroids:\n", virt)
    print("Real centroids:\n",    real)

    # safe statistics
    pop_vals  = np.array(list(mdb["population"].values()),    dtype=float)
    dist_vals = np.array(list(mdb["avg_distance"].values()), dtype=float)

    def safe_mean(a): return float(a.mean()) if a.size else np.nan
    def safe_std(a):  return float(a.std())  if a.size else np.nan

    db_df = pd.DataFrame([{
        "inertia":            np.nan,
        "silhouette":         mdb["silhouette"],
        "calinski_harabasz":  mdb["calinski_harabasz"],
        "davies_bouldin":     mdb["davies_bouldin"],
        "avg_distance_mean":  safe_mean(dist_vals),
        "avg_distance_std":   safe_std(dist_vals),
        "cluster_frac_mean":  safe_mean(pop_vals) / X.shape[0],
        "cluster_frac_std":   safe_std(pop_vals)  / X.shape[0],
    }])
    db_df["consensus_score"] = compute_consensus_score(db_df)
    db_df.to_csv(f"{name}_metrics.csv", index=False)
    print(f"Saved DBSCAN metrics to {name}_metrics.csv")

    pd.DataFrame(virt).to_csv(f"{name}_virtual_centroids.csv", index=False)
    pd.DataFrame(real).to_csv( f"{name}_real_centroids.csv",    index=False)
    print(f"Saved DBSCAN centroids to {name}_virtual_centroids.csv and {name}_real_centroids.csv")

def assign_clusters(
    clusterer_cls,
    X,
    *,
    n_clusters: int = None,
    cutoff_height: float = None,
    eps: float = None,
    min_samples: int = None,
    random_state: int = None,
    method: str = 'ward',
    metric: str = 'euclidean',
    **kwargs
) -> np.ndarray:
    """
    Fit a clustering algorithm and return labels for each sample in X.

    Args:
      clusterer_cls: one of KMeansClusterer, AgglomerativeClusterer,
                     GMMClusterer, or DBSCANClusterer.
      X            : (n_samples, n_features) data array.
      n_clusters   : number of clusters (for k-based methods).
      cutoff_height: linkage cut height (for hierarchical only).
      eps          : eps parameter (for DBSCAN only).
      min_samples  : min_samples parameter (for DBSCAN only).
      random_state : random seed for k-based methods.
      method, metric: passed to scipy linkage for hierarchical if cutoff_height is used.

    Returns:
      labels: array of shape (n_samples,) with integer cluster labels.
    """

    # 1) DBSCAN
    if clusterer_cls is DBSCANClusterer:
        cl = DBSCANClusterer(eps=eps, min_samples=min_samples)
        cl.fit(X)
        return cl.labels_

    # 2) Agglomerative (hierarchical)
    if clusterer_cls is AgglomerativeClusterer:
        # If user supplied a cutoff height, do SciPy linkage + fcluster
        if cutoff_height is not None:
            Z = linkage(X, method=method, metric=metric)
            # criterion='distance' cuts so that all merges above cutoff are separate clusters
            labels = fcluster(Z, t=cutoff_height, criterion='distance')
            return labels - 1  # make zero-based
        # otherwise fall back to n_clusters
        if n_clusters is None:
            raise ValueError("Must supply n_clusters or cutoff_height for hierarchical clustering")
        cl = AgglomerativeClusterer(n_clusters=n_clusters)
        cl.fit(X)
        return cl.labels_

    # 3) K-means
    if clusterer_cls is KMeansClusterer:
        if n_clusters is None:
            raise ValueError("Must supply n_clusters for KMeansClusterer")
        cl = KMeansClusterer(n_clusters=n_clusters, random_state=random_state, **kwargs)
        cl.fit(X)
        return cl.labels_

    # 4) Gaussian Mixture
    if clusterer_cls is GMMClusterer:
        if n_clusters is None:
            raise ValueError("Must supply n_clusters for GMMClusterer")
        cl = GMMClusterer(n_clusters=n_clusters, random_state=random_state, **kwargs)
        cl.fit(X)
        return cl.labels_

    raise ValueError(f"Unsupported clusterer class: {clusterer_cls}")


def run_method(
    name,
    clusterer_cls,
    X,
    k_range=None,
    eps=0.7,
    min_samples=5,
    **kwargs
):
    """
    Run exactly one clustering method end-to-end:
      - If clusterer_cls == DBSCANClusterer, does a single-run DBSCAN.
      - Otherwise requires k_range, does the k-range pipeline.

    Args:
      name         : str
        Prefix for output files (e.g. "kmeans", "dbscan").
      clusterer_cls: class
        One of KMeansClusterer, AgglomerativeClusterer, GMMClusterer, or DBSCANClusterer.
      X            : array-like, shape (n_samples, n_features)
      k_range      : iterable of int, optional
        Required for non-DBSCAN methods.
      eps          : float
        Only used if clusterer_cls is DBSCANClusterer.
      min_samples  : int
        Only used if clusterer_cls is DBSCANClusterer.
      **kwargs     : passed to run_and_report for k-based methods
    """
    # DBSCAN
    if clusterer_cls is DBSCANClusterer:
        run_dbscan(
            name=name,
            X=X,
            eps=eps,
            min_samples=min_samples
        )
    else:
        if k_range is None:
            raise ValueError("k_range must be provided for non-DBSCAN methods")
        run_and_report(
            name=name,
            cls=clusterer_cls,
            X=X,
            k_range=k_range,
            **kwargs
        )

def save_cluster_labels(
    X: np.ndarray,
    labels: np.ndarray,
    filepath: str
) -> None:
    """
    Dump a CSV of X coordinates with their cluster label.

    Args:
      X        : array, shape (n_samples, n_features)
      labels   : int array, shape (n_samples,)
      filepath : path to output CSV; columns = ['x0','x1',…,'label']
    """
    n_features = X.shape[1]
    data = { f"x{i}": X[:, i] for i in range(n_features) }
    data['label'] = labels
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def plot_clusters(
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray = None,
        title: str = None,
        palette: list = None,
        figsize: tuple = (6, 6),
        savepath: str = None,
        point_size: int = 20,
        alpha: float = 0.7
) -> plt.Axes:
    """
    Scatter‐plot X colored by `labels`.  Optionally overplot `centroids`.

    Args:
      X          : array-like, shape (n_samples, 2)
      labels     : int array, shape (n_samples,)
      centroids  : array, shape (n_clusters, 2), optional
      title      : figure title
      palette    : list of colors (len >= n_clusters+1), defaults to tab10
      figsize    : figure size
      savepath   : if given, calls fig.savefig(savepath)
      point_size : marker size for data points
      alpha      : point transparency

    Returns:
      ax : the matplotlib Axes instance
    """
    # 1) Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # 2) Unique labels (noise = -1)
    uniq = np.unique(labels)
    n_clusters = len(uniq[uniq >= 0])

    # 3) Choose palette
    if palette is None:
        # Use tab10, reserve index 0 for noise if present
        base = cm.get_cmap('tab10').colors
        palette = list(base) + ['#444444']  # last color for noise

    # 4) Plot each cluster
    for lab in uniq:
        mask = labels == lab
        col = palette[int(lab)] if lab >= 0 and lab < len(palette) else palette[-1]
        label_text = f"Cluster {lab}" if lab >= 0 else "Noise"
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=[col],
            s=point_size,
            alpha=alpha,
            label=label_text,
            edgecolor='k' if lab >= 0 else None,
            linewidth=0.2
        )

    # 5) Optionally plot centroids
    if centroids is not None:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            c='none',
            edgecolor='black',
            s=200,
            marker='X',
            linewidth=1.5,
            label='Centroids'
        )

    # 6) Final touches
    ax.set_aspect('equal', 'box')
    if title:
        ax.set_title(title)
    ax.legend(loc='best', fontsize='small', framealpha=0.8)
    ax.grid(True)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath)

    return ax
