# Clustering_quality

A modular Python toolkit for clustering diagnostics, consensus-based best-k selection, plotting, and export of centroids and assignments.

---

## Table of Contents

- [Installation](#installation)  
- [Overview](#overview)  
- [Quickstart](#quickstart)  
- [Core Modules](#core-modules)  
  - [clusterer.py](#clustererpy)  
  - [metrics.py](#metricspy)  
  - [clustering_utils.py](#clustering_utilspy)  
  - [selection.py](#selectionpy)  
  - [plotter.py](#plotterpy)  
- [Example Pipelines](#example-pipelines)  
  - [Running End-to-End with `run_and_report`](#running-end-to-end-with-run_and_report)  
  - [Interactive Usage](#interactive-usage)  
- [Testing & CI](#testing--ci)  
- [License](#license)  

---

## Installation

```bash
git clone https://github.com/Lapsis-glitch/Clustering_quality.git
cd Clustering_quality
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` pins:

```
numpy==1.24.3
pandas==2.2.3
scipy==1.10.1
scikit-learn==1.5.2
matplotlib==3.9.2
kneed==0.8.5
pytest==8.4.1
```

---

## Overview

1. **clusterer.py**  
   Wrappers for KMeans, Agglomerative, GMM, and DBSCAN with a unified  
   `.fit()`, `.labels_`, virtual/real-centroid API, and per-metric diagnostics.

2. **metrics.py**  
   Stand-alone functions: cophenetic correlation, inconsistency, gap statistic,  
   WCSS, unbalanced factor, BIC/AIC, average log-likelihood, DBSCAN noise/core/border.

3. **clustering_utils.py**  
   - Compute diagnostics over a range of k (`compute_metrics_over_k`)  
   - Consensus ranking: Borda, TOPSIS, vote, hybrid vote+elbow (`compute_consensus_score`)  
   - Assign clusters + export CSVs (`assign_clusters`)  
   - End-to-end pipeline: metrics CSV, consensus, centroid CSVs, 3×3 diagnostic plot (`run_and_report`)

4. **selection.py**  
   Quick helpers for manual selection:  
   - `plot_elbow(X, k_range, random_state)`  
   - `plot_dendrogram(X, method, metric)`

5. **plotter.py**  
   Generates a 3×3 grid of six common metrics + three unique metrics,  
   with optional per-metric best-k annotation.

---

## Quickstart


## Core Modules

### clusterer.py

```python
from clusterer import KMeansClusterer, AgglomerativeClusterer

# KMeans example
km = KMeansClusterer(n_clusters=4, random_state=0)
km.fit(X)                           # X: (n_samples, n_features)
labels = km.labels_                # array of length n_samples
virt    = km.get_virtual_centroids()  # (4, n_features)
real    = km.get_real_centroids()     # (4, n_features)
metrics = km.get_metrics()         # dict of diagnostics
```

`get_metrics()` always includes:

- `silhouette`, `calinski_harabasz`, `davies_bouldin`  
- `population` (cluster sizes)  
- `avg_distance` (mean distance to real centroids)

### metrics.py

```python
from metrics import (
  compute_cophenetic_correlation,
  compute_inconsistency_stats,
  compute_dendrogram_cut_height_stats,
  compute_gap_statistic,
  compute_wcss_per_cluster,
  compute_unbalanced_factor,
  compute_gmm_bic,
  compute_gmm_aic,
  compute_avg_log_likelihood_per_component,
  compute_dbscan_noise_core_border
)

# Example usage
corr = compute_cophenetic_correlation(X, method='ward')
gap  = compute_gap_statistic(X, KMeansClusterer, k=3, B=20, random_state=0)
```

### clustering_utils.py

#### compute_metrics_over_k

```python
from clustering_utils import compute_metrics_over_k

df = compute_metrics_over_k(
    KMeansClusterer,
    X,
    k_range=range(2, 11),
    random_state=42,
    sieve=10
)
```

#### compute_consensus_score

```python
from clustering_utils import compute_consensus_score

scores = compute_consensus_score(df, method='borda')
# or with per-metric best k:
scores, best_map = compute_consensus_score(
  df,
  method='vote_elbow',
  elbow_metrics=['inertia'],
  return_best_k_map=True
)
```

#### assign_clusters

```python
from clustering_utils import assign_clusters

labels_full, cl = assign_clusters(
    KMeansClusterer,
    X,
    n_clusters=best_k,
    random_state=42,
    centroids_csv="centroids.csv",
    clusters_csv="assignments.csv"
)
```

#### run_and_report

```python
from clustering_utils import run_and_report

run_and_report(
    name="kmeans",
    cls=KMeansClusterer,
    X=X,
    k_range=range(2,11),
    random_state=42,
    consensus_method="vote_elbow",
    consensus_weights=None,
    elbow_metrics=["inertia"]
)
```

Saves:

- `{name}_metrics.csv`  
- `{name}_virtual_centroids.csv`  
- `{name}_real_centroids.csv`  
- `{name}_3x3_metrics.png`

### selection.py

```python
from selection import plot_elbow, plot_dendrogram

fig1 = plot_elbow(X, k_range=range(1,11), random_state=0)
fig2 = plot_dendrogram(X, method='ward', metric='euclidean')
```

### plotter.py

```python
from plotter import plot_all_and_unique_metrics

fig = plot_all_and_unique_metrics(
    metrics_df=df,
    unique_vals=[coph, inc, mh],
    unique_errs=[None, None, None],
    unique_titles=['Cophenetic Correlation','Inconsistency','Merge Height'],
    best_k_map=best_map
)
```

---

## Example Pipelines

### Minimal working example

```python
import matplotlib.pyplot as plt
from src.clustering_utils import run_method, assign_clusters_with_sieve, assign_clusters
from src.plotter import plot_clusters
from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)
from src.synthetic_data import generate_challenging_dataset

def main():
    # 1) Generate the challenging dataset
    X = generate_challenging_dataset(random_state=123)

    # 2) Optional: visualize the dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.6)
    plt.title("Challenging Synthetic Dataset")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("challenging_dataset.png")
    plt.close()

    # 3) Define k‐range
    k_range = range(2, 11)

    # 4) Run clustering pipelines
    run_method(
        name="kmeans_challenge_sieve",
        clusterer_cls=KMeansClusterer,
        X=X,
        k_range=k_range,
        random_state=123,
        sieve = 10, # only keep every 10th point for metrics
        consensus_method="vote",  # or "borda"
        consensus_weights=None # only for TOPSIS
    )

    #Other examples below (commented out to save time)
    # run_method(
    #     name="agglo_challenge",
    #     clusterer_cls=AgglomerativeClusterer,
    #     X=X,
    #     k_range=k_range
    # )
    # run_method(
    #     name="gmm_challenge",
    #     clusterer_cls=GMMClusterer,
    #     X=X,
    #     k_range=k_range,
    #     random_state=123
    # )
    # run_method(
    #     name="dbscan_challenge",
    #     clusterer_cls=DBSCANClusterer,
    #     X=X,
    #     eps=0.8,
    #     min_samples=10
    # )

    # Run final clustering to obtain labels and centroids
    labels, cl = assign_clusters(
            KMeansClusterer,
            X,
            random_state=123,
            n_clusters=5,
            centroids_csv="kmeans_5_centroids.csv",
            clusters_csv="kmeans_5_point_assignments.csv"
        )

    centroids = cl.get_virtual_centroids()

    # 5) Plot final clustering if there is 2D data
    plot_clusters(X,labels,centroids, savepath="clusters.png")


if __name__ == "__main__":
    main()

```

---

## Testing & CI

- Tests are under `tests/`  
- Run:  
  ```bash
  pytest tests/
  ```
- GitHub Actions CI workflow is included in `.github/workflows/ci.yml`

---

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE v3. See [LICENSE](LICENSE) for details.