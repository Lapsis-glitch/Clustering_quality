# test_clustering_utils.py

import numpy as np
import pandas as pd
import os
import pytest
from sklearn.datasets import make_blobs

import src.clustering_utils as cu
from src.clustering_utils import run_method
from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)


@pytest.fixture
def X():
    # small synthetic dataset with 3 centers
    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=0)
    return X


def test_compute_metrics_over_k_kmeans(X):
    # test that compute_metrics_over_k returns expected columns and indices
    df = cu.compute_metrics_over_k(KMeansClusterer, X, k_range=range(2, 5), random_state=0)
    # Should have one row per tested k
    assert list(df["n_clusters"]) == [2, 3, 4]
    # All common metric columns must be present
    expected_cols = {
        "n_clusters", "inertia", "silhouette", "calinski_harabasz",
        "davies_bouldin", "avg_distance_mean", "avg_distance_std",
        "cluster_frac_mean", "cluster_frac_std"
    }
    assert expected_cols.issubset(df.columns)
    # Ensure no column is entirely NaN
    for col in expected_cols - {"n_clusters"}:
        assert not df[col].isna().all()


def test_compute_consensus_score_monotonicity():
    # Build a toy DataFrame where metrics linearly increase/decrease
    df = pd.DataFrame({
        "inertia":            [1.0, 2.0, 3.0],  # lower is better
        "silhouette":         [3.0, 2.0, 1.0],  # higher is better
        "calinski_harabasz":  [1.0, 2.0, 3.0],  # higher is better
        "davies_bouldin":     [3.0, 2.0, 1.0],  # lower is better
        "avg_distance_mean":  [1.0, 2.0, 3.0],  # lower is better
        "cluster_frac_std":   [3.0, 2.0, 1.0],  # lower is better
    })
    scores = cu.compute_consensus_score(df)
    # Should be between 0 and 1
    assert all(0.0 <= s <= 1.0 for s in scores)
    # The middle row is exactly average → raw=0.5 → final score=0
    assert pytest.approx(scores.iloc[1], abs=1e-8) == 0.0
    # The best “ideal” row (first or last) should get the highest score
    assert scores.max() == scores.iloc[[0,2]].max()


@pytest.mark.parametrize("clusterer_cls, name, extras", [
    (KMeansClusterer,       "kmeans",       {"random_state": 0}),
    (AgglomerativeClusterer,"agglomerative",{}),
    (GMMClusterer,          "gmm",          {"random_state": 0}),
])
def test_run_method_k_range(tmp_path, monkeypatch, X, clusterer_cls, name, extras):
    # Use run_method to execute k-range pipeline and verify CSVs + PNG are created
    monkeypatch.chdir(tmp_path)
    k_range = range(2, 5)
    run_method(
        name=name,
        clusterer_cls=clusterer_cls,
        X=X,
        k_range=k_range,
        **extras
    )

    # Check metrics CSV
    metrics_csv = tmp_path / f"{name}_metrics.csv"
    assert metrics_csv.exists()
    df = pd.read_csv(metrics_csv)
    # It must have one row per k and a consensus_score column
    assert list(df["n_clusters"]) == list(k_range)
    assert "consensus_score" in df.columns

    # Check centroids CSVs
    virt_csv = tmp_path / f"{name}_virtual_centroids.csv"
    real_csv = tmp_path / f"{name}_real_centroids.csv"
    assert virt_csv.exists()
    assert real_csv.exists()
    df_virt = pd.read_csv(virt_csv)
    df_real = pd.read_csv(real_csv)
    # Number of rows equals best_k (≥2)
    best_k = int(df.loc[df["consensus_score"].idxmax(), "n_clusters"])
    assert len(df_virt) == best_k
    assert len(df_real) == best_k

    # Check plot
    plot_png = tmp_path / f"{name}_3x3_metrics.png"
    assert plot_png.exists()


def test_run_method_dbscan(tmp_path, monkeypatch, X):
    # Single-run DBSCAN pipeline
    monkeypatch.chdir(tmp_path)
    run_method(
        name="dbscan",
        clusterer_cls=DBSCANClusterer,
        X=X,
        eps=0.5,
        min_samples=5
    )

    # Check metrics CSV
    metrics_csv = tmp_path / "dbscan_metrics.csv"
    assert metrics_csv.exists()
    df = pd.read_csv(metrics_csv)
    # Single row, consensus_score present
    assert df.shape[0] == 1
    assert "consensus_score" in df.columns

    # Check centroids CSVs
    virt_csv = tmp_path / "dbscan_virtual_centroids.csv"
    real_csv = tmp_path / "dbscan_real_centroids.csv"
    assert virt_csv.exists()
    assert real_csv.exists()

    # No runtime warnings should be emitted (safe empty‐slice handling)