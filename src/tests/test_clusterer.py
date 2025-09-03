import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.clusterer import (
    KMeansClusterer,
    AgglomerativeClusterer,
    GMMClusterer,
    DBSCANClusterer,
)


@pytest.fixture
def blob_data():
    X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.5, random_state=0)
    return X


def test_kmeans_clusterer_basic(blob_data):
    X = blob_data
    km = KMeansClusterer(n_clusters=3, random_state=0)
    km.fit(X)

    # labels_ and centroids_
    assert isinstance(km.labels_, np.ndarray)
    assert isinstance(km.centroids_, np.ndarray)
    assert km.centroids_.shape == (3, X.shape[1])
    # virtual centroids should equal centroids_
    virt = km.get_virtual_centroids()
    np.testing.assert_allclose(virt, km.centroids_)
    # real centroids are actual data points
    real = km.get_real_centroids()
    assert real.shape == virt.shape
    for rc in real:
        assert any(np.all(rc == xi) for xi in X)

    m = km.get_metrics()
    # common metrics present
    for key in ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'population', 'avg_distance']:
        assert key in m
    # population sizes sum to total samples
    assert sum(m['population'].values()) == X.shape[0]


def test_agglomerative_clusterer_basic(blob_data):
    X = blob_data
    ac = AgglomerativeClusterer(n_clusters=3)
    ac.fit(X)

    # virtual centroids from cluster means
    virt = ac.get_virtual_centroids()
    assert virt.shape == (3, X.shape[1])
    real = ac.get_real_centroids()
    assert real.shape == virt.shape

    m = ac.get_metrics()
    assert 'silhouette' in m
    # population keys and values
    pops = m['population']
    assert isinstance(pops, dict)
    assert sum(pops.values()) == X.shape[0]


def test_gmm_clusterer_basic(blob_data):
    X = blob_data
    gm = GMMClusterer(n_clusters=3, random_state=0)
    gm.fit(X)

    # model and centroids_
    assert hasattr(gm, 'model')
    assert hasattr(gm, 'centroids_')
    virt = gm.get_virtual_centroids()
    np.testing.assert_allclose(virt, gm.centroids_)
    real = gm.get_real_centroids()
    assert real.shape == virt.shape

    m = gm.get_metrics()
    for key in ['silhouette', 'calinski_harabasz', 'davies_bouldin']:
        assert key in m
    # population dict
    assert isinstance(m['population'], dict)


def test_dbscan_clusterer_basic(blob_data):
    X = blob_data
    # run with default eps so some clusters, some noise
    db = DBSCANClusterer(eps=0.7, min_samples=5)
    db.fit(X)

    labels = db.labels_
    assert isinstance(labels, np.ndarray)
    m = db.get_metrics()
    # DBSCAN metrics
    for key in ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'population', 'avg_distance']:
        assert key in m

    virt = db.get_virtual_centroids()
    real = db.get_real_centroids()
    # virtual & real centroids should be consistent shapes
    assert virt.shape == real.shape
    # If no true clusters, shapes will be (0, dim)
    assert virt.ndim == 2


def test_dbscan_all_noise(blob_data):
    X = blob_data
    # eps too small → everything noise
    db = DBSCANClusterer(eps=0.01, min_samples=5)
    db.fit(X)

    # no clusters
    virt = db.get_virtual_centroids()
    real = db.get_real_centroids()
    assert virt.shape[0] == 0
    assert real.shape[0] == 0

    m = db.get_metrics()
    # silhouette must be NaN when fewer than 2 clusters
    assert np.isnan(m['silhouette'])