# run_challenging.py

import matplotlib.pyplot as plt
from src.clustering_utils import run_method
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
        name="kmeans_challenge",
        clusterer_cls=KMeansClusterer,
        X=X,
        k_range=k_range,
        random_state=123
    )
    run_method(
        name="agglo_challenge",
        clusterer_cls=AgglomerativeClusterer,
        X=X,
        k_range=k_range
    )
    run_method(
        name="gmm_challenge",
        clusterer_cls=GMMClusterer,
        X=X,
        k_range=k_range,
        random_state=123
    )
    run_method(
        name="dbscan_challenge",
        clusterer_cls=DBSCANClusterer,
        X=X,
        eps=0.8,
        min_samples=10
    )

if __name__ == "__main__":
    main()
