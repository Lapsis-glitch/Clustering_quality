# test_selection.py

import numpy as np
import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import make_blobs

from src.selection import plot_elbow, plot_dendrogram


def test_plot_elbow_outputs_figure(tmp_path):
    X, _ = make_blobs(n_samples=50, centers=3, random_state=0)
    fig = plot_elbow(X, k_range=range(1, 6), random_state=0)
    # Should return a Matplotlib Figure or Axes object
    assert hasattr(fig, 'savefig') or hasattr(fig, 'figure')
    # Check that it has at least one line plotted
    axes = fig.axes if hasattr(fig, 'axes') else [fig]
    lines = sum(len(ax.get_lines()) for ax in axes)
    assert lines >= 1
    plt.close(fig)


def test_plot_dendrogram_outputs_figure(tmp_path):
    X, _ = make_blobs(n_samples=30, centers=2, random_state=1)
    fig = plot_dendrogram(X, method='ward')
    assert hasattr(fig, 'savefig') or hasattr(fig, 'axes')
    # The dendrogram plotting typically adds lines/collections
    axes = fig.axes
    # Should have at least one LineCollection or PolyCollection
    has_dendro = any(
        any(isinstance(c, (plt.Line2D, plt.PolyCollection)) for c in ax.get_children())
        for ax in axes
    )
    assert has_dendro
    plt.close(fig)