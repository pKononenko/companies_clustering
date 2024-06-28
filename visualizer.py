import os
import numpy as np

# Set the MPLCONFIGDIR environment variable
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
# Disable Cython extensions for Matplotlib
os.environ['USE_CYTHON'] = 'False'

import matplotlib.pyplot as plt


class DocumentClusterVisualizer:
    @staticmethod
    def visualize(X: np.ndarray, labels: np.ndarray) -> None:
        plt.figure(figsize=(10, 8))
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = "k"  # Black color for noise

            class_member_mask = labels == k
            xy = X[class_member_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=col,
                markeredgecolor="k",
                markersize=6,
            )

        plt.title("HDBSCAN Clustering Results")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()