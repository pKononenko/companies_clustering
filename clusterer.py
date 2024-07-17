import numpy as np
import pandas as pd
from typing import List
from loguru import logger
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class DocumentClusterer:
    def __init__(self, min_cluster_size: int = 5, max_cluster_size: int = 10, min_samples: int = 1, cluster_selection_method: str = "eom", metric: str = "euclidean"):
        self.hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, min_samples=min_samples, cluster_selection_method=cluster_selection_method, metric=metric, p=4)

    def cluster_documents(self, X: np.ndarray) -> np.ndarray:
        logger.info("Document clustering...")
        return self.hdbscan.fit_predict(X)

    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> None:
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(X, labels)
            db_index = davies_bouldin_score(X, labels)
            ch_index = calinski_harabasz_score(X, labels)
            avg_score = (silhouette_avg + (1 / db_index) + ch_index) / 3
            logger.success(f"Silhouette Score: {silhouette_avg}")
            logger.success(f"Score: {avg_score}")
        else:
            logger.info("Only one cluster found.")

    def save_results(self, filenames: List[str], labels: np.ndarray, output_file: str = "clustering_results.csv") -> None:
        results = pd.DataFrame({"Filename": filenames, "Cluster": labels})
        results.to_csv(output_file, index=False)
