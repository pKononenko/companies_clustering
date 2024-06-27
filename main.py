import re
## temp
import time
## temp
import nltk
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#import spacy
#from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

import optuna
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split
#from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from doc_loader import DocumentLoader

nltk.download("stopwords")
nltk.download("wordnet")


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        )
        return text

    def preprocess_documents(self, documents: List[str]) -> List[str]:
        logger.info("Documents preprocessing..")
        with ProcessPoolExecutor() as executor:
            processed_docs = list(executor.map(self.preprocess_text, documents))
        return processed_docs


class FeatureExtractor:
    def __init__(self, max_features: int):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def extract_features(self, processed_docs: List[str]) -> np.ndarray:
        logger.info("Feature extraction...")
        return self.vectorizer.fit_transform(processed_docs).toarray()


class DimensionalityReducer:
    def __init__(self, n_components: int):
        self.pca = PCA(n_components=n_components)

    def reduce_dimensionality(self, X: np.ndarray) -> np.ndarray:
        logger.info("Dimensionality reduction...")
        return self.pca.fit_transform(X)


class DocumentClusterer:
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 1):
        self.hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

    def cluster_documents(self, X: np.ndarray) -> np.ndarray:
        logger.info("Document clustering...")
        return self.hdbscan.fit_predict(X)

    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> None:
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(X, labels)
            print(f"Silhouette Score: {silhouette_avg}")
        else:
            print("Only one cluster found.")

    def save_results(self, filenames: List[str], labels: np.ndarray, output_file: str = "clustering_results.csv") -> None:
        results = pd.DataFrame({"Filename": filenames, "Cluster": labels})
        results.to_csv(output_file, index=False)


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


def objective(trial):
    max_features = trial.suggest_int("max_features", 1000, 10000, step=1000)
    n_components = trial.suggest_int("n_components", 10, 200, step=10)
    min_cluster_size = trial.suggest_int("min_cluster_size", 10, 200, step=10)
    min_samples = trial.suggest_int("min_samples", 1, 20)

    folder_path = "companies_data"
    loader = DocumentLoader(folder_path)
    documents, filenames = loader.load_documents(num_samples=2500)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)
    logger.success("Documents successfully preprocessed.")

    # in case of errors with features extractio
    # when num of features is incorrect
    X = None
    try:
        extractor = FeatureExtractor(max_features=max_features)
        X = extractor.extract_features(processed_docs)
    except:
        logger.warning("Number of features is not matching conditions. Optimization trial skipped.")
        return -1

    reducer = DimensionalityReducer(n_components=n_components)
    X_reduced = reducer.reduce_dimensionality(X)

    clusterer = DocumentClusterer(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.cluster_documents(X_reduced)
    logger.success(f"Number of clustered labels: {len(labels)}")

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X_reduced, labels)
        return silhouette_avg
    else:
        return -1


if __name__ == "__main__":
    #logger.info("OPTIMIZING/SEARCHING BEST HYPERPARAMETERS...")
    #study = optuna.create_study(direction="maximize")
    #study.optimize(objective, n_trials=5)

    #logger.success(f"Best hyperparameters: {study.best_params}")

    #Best hyperparameters: {'max_features': 1000, 'n_components': 100, 'min_cluster_size': 70, 'min_samples': 17}
    #best_max_features = study.best_params["max_features"] 
    #best_n_components = study.best_params["n_components"] 
    #best_min_cluster_size = study.best_params["min_cluster_size"]
    #best_min_samples = study.best_params["min_samples"]

    best_max_features = 1000
    best_n_components = 100
    best_min_cluster_size = 70
    best_min_samples = 10
    
    ## temp
    #whole_process_start_time = time.time()
    ## temp
    folder_path = "companies_data"
    loader = DocumentLoader(folder_path)
    #documents, filenames = loader.load_documents(num_samples=7500)
    documents, filenames = loader.load_documents(num_samples=3500)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)

    extractor = FeatureExtractor(max_features=best_max_features)
    X = extractor.extract_features(processed_docs)
    reducer = DimensionalityReducer(n_components=best_n_components)
    X_reduced = reducer.reduce_dimensionality(X)

    clusterer = DocumentClusterer(min_cluster_size=best_min_cluster_size, min_samples=best_min_samples)
    labels = clusterer.cluster_documents(X_reduced)

    clusterer.evaluate_clustering(X_reduced, labels)
    ## temp
    #whole_process_end_time = time.time()
    #print(whole_process_end_time - whole_process_start_time)
    ## temp

    clusterer.save_results(filenames, labels)

    visualizer = DocumentClusterVisualizer()
    visualizer.visualize(X_reduced, labels)
