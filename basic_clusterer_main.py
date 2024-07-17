# import optuna
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from doc_loader import DocumentLoader
from visualizer import DocumentClusterVisualizer
from txt_preprocessor import TextPreprocessor
from corpus_loader import check_nltk_data
from dimensionality_reducer import DimensionalityReducer
from feature_extractor import FeatureExtractor
from clusterer import DocumentClusterer


# NLTK corpus loading
check_nltk_data()


def objective(trial):
    max_features = trial.suggest_int("max_features", 1000, 10000, step=1000)
    n_components = trial.suggest_int("n_components", 10, 200, step=10)
    min_cluster_size = trial.suggest_int("min_cluster_size", 10, 100, step=10)
    max_cluster_size = trial.suggest_int("max_cluster_size", 0, 110, step=20)
    min_samples = trial.suggest_int("min_samples", 1, 20)
    cluster_selection_method = trial.suggest_categorical(
        "cluster_selection_method", ["eom"]
    )  # , "leaf"])
    metric = trial.suggest_categorical(
        "metric", ["euclidean"]
    )  # , 'hamming', 'jaccard', 'manhattan'])

    folder_path = "companies_data"
    loader = DocumentLoader(folder_path)
    documents, filenames = loader.load_documents(num_samples=1500)
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
        logger.warning(
            "Number of features is not matching conditions. Optimization trial skipped."
        )
        return -1

    reducer = DimensionalityReducer(n_components=n_components)
    X_reduced = reducer.reduce_dimensionality(X)

    clusterer = DocumentClusterer(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
    )
    labels = clusterer.cluster_documents(X_reduced)
    logger.success(f"Number of clustered labels: {len(labels)}")

    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X_reduced, labels)
        db_index = davies_bouldin_score(X_reduced, labels)
        ch_index = calinski_harabasz_score(X_reduced, labels)
        return (silhouette_avg + (1 / db_index) + ch_index) / 3
    else:
        return -1


if __name__ == "__main__":
    # uncomment for hyperparameters search
    # logger.info("OPTIMIZING/SEARCHING BEST HYPERPARAMETERS...")
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=8)

    # logger.success(f"Best hyperparameters: {study.best_params}")

    # Best hyperparameters: {'max_features': 4000, 'n_components': 90, 'min_cluster_size': 110, 'min_samples': 9, 'cluster_selection_method': 'leaf', 'metric': 'manhattan'}
    # best_max_features = study.best_params["max_features"]
    # best_n_components = study.best_params["n_components"]
    # best_min_cluster_size = study.best_params["min_cluster_size"]
    # best_max_cluster_size = study.best_params["min_cluster_size"]
    # best_min_samples = study.best_params["min_samples"]
    # best_cluster_selection_method = study.best_params["cluster_selection_method"]
    # best_metric = study.best_params["metric"]

    best_max_features = 3000  # 7000
    best_n_components = 120  # 100
    best_min_cluster_size = 25
    best_max_cluster_size = 0
    best_min_samples = 10
    best_cluster_selection_method = "eom"
    best_metric = "chebyshev"
    # good metrics (top)
    # cityblock -0.05
    # manhattan -0.03
    # minkowski 0.034
    # euclidean 0.02
    # chebyshev 0.13

    folder_path = "companies_data"
    loader = DocumentLoader(folder_path)
    # documents, filenames = loader.load_documents(num_samples=7500)
    documents, filenames = loader.load_documents(num_samples=500)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)

    extractor = FeatureExtractor(max_features=best_max_features)
    X = extractor.extract_features(processed_docs)

    reducer = DimensionalityReducer(n_components=best_n_components)
    X_reduced = reducer.reduce_dimensionality(X)

    clusterer = DocumentClusterer(
        min_cluster_size=best_min_cluster_size,
        max_cluster_size=best_max_cluster_size,
        min_samples=best_min_samples,
        cluster_selection_method=best_cluster_selection_method,
        metric=best_metric,
    )

    labels = clusterer.cluster_documents(X_reduced)

    clusterer.evaluate_clustering(X_reduced, labels)

    clusterer.save_results(filenames, labels)

    visualizer = DocumentClusterVisualizer()
    visualizer.visualize(X_reduced, labels)
