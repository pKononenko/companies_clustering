import numpy as np
from typing import List
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.feature_extraction.text import HashingVectorizer


class FeatureExtractor:
    """Textual features extractor"""

    def __init__(self, max_features: int):
        """
        Args:
            max_features (int): Max amount of features to be extracted.
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def extract_features(self, processed_docs: List[str]) -> np.ndarray:
        """Extract features via Tfidf.

        Args:
            processed_docs (List[str]): List of preprocesses docs.

        Returns:
            np.ndarray: Features array.
        """
        logger.info("Feature extraction...")
        return self.vectorizer.fit_transform(processed_docs).toarray()


# older optimized version
"""class FeatureExtractor:
    def __init__(self, max_features: int):
        self.vectorizer = HashingVectorizer(n_features=max_features, alternate_sign=False)

    def extract_features(self, processed_docs: List[str]) -> np.ndarray:
        logger.info("Feature extraction...")
        features = Parallel(n_jobs=-1)(delayed(self.vectorizer.transform)([doc]) for doc in processed_docs)
        features = np.vstack([f.toarray() for f in features])
        logger.info(f"Extracted features with shape {features.shape}.")
        return features"""
