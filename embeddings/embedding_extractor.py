import joblib
import numpy as np
from loguru import logger
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class EmbeddingExtractor:
    """Class to extract embeddings using a transformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            filepath (str): Path for embeddings.
        """
        self.model = SentenceTransformer(model_name)

    def extract_embeddings(self, documents: List[str]) -> np.ndarray:
        """Extract embeddings from text documents.

        Args:
            documents (List[str]): Preprocessed textual data array.

        Returns:
            np.ndarray: Embeddings array.
        """
        logger.info("Embeddings extraction...")
        embeddings = self.model.encode(documents, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        logger.success("Embeddings extracted successfully.")
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """Save embeddings to a file.

        Args:
            embeddings (np.ndarray): Embeddings array.
            filepath (str): Path for embeddings.
        """
        joblib.dump(embeddings, filepath)

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from a file.

        Args:
            filepath (str): Path for embeddings.

        Returns:
            np.ndarray: Loaded embeddings array.
        """
        return joblib.load(filepath)
