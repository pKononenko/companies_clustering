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
        embeddings = self.model.encode(
            documents, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        logger.success("Embeddings extracted successfully.")
        return embeddings

    def extract_sentence_embeddings(
        self, document: str
    ) -> Tuple[List[str], np.ndarray]:
        """Extract embeddings from sentences in a single text document.

        Args:
            document (str): Document text.

        Returns:
            Tuple[List[str], np.ndarray]: Sentences and their embeddings.
        """
        sentences = document.split(".")
        sentence_embeddings = self.model.encode(
            sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return sentences, sentence_embeddings

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

    def find_differences_with_bert(
        self, base_doc: str, other_docs: List[str], threshold: float = 0.3
    ) -> List[Tuple[str, List[str]]]:
        """Find differences between the base document and other documents using BERT embeddings.

        Args:
            base_doc (str): The base document content.
            other_docs (List[str]): List of other document contents to compare against the base document.

        Returns:
            List[Tuple[str, List[str]]]: A list of tuples containing the list of differing sentences.
        """
        base_sentences, base_embeddings = self.extract_sentence_embeddings(base_doc)
        differences = []

        for other_doc in other_docs:
            other_sentences, other_embeddings = self.extract_sentence_embeddings(
                other_doc
            )
            diff_sentences = []

            for i, other_embedding in enumerate(other_embeddings):
                distances = np.linalg.norm(base_embeddings - other_embedding, axis=1)
                min_distance_idx = np.argmin(distances)
                if (
                    distances[min_distance_idx] > threshold
                ):  # Adjust the threshold as needed
                    diff_sentences.append(other_sentences[i])

            differences.append((other_doc, diff_sentences))

        return differences
