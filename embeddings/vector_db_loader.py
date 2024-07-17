import joblib
import numpy as np
from typing import List, Tuple
import faiss


class FaissIndex:
    """Class for managing Faiss index."""

    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim (int): Embedding dimension.
        """
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.document_ids = []

    def add_embeddings(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Add embeddings to the Faiss index.

        Args:
            embeddings (np.ndarray): Embeddings multi-array.
            ids (List[str]):
        """
        self.index.add(embeddings)
        self.document_ids.extend(ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar embeddings in the Faiss index.

        Args:
            query_embedding (np.ndarray): Embeddings for search.
            top_k (int): Amount of top similar documents.

        Returns:
            List[Tuple[int, float]]: Simmilar documents list.
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return [(self.document_ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    def save_index(self, index_path: str, ids_path: str) -> None:
        """Save Faiss index and document IDs."""
        faiss.write_index(self.index, index_path)
        joblib.dump(self.document_ids, ids_path)

    def load_index(self, index_path: str, ids_path: str) -> None:
        """Load Faiss index and document IDs."""
        self.index = faiss.read_index(index_path)
        self.document_ids = joblib.load(ids_path)

    def document_exists(self, document: str, embeddings: np.ndarray, threshold: float = 1e-5) -> bool:
        """Check if a document already exists in the Faiss index."""
        query_embedding = embeddings[-1].reshape(1, -1)
        _, distances = self.index.search(query_embedding, 1)
        return distances[0][0] < threshold  # Threshold for considering a document as identical
