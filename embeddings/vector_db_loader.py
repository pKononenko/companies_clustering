import os
import joblib
import pandas as pd
from loguru import logger
import numpy as np
from typing import List, Tuple
import faiss
from embeddings.embedding_extractor import EmbeddingExtractor
from concurrent.futures import ThreadPoolExecutor


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

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings in the Faiss index.

        Args:
            query_embedding (np.ndarray): Embeddings for search.
            top_k (int): Amount of top similar documents.

        Returns:
            List[Tuple[int, float]]: Similar documents list.
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return [
            (self.document_ids[idx], distances[0][i])
            for i, idx in enumerate(indices[0])
            if distances[0][i] > 0
        ]

    # temp
    # def parallel_search(self, query_embeddings: np.ndarray, top_k: int = 5,
    #                           num_workers: int = None) -> List[List[Tuple[str, float]]]:
    #     """Search for similar embeddings in the Faiss index in parallel."""
    #     def search_single(query_embedding):
    #         return self.search(query_embedding, top_k)
    #
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         #results = list(executor.map(lambda qe: self.search(qe, top_k), query_embeddings))
    #         #results = list(executor.map(self.search, (query_embeddings, top_k)))
    #         results = list(executor.map(search_single, query_embeddings))
    #     return results

    def update_embeddings(self, existing_filenames: List[str], processed_docs: List[str], filenames: List[str], extractor: EmbeddingExtractor) -> None:
        """Update Faiss index embeddings"""
        new_embeddings = []
        new_filenames = []

        for doc, filename in zip(processed_docs, filenames):
            if filename not in existing_filenames:
                new_embedding = extractor.extract_embeddings([doc]).reshape(1, -1)
                if not self.document_exists(new_embedding):
                    new_embeddings.append(new_embedding)
                    new_filenames.append(filename)

        if new_embeddings:
            new_embeddings = np.vstack(new_embeddings)
            self.add_embeddings(new_embeddings, new_filenames)
            self.save_index("faiss_index.bin", "document_ids.pkl")
            joblib.dump(existing_filenames + new_filenames, "document_ids.pkl")

    def remove_duplicates(self) -> None:
        """Remove duplicate embeddings from the Faiss index."""
        unique_ids, unique_indices = np.unique(self.document_ids, return_index=True)
        unique_embeddings = self.index.reconstruct_n(0, len(unique_ids))

        self.index = faiss.IndexFlatL2(unique_embeddings.shape[1])
        self.index.add(unique_embeddings)
        self.document_ids = unique_ids.tolist()

    def save_index(self, index_path: str, ids_path: str) -> None:
        """Save Faiss index and document IDs."""
        faiss.write_index(self.index, index_path)
        joblib.dump(self.document_ids, ids_path)

    def load_index(self, index_path: str, ids_path: str) -> None:
        """Load Faiss index and document IDs."""
        self.index = faiss.read_index(index_path)
        self.document_ids = joblib.load(ids_path)

    @staticmethod
    def save_csv(new_doc: str, similar_docs: List[Tuple[str, float]], csv_filename: str = 'search_results.csv') -> None:
        """Saving search results to .csv file"""
        new_data = []
        updated_data = None
        for doc_id, dist in similar_docs:
            new_data.extend([
                {
                    "Document": new_doc,
                    "Similar_Document": doc_id,
                    "Distance": dist,
                    "Similar_Document_Content": "" # UPDATE IT
                }
            ])
        new_data_df = pd.DataFrame(new_data, columns=["Document", "Similar_Document", "Distance", "Similar_Document_Content"])

        if os.path.exists(csv_filename):
            existing_data = pd.read_csv(csv_filename)
            updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)
        else:
            updated_data = new_data_df

        updated_data.to_csv(csv_filename, index=False)
        logger.success(f"Search results save in {csv_filename}.")

    def document_exists(
        self, embeddings: np.ndarray, threshold: float = 1e-7
    ) -> bool:
        """Check if a document already exists in the Faiss index."""
        query_embedding = embeddings[-1].reshape(1, -1)
        _, distances = self.index.search(query_embedding, 1)
        return (
            distances[0][0] < threshold
        )  # Threshold for considering a document as identical
