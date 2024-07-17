
import numpy as np
from sklearn.decomposition import IncrementalPCA
from loguru import logger


class DimensionalityReducer:
    """Data dimensions reducer class"""

    def __init__(self, n_components: int):
        """
        Args:
            n_components (int): Number of components to reduce dimensions.
        """
        self.pca = IncrementalPCA(n_components=n_components)

    def reduce_dimensionality(self, X: np.ndarray) -> np.ndarray:
        """Reduce dimensions.

        Args:
            X (np.ndarray): Textual data array.

        Returns:
            np.ndarray: Reduced data array.
        """
        logger.info("Dimensionality reduction...")
        return self.pca.fit_transform(X)
