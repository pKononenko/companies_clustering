import os
import mmap
from typing import List, Tuple

from concurrent.futures import ThreadPoolExecutor

import random
from loguru import logger


class DocumentLoader:
    """Folder data loader class"""

    def __init__(self, folder_path: str, num_workers: int = None):
        """
        Args:
            folder_path (str): Path to folder with files.
            num_workers (int, optional): Number of ThreadPool workers.
                Defaults to None. None equals to all available workers.
        """
        self.folder_path = folder_path
        self.num_workers = num_workers

    def _load_single_document(self, filename: str) -> Tuple[str, str]:
        """Memory optimized large text document loading.

        Args:
            filename (str): Textfile name.

        Returns:
            Tuple[str, str]: Filename and its content.
        """
        filepath = os.path.join(self.folder_path, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            # Memory-map the file, size 0 means whole file
            with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                content = mm.read().decode("utf-8")
        return content, filename

    def load_documents(self, num_samples: int = None) -> Tuple[List[str], List[str]]:
        """Load documents from folder.

        Args:
            num_samples (int, optional): Number of files to extract.
                Defaults to None.

        Returns:
            Tuple[List[str], List[str]]: Documents contents and their filenames.
        """
        logger.info(f"Loading Documents from '{self.folder_path}' folder...")
        filenames = [f for f in os.listdir(self.folder_path) if f.endswith(".txt")]
        if num_samples:
            filenames = random.choices(filenames, k=num_samples)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_single_document, filenames))

        documents, filenames = zip(*results)
        logger.success("Documents successfully loaded.")
        return list(documents), list(filenames)
