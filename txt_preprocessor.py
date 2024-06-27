import re
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

import pyximport;
pyximport.install(pyimport=True)

from cython_modules.text_processing import remove_punctuation, lemmatize_text


### TODO: OPTIMIZE LIBRARY LOADING IN FUTURE ###
### TODO: RESEARCH MORE CYTHON TO SPEED UP PROCESSOR ###
### TODO: CHECK ISSUE WITH PROCESSING ORDER WITH MULTITHREADING ###
### TODO: CHECK ISSUE WITH SEVERAL IDENTICAL FILES IN CLUSTERS ###
class TextPreprocessor:
    """Documents text preprocessor class"""

    def __init__(self, num_workers: int = None):
        """
        Args:
            num_workers (int, optional): Number of ProcessPool workers.
                Defaults to None. None equals to all available workers.
        """
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.num_workers = num_workers
    
    def preprocess_text(self, text: str) -> str:
        """Preprocessing and lemmatization.

        Args:
            text (str): Document content.

        Returns:
            str: Preprocessed text/tokens.
        """
        text = text.lower()  # Lowercase text
        text = remove_punctuation(text)  # Remove punctuation
        tokens = text.split()
        lemmatized_tokens = lemmatize_text(tokens, self.stop_words, self.lemmatizer)
        return " ".join(lemmatized_tokens)

    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """Preprocessing and lemmatization of documents content.

        Args:
            documents (List[str]): Documents content list.

        Returns:
            List[str]: List of preprocessed documents contents.
        """
        logger.info("Documents preprocessing..")
        with ProcessPoolExecutor() as executor:
            processed_docs = list(executor.map(self.preprocess_text, documents))
        logger.success("Documents preprocessed.")
        return processed_docs
