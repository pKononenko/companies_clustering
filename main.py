import os
#import mmap
#from typing import List, Tuple
#from sentence_transformers import SentenceTransformer
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
#import numpy as np
#import faiss
#import joblib
#import nltk

from loguru import logger

from doc_loader import DocumentLoader
#from visualizer import DocumentClusterVisualizer
from txt_preprocessor import TextPreprocessor
from corpus_loader import check_nltk_data
from embeddings.embedding_extractor import EmbeddingExtractor
from embeddings.vector_db_loader import FaissIndex
#from dimensionality_reducer import DimensionalityReducer
#from feature_extractor import FeatureExtractor
#from clusterer import DocumentClusterer


# NLTK corpus loading
check_nltk_data()


if __name__ == "__main__":
    folder_path = "companies_data"
    loader = DocumentLoader(folder_path)
    documents, filenames = loader.load_documents(num_samples=2000)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)

    extractor = EmbeddingExtractor(model_name='bert-base-nli-mean-tokens')
    embeddings = extractor.extract_embeddings(processed_docs)

    embedding_dim = embeddings.shape[1]
    faiss_index = FaissIndex(embedding_dim)

    # Check if embeddings vector db exists
    if os.path.exists("faiss_index.bin") and os.path.exists("document_ids.pkl"):
        logger.info("Vector DB exists. Loading embeddings...")
        faiss_index.load_index("faiss_index.bin", "document_ids.pkl")
    else:
        faiss_index.add_embeddings(embeddings, filenames)
        faiss_index.save_index("faiss_index.bin", "document_ids.pkl")

    # Example of adding a new document and checking for similarity
    new_doc = "companies_data/357156 - NextML AB.txt"
    with open(new_doc, "r", encoding="utf-8") as file:
        new_doc_content = file.read()
    new_doc_processed = preprocessor.preprocess_text(new_doc_content)
    new_doc_embedding = extractor.extract_embeddings([new_doc_processed])

    if not faiss_index.document_exists(new_doc_processed, new_doc_embedding):
        faiss_index.add_embeddings(new_doc_embedding, [new_doc])
        faiss_index.save_index("faiss_index.bin", "document_ids.pkl")

    # Search for similar documents
    similar_docs = faiss_index.search(new_doc_embedding, top_k=5)
    for doc_id, dist in similar_docs:
        print(f"Document: {doc_id}, Distance: {dist}")
