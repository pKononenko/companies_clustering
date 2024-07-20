import os
import joblib
from loguru import logger

from doc_loader import DocumentLoader
from txt_preprocessor import TextPreprocessor
from corpus_loader import check_nltk_data
from embeddings.embedding_extractor import EmbeddingExtractor
from embeddings.vector_db_loader import FaissIndex


# NLTK corpus loading
check_nltk_data()


if __name__ == "__main__":
    folder_path = "companies_data"
    loader = DocumentLoader(folder_path)
    documents, filenames = loader.load_documents(num_samples=1000)#500)
    preprocessor = TextPreprocessor()
    processed_docs = preprocessor.preprocess_documents(documents)

    extractor = EmbeddingExtractor(model_name='bert-base-nli-mean-tokens')
    #embeddings = extractor.extract_embeddings(processed_docs)
    embeddings = extractor.parallel_embedding_extraction(processed_docs, num_workers=8)

    embedding_dim = embeddings.shape[1]
    faiss_index = FaissIndex(embedding_dim)

    if os.path.exists("faiss_index.bin") and os.path.exists("document_ids.pkl"):
        faiss_index.load_index("faiss_index.bin", "document_ids.pkl")
        existing_filenames = joblib.load("document_ids.pkl")
        faiss_index.update_embeddings(existing_filenames, processed_docs, filenames, extractor)
    else:
        faiss_index.add_embeddings(embeddings, filenames)
        faiss_index.save_index("faiss_index.bin", "document_ids.pkl")
        joblib.dump(filenames, "document_ids.pkl")

    print(faiss_index.index.ntotal)

    # NO CONCURENTS: "354271 - modl.ai.txt"
    # NORMAL RESULT: "334889 - Body&Fit.txt";"323861 - 1000Farmacie.txt"
    new_doc = "323974 - Onc.AI.txt"#"359874 - Slim.AI.txt"
    new_doc_content, _ = loader._load_single_document(new_doc)
    new_doc_processed = preprocessor.preprocess_text(new_doc_content)
    new_doc_embedding = extractor.extract_embeddings([new_doc_processed])

    if not faiss_index.document_exists(new_doc_embedding):
        faiss_index.add_embeddings(new_doc_embedding, [new_doc])
        faiss_index.save_index("faiss_index.bin", "document_ids.pkl")

    similar_docs = faiss_index.search(new_doc_embedding, top_k=5)
    #similar_docs = faiss_index.parallel_search(new_doc_embedding, top_k=5)
    faiss_index.save_csv(new_doc, similar_docs)
    for doc_id, dist in similar_docs:
        print(f"Document: {doc_id}, Distance: {dist}")

    # similar_doc_ids = [doc_id for doc_id, _ in similar_docs]
    # print(filenames)
    # similar_doc_contents = [documents[filenames.index(doc_id)] for doc_id in similar_doc_ids]
    # differences = extractor.find_differences_with_bert(new_doc_content, similar_doc_contents)
    # for doc, diff in differences:
    #     print(f"Differences in document {doc}: {diff}")
