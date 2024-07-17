      ########
   ##############
### NOT DONE YET ###
   ##############
      ########
import os
import click
from loguru import logger


from doc_loader import DocumentLoader
from txt_preprocessor import TextPreprocessor
from corpus_loader import check_nltk_data
from embeddings.embedding_extractor import EmbeddingExtractor
from embeddings.vector_db_loader import FaissIndex

# NLTK corpus loading
check_nltk_data()


@click.command()
@click.option("-o", "--option", type=click.INT, default=1)
@click.option("-n", "--num_docs", type=click.INT, default=200)
def cli(option, num_docs):
    folder_path = "companies_data"

    if option == 1:
        loader = DocumentLoader(folder_path)
        documents, filenames = loader.load_documents(num_samples=200)  # 2000)
        preprocessor = TextPreprocessor()
        processed_docs = preprocessor.preprocess_documents(documents)

        extractor = EmbeddingExtractor(model_name='bert-base-nli-mean-tokens')
        embeddings = extractor.extract_embeddings(processed_docs)

        embedding_dim = embeddings.shape[1]

        # Check if embeddings vector db exists
        if os.path.exists("faiss_index.bin") and os.path.exists("document_ids.pkl"):
            logger.info("Vector DB exists. Loading embeddings...")
            faiss_index.load_index("faiss_index.bin", "document_ids.pkl")
        else:
            faiss_index.add_embeddings(embeddings, filenames)
            faiss_index.save_index("faiss_index.bin", "document_ids.pkl")

    elif option == 2:

    new_doc = "companies_data/357156 - NextML AB.txt"
    with open(new_doc, "r", encoding="utf-8") as file:
        new_doc_content = file.read()
    new_doc_processed = preprocessor.preprocess_text(new_doc_content)
    new_doc_embedding = extractor.extract_embeddings([new_doc_processed])

    if not faiss_index.document_exists(new_doc_processed, new_doc_embedding):
        faiss_index.add_embeddings(new_doc_embedding, [new_doc])
        faiss_index.save_index("faiss_index.bin", "document_ids.pkl")

    similar_docs = faiss_index.search(new_doc_embedding, top_k=5)
    for doc_id, dist in similar_docs:
        print(f"Document: {doc_id}, Distance: {dist}")

    similar_doc_ids = [doc_id for doc_id, _ in similar_docs]
    similar_doc_contents = [documents[filenames.index(doc_id)] for doc_id in similar_doc_ids]
    differences = extractor.find_differences_with_bert(new_doc_content, similar_doc_contents)
    for doc, diff in differences:
        print(f"Differences in document {doc}: {diff}")

if __name__ == "__main__":
    cli()