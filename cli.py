      ########
   ##############
### NOT DONE YET ###
   ##############
      ########
import os
import click
import joblib
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
@click.option("-d", "--new_doc", type=click.STRING, default="")
@click.option("-k", "--top_k", type=click.INT, default=5)
def cli(option, num_docs, new_doc, top_k):
    folder_path = "companies_data"

    loader = DocumentLoader(folder_path)
    preprocessor = TextPreprocessor()
    extractor = EmbeddingExtractor(model_name='bert-base-nli-mean-tokens')

    embedding_dim = 768 # for bert base
    faiss_index = FaissIndex(embedding_dim)

    if option == 1:
        documents, filenames = loader.load_documents(num_samples=num_docs)
        processed_docs = preprocessor.preprocess_documents(documents)
        embeddings = extractor.extract_embeddings(processed_docs)
        #embeddings = extractor.parallel_embedding_extraction(processed_docs, num_workers=None)

        # Check if embeddings vector db exists
        if os.path.exists("faiss_index.bin") and os.path.exists("document_ids.pkl"):
            faiss_index.load_index("faiss_index.bin", "document_ids.pkl")
            existing_filenames = joblib.load("document_ids.pkl")
            print(existing_filenames)
            faiss_index.update_embeddings(existing_filenames, processed_docs, filenames, extractor)
        else:
            faiss_index.add_embeddings(embeddings, filenames)
            faiss_index.save_index("faiss_index.bin", "document_ids.pkl")
            joblib.dump(filenames, "document_ids.pkl")

        faiss_index.remove_duplicates()

        print(faiss_index.index.ntotal)

    elif option == 2:
        if not os.path.exists("faiss_index.bin") or not os.path.exists("document_ids.pkl"):
            logger.error("Faiss index DB not found, create it with commands: python cli.py -o 1 --num_docs 1")

        else:
            faiss_index.load_index("faiss_index.bin", "document_ids.pkl")

            new_doc_content, _ = loader._load_single_document(new_doc)
            new_doc_processed = preprocessor.preprocess_text(new_doc_content)
            new_doc_embedding = extractor.extract_embeddings([new_doc_processed])

            if not faiss_index.document_exists(new_doc_embedding):
                faiss_index.add_embeddings(new_doc_embedding, [new_doc])
                faiss_index.save_index("faiss_index.bin", "document_ids.pkl")

            similar_docs = faiss_index.search(new_doc_embedding, top_k=top_k)
            #similar_docs = faiss_index.parallel_search(new_doc_embedding, top_k=top_k)
            faiss_index.save_csv(new_doc, similar_docs)
            for doc_id, dist in similar_docs:
                print(f"Document: {doc_id}, Distance: {dist}")


if __name__ == "__main__":
    cli()
