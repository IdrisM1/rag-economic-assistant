# index.py

import logging
from pathlib import Path
from ingest import (
    get_pdf_files,
    parse_pdfs,
    filter_chunks,
    chunk_documents,
    clean_chunk_text,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DATA_DIR = "./data/reports"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings_store():
    """
    Main function to orchestrate the creation of the ChromaDB vector store.
    It processes PDFs from the data directory, cleans them, chunks them,
    and stores their embeddings in the database.
    """
    logger.info("--- Starting Document Indexing Process ---")
    
    # 1. Load PDF files
    pdf_files = get_pdf_files(DATA_DIR)
    if not pdf_files:
        logger.warning("No PDF files found in the data directory. Aborting.")
        return
    logger.info(f"Found {len(pdf_files)} PDF files to process.")

    # 2. Parse PDF content
    docs = parse_pdfs(pdf_files)
    logger.info(f"Parsed {len(docs)} pages from all documents.")

    # 3. Filter out noisy or irrelevant pages
    filtered_docs = filter_chunks(docs)
    logger.info(f"Kept {len(filtered_docs)} pages after filtering.")

    # 4. Split documents into smaller chunks
    chunked_docs = chunk_documents(filtered_docs)
    logger.info(f"Split documents into {len(chunked_docs)} chunks.")

    # 5. Clean text content of each chunk
    for doc in chunked_docs:
        doc.page_content = clean_chunk_text(doc.page_content, doc.metadata.get("source", ""))
    logger.info("Cleaned text content for all chunks.")

    # 6. Initialize embedding model
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info("Initialized embedding model.")

    # 7. Create and persist the ChromaDB vector store
    logger.info("Creating and persisting the vector store... (This may take a while)")
    Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    logger.info(f"✅ --- Document Indexing Process Finished Successfully ---")
    logger.info(f"Database saved at: {DB_PATH}")


if __name__ == "__main__":
    create_embeddings_store()