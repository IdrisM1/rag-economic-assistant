from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import get_pdf_files, parse_pdfs, filter_chunks, chunk_documents, clean_chunk_text
from pathlib import Path

def create_embeddings_store():
    pdf_files = get_pdf_files("./data/reports")
    docs = parse_pdfs(pdf_files)
    docs = filter_chunks(docs)
    split_docs = chunk_documents(docs)

    for doc in split_docs:
        filepath = doc.metadata.get('source', '')
        doc.page_content = clean_chunk_text(doc.page_content, filepath)

    # Utilise un modèle d'embeddings open source
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding_function, persist_directory="./chroma_db")
    vectordb.persist()
    print("Indexation terminée avec embeddings HuggingFace et base persistée dans ./chroma_db")

if __name__ == "__main__":
    create_embeddings_store()
