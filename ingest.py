# ingest.py

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from typing import List
from langchain.schema import Document

# --- Document Loading and Parsing Functions ---

def get_pdf_files(directory: str) -> List[Path]:
    """
    Finds all PDF files in a given directory.

    Args:
        directory (str): The path to the directory to search.

    Returns:
        List[Path]: A list of Path objects for each found PDF.
    """
    data_dir = Path(directory)
    return list(data_dir.glob("*.pdf"))

def parse_pdfs(pdf_files: List[Path]) -> List[Document]:
    """
    Loads and parses content from a list of PDF files.

    Args:
        pdf_files (List[Path]): A list of paths to the PDF files.

    Returns:
        List[Document]: A list of LangChain Document objects, one for each page.
    """
    docs = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs

# --- Text Cleaning and Chunking Functions ---

def filter_chunks(docs: List[Document]) -> List[Document]:
    """
    Filters out documents (pages) that are likely irrelevant metadata or too short.

    Args:
        docs (List[Document]): A list of documents to filter.

    Returns:
        List[Document]: The filtered list of documents.
    """
    filtered = []
    for doc in docs:
        text = doc.page_content.strip().lower()
        if len(text) < 100:  # Filter out very short pages
            continue
        
        # Heuristics to detect metadata pages
        meta_keywords = ["isbn", "issn", "copyright", "photo credits", "doi"]
        meta_hits = sum(1 for kw in meta_keywords if kw in text)
        ratio = meta_hits / max(len(text.split()), 1)
        if ratio > 0.2:  # If more than 20% of words are metadata keywords
            continue
            
        filtered.append(doc)
    return filtered

def clean_chunk_text(text: str, filepath: str) -> str:
    """
    Cleans the text of a chunk by removing known headers/footers based on the source file.

    Args:
        text (str): The text content of the chunk.
        filepath (str): The path of the source file, used to apply specific cleaning rules.

    Returns:
        str: The cleaned text.
    """
    # This function can be expanded with more regex rules for different report formats
    # Example rule:
    text = re.sub(r"OECD ECONOMIC OUTLOOK, VOLUME 2025 ISSUE 1 © OECD 2025", "", text, flags=re.IGNORECASE)
    # Add other rules here...
    return text.strip()

def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits a list of large documents into smaller, overlapping chunks.

    Args:
        docs (List[Document]): The documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of the new, smaller chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

# --- Main execution block for testing or direct export ---
if __name__ == "__main__":
    """
    This block runs when the script is executed directly.
    It performs a full processing pipeline and saves all chunks to a text file for review.
    """
    print("--- Starting Chunk Extraction Process ---")
    
    pdf_files = get_pdf_files("./data/reports")
    print(f"1. Found PDF files: {[str(f.name) for f in pdf_files]}")

    docs = parse_pdfs(pdf_files)
    print(f"2. Loaded {len(docs)} pages.")

    docs = filter_chunks(docs)
    print(f"3. {len(docs)} pages remaining after filtering.")

    split_docs = chunk_documents(docs)
    print(f"4. Generated {len(split_docs)} chunks.")

    output_filename = "chunks_for_review.txt"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("LIST OF ALL CHUNKS IN THE CORPUS\n\n")
        
        for i, doc in enumerate(split_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            page = doc.metadata.get('page', 'N/A')
            cleaned_text = clean_chunk_text(doc.page_content, source)
            
            f.write(f"{'='*80}\n")
            f.write(f"CHUNK #{i + 1}\n")
            f.write(f"Source: {source} (Page: {page + 1})\n")
            f.write(f"{'-'*80}\n\n")
            f.write(cleaned_text)
            f.write("\n\n")
            
    print(f"\n✅ Successfully generated '{output_filename}' with {len(split_docs)} chunks.")