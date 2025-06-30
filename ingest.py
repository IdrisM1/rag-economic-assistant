from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def get_pdf_files(directory):
    data_dir = Path(directory)
    return list(data_dir.glob("*.pdf"))

def parse_pdfs(pdf_files):
    docs = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs

def filter_chunks(docs):
    filtered = []
    for doc in docs:
        text = doc.page_content.strip().lower()
        if len(text) < 100:
            continue
        meta_keywords = ["isbn", "issn", "copyright", "photo credits", "note by the republic", "doi"]
        meta_hits = sum(1 for kw in meta_keywords if kw in text)
        ratio = meta_hits / max(len(text.split()), 1)
        if ratio > 0.2:
            continue
        filtered.append(doc)
    return filtered

def clean_chunk_text(text, filepath):
    filename = Path(filepath).name
    if filename == "83363382-en.pdf":
        text = re.sub(r"OECD ECONOMIC OUTLOOK, VOLUME 2025 ISSUE 1 © OECD 2025", "", text)
    elif filename == "a859bbac-fr.pdf":
        text = re.sub(r"PERSPECTIVES DE L’EMPLOI DE L’OCDE 2024 © OCDE 2024", "", text)
    elif filename == "French PDF.pdf":
        text = re.sub(r"\\d*RAPPORT ANNUEL 2024 DE LA BANQUE MONDIALE\\d*", "", text)
    elif filename == "KS-01-24-011-EN-N.pdf":
        text = re.sub(r"\\d* Quality report on European statistics on population and migration - 2024 edition \\d*", "", text)
    elif filename == "KS-01-24-023-EN-N.pdf":
        text = re.sub(r"\\d* Quality report on National and Regional Accounts 2023 data transmissions \\d*", "", text)
    elif filename == "KS-FT-24-005-EN-N.pdf":
        text = re.sub(r"\\d*\\s*Quality report on European statistics on research and development, 2024 Edition\\s*\\d*", "", text)
    elif filename == "text (1).pdf":
        text = re.sub(r"[ivxlc]+?International Monetary Fund \\| April 2025 [ivxlc]+", "", text, flags=re.IGNORECASE)
    elif filename == "text.pdf":
        text = re.sub(r"\\d*International Monetary Fund \\| April 2025\\d*", "", text)
    return text.strip()

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

def main():
    pdf_files = get_pdf_files("./data/reports")
    print(f"PDF trouvés : {[str(f) for f in pdf_files]}")

    docs = parse_pdfs(pdf_files)
    print(f"{len(docs)} documents chargés.")

    docs = filter_chunks(docs)
    print(f"{len(docs)} documents après filtrage.")

    split_docs = chunk_documents(docs)
    print(f"{len(split_docs)} chunks générés.")

    for i, doc in enumerate(split_docs[:5]):
        filepath = doc.metadata.get('source', '')
        cleaned_text = clean_chunk_text(doc.page_content, filepath)
        print("\\n" + "="*40)
        print(f"Chunk {i+1}:")
        print(cleaned_text[:500])

if __name__ == "__main__":
    main()
