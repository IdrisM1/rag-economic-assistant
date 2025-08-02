# ingest.py - Version modifiée pour exporter les chunks

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
    # Expressions régulières pour nettoyer les en-têtes/pieds de page spécifiques à vos documents
    text = re.sub(r"OECD ECONOMIC OUTLOOK, VOLUME 2025 ISSUE 1 © OECD 2025", "", text, flags=re.IGNORECASE)
    text = re.sub(r"PERSPECTIVES DE L’EMPLOI DE L’OCDE 2024 © OCDE 2024", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d*RAPPORT ANNUEL 2024 DE LA BANQUE MONDIALE\d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d* Quality report on European statistics on population and migration - 2024 edition \d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d* Quality report on National and Regional Accounts 2023 data transmissions \d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d*\s*Quality report on European statistics on research and development, 2024 Edition\s*\d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ivxlc]+\s*International Monetary Fund\s*\|\s*April 2025\s*[ivxlc]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d*\s*International Monetary Fund\s*\|\s*April 2025\s*\d*", "", text, flags=re.IGNORECASE)
    return text.strip()

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

# === SECTION PRINCIPALE MODIFIÉE POUR L'EXPORTATION ===
if __name__ == "__main__":
    print("Lancement du processus d'extraction des chunks...")
    
    # 1. Exécuter le pipeline complet
    pdf_files = get_pdf_files("./data/reports")
    print(f"1. Fichiers PDF trouvés : {[str(f.name) for f in pdf_files]}")

    docs = parse_pdfs(pdf_files)
    print(f"2. {len(docs)} pages chargées.")

    docs = filter_chunks(docs)
    print(f"3. {len(docs)} pages après filtrage du bruit.")

    split_docs = chunk_documents(docs)
    print(f"4. {len(split_docs)} chunks générés.")

    output_filename = "chunks_for_review.txt"
    
    # 2. Écrire chaque chunk nettoyé dans un fichier texte
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("LISTE DE TOUS LES CHUNKS DU CORPUS\n\n")
        
        for i, doc in enumerate(split_docs):
            source = doc.metadata.get('source', 'Source inconnue')
            page = doc.metadata.get('page', 'N/A')
            cleaned_text = clean_chunk_text(doc.page_content, source)
            
            # Ajout d'un en-tête pour chaque chunk pour plus de clarté
            f.write(f"{'='*80}\n")
            f.write(f"CHUNK N°{i + 1}\n")
            f.write(f"Source: {source} (Page: {page + 1})\n") # +1 pour un numéro de page plus intuitif
            f.write(f"{'-'*80}\n\n")
            f.write(cleaned_text)
            f.write("\n\n")
            
    print(f"\n✅ Fichier '{output_filename}' généré avec succès. Il contient {len(split_docs)} chunks.")
    print("Vous pouvez maintenant me fournir le contenu de ce fichier.")