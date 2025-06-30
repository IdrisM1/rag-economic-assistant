from pathlib import Path

# Dossier où tu as mis tes rapports
data_dir = Path("./data/reports")
pdf_files = list(data_dir.glob("*.pdf"))

print(f"PDF trouvés : {[str(f) for f in pdf_files]}")
