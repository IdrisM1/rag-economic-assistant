# app.py (Version Finale Autonome)
import streamlit as st
import os
import time
from rag_agent import RAGAgent
from index import create_embeddings_store # On importe votre fonction d'indexation

# --- Configuration de la page ---
st.set_page_config(page_title="Assistant Économique", layout="wide")

st.title("🤖 Assistant d'Analyse Économique et Financière")
st.markdown("Basé sur des rapports de l'OCDE, FMI, BCE, Fed, etc.")

# --- Logique de chargement et de construction de la base de données ---
DB_PATH = "./chroma_db"

# @st.cache_resource garantit que ce bloc n'est exécuté qu'une seule fois.
@st.cache_resource
def initialize_agent():
    """
    Vérifie si la base de données existe. Si non, la construit.
    Ensuite, initialise et retourne l'agent RAG.
    """
    if not os.path.exists(DB_PATH):
        st.info("Base de données non trouvée. Lancement du processus d'indexation...")
        st.warning("Cette opération peut prendre plusieurs minutes au premier démarrage.")
        
        progress_bar = st.progress(0, text="Indexation des documents...")
        
        try:
            # On appelle la fonction de votre script index.py
            create_embeddings_store()
            progress_bar.progress(100, text="✅ Base de données construite avec succès !")
            time.sleep(2) # Laisse le temps à l'utilisateur de lire le message
        except Exception as e:
            st.error(f"Erreur lors de la création de la base de données : {e}")
            return None
    
    st.info("Initialisation de l'agent RAG...")
    agent = RAGAgent()
    st.info("Agent prêt.")
    return agent

# --- Lancement de l'application ---
agent = initialize_agent()

if agent:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recherche et analyse en cours..."):
                try:
                    result = agent.query(prompt)
                    answer = result.get("answer", "Désolé, une erreur est survenue.")
                    full_response = answer
                    source_docs = result.get('source_documents')
                    if source_docs:
                        full_response += "\n\n---\n**📚 Sources :**\n"
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get('source', 'Source inconnue')
                            page = doc.metadata.get('page')
                            display_name = f"* `[{i+1}]` {source_name}"
                            if page is not None:
                                display_name += f" (Page: {int(page) + 1})" # Les pages sont indexées à 0
                            full_response += display_name + "\n"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Erreur lors de la requête : {e}")
else:
    st.error("L'initialisation de l'agent a échoué. L'application ne peut pas continuer.")