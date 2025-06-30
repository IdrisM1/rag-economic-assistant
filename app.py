# app.py (Version avec indicateur de progression)

# --- DEBUT DU PATCH POUR SQLITE ---
# Correction pour forcer l'utilisation d'une version récente de sqlite3
# compatible avec ChromaDB sur des environnements comme Streamlit Cloud.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- FIN DU PATCH ---

import streamlit as st
import os
import time
from rag_agent import RAGAgent
from index import create_embeddings_store

# --- Configuration de la page ---
st.set_page_config(page_title="Assistant Économique", layout="wide")

st.title("🤖 Assistant d'Analyse Économique et Financière")
st.markdown("Basé sur des rapports de l'OCDE, FMI, BCE, Fed, etc.")

# --- Logique de chargement et de construction de la base de données ---
DB_PATH = "./chroma_db"

@st.cache_resource
def initialize_agent():
    if not os.path.exists(DB_PATH):
        with st.spinner("Base de données non trouvée. Lancement de l'indexation des documents... (cette opération peut prendre plusieurs minutes)"):
            try:
                create_embeddings_store()
            except Exception as e:
                st.error(f"Erreur lors de la création de la base de données : {e}")
                return None
    
    with st.spinner("Initialisation de l'agent RAG..."):
        agent = RAGAgent()
    return agent

agent = initialize_agent()

if agent:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Je suis prêt. Posez-moi une question sur l'économie."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- DÉBUT DE LA NOUVELLE LOGIQUE D'AFFICHAGE AVEC PROGRESSION ---
        with st.chat_message("assistant"):
            final_answer = ""
            final_sources = []
            
            # 1. On crée un conteneur st.status pour afficher la progression
            with st.status("Lancement du processus...", expanded=True) as status:
                try:
                    # 2. On boucle sur le "stream" d'événements de l'agent
                    for event in agent.stream_query(prompt):
                        # On vérifie quel nœud vient de se terminer
                        if "on_chain_end" in event['event']:
                            node_name = event['name']
                            # 3. On met à jour le message en fonction de l'étape
                            if node_name == "relevance_check":
                                status.update(label="✅ Pertinence vérifiée. Classification de la question...", state="running")
                            elif node_name == "classify_question":
                                status.update(label="✅ Classification terminée. Lancement de la recherche...", state="running")
                            elif node_name in ["search", "hybrid_search", "web_search"]:
                                status.update(label="✅ Recherche terminée. Construction du contexte...", state="running")
                            elif node_name == "context":
                                status.update(label="✅ Contexte assemblé. Génération de la réponse...", state="running")
                            elif node_name == "answer":
                                status.update(label="✅ Réponse générée. Examen par la critique...", state="running")
                            elif node_name == "critique_answer":
                                status.update(label="✅ Critique terminée.", state="running")

                        # On récupère le résultat final lorsque le graphe se termine
                        if event['event'] == 'on_graph_end':
                            final_result = event['data']['output']
                            final_answer = final_result.get("answer", "Une erreur est survenue.")
                            final_sources = final_result.get('source_documents', [])
                            status.update(label="Processus terminé !", state="complete", expanded=False)

                except Exception as e:
                    logger.error(f"Erreur lors du streaming de la requête : {e}", exc_info=True)
                    final_answer = f"Une erreur est survenue pendant le traitement : {e}"

            # 4. On affiche la réponse finale et les sources
            full_response = final_answer
            if final_sources:
                full_response += "\n\n---\n**📚 Sources utilisées :**\n"
                for i, doc in enumerate(final_sources):
                    source_name = doc.metadata.get('source', 'Source inconnue')
                    page = doc.metadata.get('page')
                    display_name = f"* `[{i+1}]` {source_name}"
                    if page is not None:
                        display_name += f" (Page: {int(page) + 1})"
                    full_response += display_name + "\n"
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        # --- FIN DE LA NOUVELLE LOGIQUE D'AFFICHAGE ---
else:
    st.error("L'initialisation de l'agent a échoué. L'application ne peut pas continuer.")