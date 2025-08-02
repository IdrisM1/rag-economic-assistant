# app.py

import streamlit as st
import os
import logging
from pathlib import Path

# CORRIGÉ : Importation depuis le bon fichier
from rag_agent_2 import RAGAgent
from index import create_embeddings_store

# --- Configuration de la page et du logging ---
st.set_page_config(
    page_title="Assistant Économique",
    page_icon="🤖",
    layout="wide"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Styles CSS personnalisés pour une meilleure apparence ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #e1f5fe;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: #f1f8e9;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# --- Fonctions principales ---

@st.cache_resource
def initialize_agent():
    """
    Charge l'agent RAG. Si la base de données n'existe pas, elle la crée.
    Utilise le cache de Streamlit pour ne charger l'agent qu'une seule fois.
    """
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        with st.spinner("Base de données non trouvée. Lancement de l'indexation des documents... (cette opération peut prendre plusieurs minutes)"):
            try:
                create_embeddings_store()
                st.success("Base de données créée et indexée avec succès !")
            except Exception as e:
                st.error(f"Erreur lors de la création de la base de données : {e}")
                return None
    
    with st.spinner("Chargement de l'assistant économique..."):
        try:
            agent = RAGAgent()
            return agent
        except Exception as e:
            st.error(f"Erreur lors de l'initialisation de l'agent : {e}")
            return None

def format_sources(source_docs: list) -> str:
    """Met en forme la liste des documents sources pour l'affichage."""
    if not source_docs:
        return ""
    
    source_list = []
    for i, doc in enumerate(source_docs):
        source_name = Path(doc.metadata.get('source', 'Source inconnue')).name
        page = doc.metadata.get('page')
        display_name = f"* **[{i+1}]** {source_name}"
        if page is not None:
            display_name += f" (Page: {int(page) + 1})"
        source_list.append(display_name)
        
    return "\n\n---\n**📚 Sources utilisées :**\n" + "\n".join(source_list)


# --- Interface Principale de l'Application ---

st.title("🤖 Assistant d'Analyse Économique")
st.markdown("Interrogez des rapports de l'OCDE, du FMI, de la BCE et plus encore.")

# Initialisation de l'agent
agent = initialize_agent()

if agent:
    # Initialisation de l'état de la session pour la conversation
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Bonjour ! Je suis prêt à répondre à vos questions sur l'économie."}]

    # Affichage des messages de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Champ de saisie pour l'utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajout du message de l'utilisateur à l'historique et affichage
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Génération et affichage de la réponse de l'assistant
        with st.chat_message("assistant"):
            with st.spinner("L'agent réfléchit..."):
                try:
                    # Préparation de l'historique pour l'agent (format attendu)
                    chat_history_for_agent = []
                    # On ne prend que les 10 derniers messages pour ne pas surcharger le contexte
                    recent_messages = st.session_state.messages[-11:-1] 

                    for i in range(0, len(recent_messages), 2):
                        if i+1 < len(recent_messages) and recent_messages[i]['role'] == 'user' and recent_messages[i+1]['role'] == 'assistant':
                            user_msg = recent_messages[i]['content']
                            assistant_msg = recent_messages[i+1]['content']
                            chat_history_for_agent.append((user_msg, assistant_msg))

                    # Appel de l'agent avec la question et l'historique
                    final_state = agent.query(prompt, chat_history_for_agent)
                    
                    answer = final_state.get("answer", "Désolé, une erreur est survenue.")
                    sources = final_state.get('source_documents', [])
                    
                    # Formatage de la réponse complète
                    full_response = answer + format_sources(sources)
                    
                    st.markdown(full_response)
                    
                    # Ajout de la réponse complète de l'assistant à l'historique
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    logger.error(f"Erreur lors de l'appel à l'agent : {e}", exc_info=True)
                    st.error(f"Une erreur est survenue : {e}")

else:
    st.error("L'initialisation de l'agent a échoué. L'application ne peut pas continuer. Veuillez vérifier les logs.")