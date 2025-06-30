# app.py
import streamlit as st
from rag_agent import RAGAgent  # On importe votre classe d'agent

# Configuration de la page Streamlit
st.set_page_config(page_title="Assistant Économique", layout="wide")

# Titre et description
st.title("🤖 Assistant d'Analyse Économique et Financière")
st.markdown("Posez une question sur l'économie mondiale, la politique monétaire ou le commerce international. L'assistant utilise une base de connaissance de rapports récents (OCDE, FMI, BCE, etc.) et la recherche web pour répondre.")

# Initialisation de l'agent (mis en cache pour la performance)
@st.cache_resource
def load_rag_agent():
    """Charge l'agent RAG une seule fois pour toute la session."""
    try:
        agent = RAGAgent()
        return agent
    except Exception as e:
        st.error(f"Une erreur critique est survenue lors du chargement de l'agent : {e}")
        return None

agent = load_rag_agent()

if agent:
    # Initialisation de l'historique de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des messages de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Champ de saisie pour la nouvelle question
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajout et affichage du message de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Génération et affichage de la réponse de l'assistant
        with st.chat_message("assistant"):
            with st.spinner("Recherche et analyse en cours..."):
                try:
                    # Appel de la logique de l'agent
                    result = agent.query(prompt)
                    
                    # Formatage de la réponse finale
                    answer = result.get("answer", "Désolé, une erreur est survenue lors de la génération de la réponse.")
                    full_response = answer
                    source_docs = result.get('source_documents')
                    if source_docs:
                        full_response += "\n\n---\n**📚 Sources utilisées :**\n"
                        for i, doc in enumerate(source_docs):
                            source_name = doc.metadata.get('source', 'Source inconnue')
                            page = doc.metadata.get('page')
                            display_name = f"* `[{i+1}]` {source_name}"
                            if page is not None:
                                display_name += f" (Page: {int(page)})"
                            full_response += display_name + "\n"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Erreur lors de la requête : {e}")
else:
    st.warning("L'agent n'a pas pu être initialisé. L'application ne peut pas démarrer.")