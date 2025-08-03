# app.py

import streamlit as st
import os
import logging
from pathlib import Path

# Import necessary classes and functions from other project files
from rag_agent import RAGAgent
from index import create_embeddings_store

# --- Page and Logging Configuration ---
st.set_page_config(
    page_title="Economic Assistant",
    page_icon="🤖",
    layout="wide"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Custom CSS for a better look and feel ---
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
</style>
""", unsafe_allow_html=True)


# --- Core Functions ---

@st.cache_resource
def initialize_agent():
    """
    Loads the RAG agent. If the vector database doesn't exist, it triggers its creation.
    Uses Streamlit's cache to load the agent only once per session, improving performance.

    Returns:
        RAGAgent | None: The initialized RAG agent instance, or None if initialization fails.
    """
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        with st.spinner("Database not found. Indexing documents... (this may take several minutes)"):
            try:
                create_embeddings_store()
                st.success("Database created and indexed successfully!")
            except Exception as e:
                st.error(f"Error during database creation: {e}")
                return None
    
    with st.spinner("Loading the Economic Assistant..."):
        try:
            agent = RAGAgent()
            return agent
        except Exception as e:
            st.error(f"Error during agent initialization: {e}")
            return None

def format_sources(source_docs: list) -> str:
    """
    Formats the list of source documents for clean display in the UI.

    Args:
        source_docs (list): A list of LangChain Document objects.

    Returns:
        str: A formatted markdown string of the sources.
    """
    if not source_docs:
        return ""
    
    source_list = []
    for i, doc in enumerate(source_docs):
        # Extract filename and page number from metadata
        source_name = Path(doc.metadata.get('source', 'Unknown Source')).name
        page = doc.metadata.get('page')
        
        display_name = f"* **[{i+1}]** {source_name}"
        if page is not None:
            # Add 1 to page number for human-readable format (pages are 0-indexed)
            display_name += f" (Page: {int(page) + 1})"
        source_list.append(display_name)
        
    return "\n\n---\n**📚 Sources Used:**\n" + "\n".join(source_list)


# --- Main Application Interface ---

st.title("🤖 Economic Analysis Assistant")
st.markdown("Query reports from the OECD, IMF, ECB, and more.")

# Initialize the agent using the cached function
agent = initialize_agent()

if agent:
    # Initialize chat history in Streamlit's session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to answer your questions on economics."}]

    # Display past messages from the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Wait for and handle user input
    if prompt := st.chat_input("Ask your question here..."):
        # Add user's message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("The agent is thinking..."):
                try:
                    # Prepare the chat history in the format expected by the agent
                    # The agent's query method expects a list of (user_msg, assistant_msg) tuples
                    chat_history_for_agent = []
                    for msg in st.session_state.messages[:-1]: # Exclude the current user question
                        if msg["role"] == "user":
                            # Find the corresponding assistant response
                            assistant_response = next((m["content"] for m in st.session_state.messages if m["role"] == "assistant" and st.session_state.messages.index(m) > st.session_state.messages.index(msg)), "")
                            if assistant_response:
                                chat_history_for_agent.append((msg["content"], assistant_response))

                    # Call the agent's query method with the prompt and formatted history
                    final_state = agent.query(prompt, chat_history_for_agent)
                    
                    answer = final_state.get("answer", "Sorry, an error occurred.")
                    sources = final_state.get('source_documents', [])
                    
                    # Format the full response including the answer and sources
                    full_response = answer + format_sources(sources)
                    
                    st.markdown(full_response)
                    
                    # Add the assistant's full response to the session state history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    logger.error(f"Error during agent call: {e}", exc_info=True)
                    st.error(f"An error occurred: {e}")

else:
    st.error("Agent initialization failed. The application cannot continue. Please check the logs.")