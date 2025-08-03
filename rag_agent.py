# rag_agent.py

import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple

# Load environment variables from a .env file
load_dotenv()

# LangChain and related library imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
import logging

# Local imports
from tavily_agent import TavilySearchAgent
from sentence_transformers.cross_encoder import CrossEncoder

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Configuration Class ---
class RAGConfig:
    """Centralized configuration for the RAG agent."""
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_DB_PATH = "./chroma_db"
    
    LLM_GENERATOR_MODEL = "llama3:8b"
    LLM_CLASSIFIER_MODEL = "gemma:2b"
    
    # Re-Ranking Parameters
    INITIAL_SEARCH_K = 15  # Retrieve more documents initially for the re-ranker
    RERANK_MODEL = "cross-encoder/ms-marco-minilm-l-6-v2" # Lightweight and effective model
    RERANK_TOP_K = 4       # Keep the top 4 documents after re-ranking
    
    MAX_CONTEXT_LENGTH = 16000

# --- Agent State Schema ---
class AgentState(BaseModel):
    """
    Defines the state that is passed between nodes in the graph.
    It holds all the data required for the agent's execution flow.
    """
    query: str = ""
    documents: List[Document] = []
    # NEW: Store for the conversation history
    chat_history: List[Tuple[str, str]] = [] # List of (human, ai) tuples
    
    tavily_results: List[Dict] = []
    context: str = ""
    answer: str = ""
    source_documents: List[Document] = []
    strategy_log: List[str] = []
    critique: str = ""
    relevance: str = ""
    local_search_sufficiency: str = ""


# --- Main Agent Class ---
class RAGAgent:
    """
    A conversational RAG agent that uses a graph-based approach (LangGraph)
    to answer questions based on a local document store and web search.
    """
    def __init__(self, config: RAGConfig = None):
        """Initializes the agent, its components, and builds the graph."""
        self.config = config or RAGConfig()
        self._initialize_components()
        if "TAVILY_API_KEY" not in os.environ:
            raise ValueError("❌ Tavily API key (TAVILY_API_KEY) not found in .env file.")
        self.tavily_agent = TavilySearchAgent(api_key=os.environ["TAVILY_API_KEY"])
        self.graph = self._build_graph()

    def _initialize_components(self):
        """Initializes all major components like LLMs, vector DB, prompts, and the re-ranker."""
        logger.info("Initializing components...")
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
        self.vectordb = Chroma(persist_directory=self.config.CHROMA_DB_PATH, embedding_function=self.embedding_function)
        
        self.llm_generator = OllamaLLM(model=self.config.LLM_GENERATOR_MODEL, temperature=0.1)
        self.llm_classifier = OllamaLLM(model=self.config.LLM_CLASSIFIER_MODEL, temperature=0.0)

        logger.info(f"Loading re-ranking model: {self.config.RERANK_MODEL}")
        self.reranker = CrossEncoder(self.config.RERANK_MODEL)
        
        self._initialize_prompts()
        logger.info("✅ Components initialized successfully")

    def _initialize_prompts(self):
        """
        Defines all PromptTemplates used by the agent for clarity and maintainability.
        Note: The prompt text is in French as it's the target language for the LLM interaction.
        """
        self.condense_question_prompt = PromptTemplate(
            template="""Étant donné un historique de conversation et une question de suivi, reformule la question de suivi pour en faire une question autonome.

**Règles strictes :**
1. La question autonome DOIT être compréhensible sans l'historique.
2. Si la question de suivi est déjà autonome (par exemple, "Quelles sont les prévisions économiques pour l'Allemagne ?"), renvoie-la telle quelle, sans modification.
3. NE RÉPONDS JAMAIS à la question, contente-toi de la reformuler.
4. Formule la question en français.

**Exemples :**
---
Historique:
Humain: Quelles sont les prévisions d'inflation pour la France ?
IA: L'inflation devrait baisser à 2% en 2025.
Question de suivi: Et pour l'Allemagne ?
Question Autonome: Quelles sont les prévisions d'inflation pour l'Allemagne ?
---
Historique:
Humain: Quelles sont les prévisions d'inflation pour la France ?
IA: L'inflation devrait baisser à 2% en 2025.
Question de suivi: Quels sont les principaux risques ?
Question Autonome: Quels sont les principaux risques qui pèsent sur les prévisions d'inflation en France ?
---

**À toi de jouer :**
Historique de la conversation:
{chat_history}
Question de suivi: {query}
Question Autonome:""",
            input_variables=["chat_history", "query"],
        )
        
        self.relevance_prompt = PromptTemplate(
            template="""Analyse la question suivante. Ton unique tâche est de répondre en un seul mot, en français. Réponds UNIQUEMENT par "pertinente" si la question concerne l'économie, la finance ou les statistiques, ou par "non_pertinente" dans tous les autres cas.
Question: "{query}"
Réponse (un seul mot):""",
            input_variables=["query"],
        )
        
        self.answer_prompt = PromptTemplate(
            template="""Tu es un analyste économique expert. Ton rôle est de synthétiser les informations du contexte fourni pour répondre à la question de l'utilisateur.

Contexte Fourni :
---
{context}
---

**Instructions IMPÉRATIVES :**
1. **Adéquation stricte :** Base ta réponse UNIQUEMENT sur les informations contenues dans le contexte fourni. Ne fais aucune supposition et n'ajoute pas d'informations externes.
2. **Honnêteté directe :** Si le contexte ne contient pas de réponse directe à la question, commence ta réponse par "D'après les documents fournis, il n'y a pas d'information précise concernant...".
3. **Pertinence avant tout :** Si tu ne trouves pas de réponse directe, ne synthétise des informations connexes que si elles apportent un éclairage DIRECTEMENT pertinent à la question. **Ne tente JAMAIS de créer des liens entre des sujets non pertinents.** Si les informations connexes sont trop éloignées, dis simplement que tu n'as pas la réponse.
4. **Structure et clarté :** Cite tes sources après chaque information avec le format `[1]`, `[2]`, etc.
5. **Conclusion concise :** Termine par une brève synthèse de 1 à 2 phrases. Ne répète pas les difficultés ou le manque d'information si tu l'as déjà mentionné au début.

Question : {query}
Réponse d'expert :""",
            input_variables=["context", "query"],
        )

        self.critique_prompt = PromptTemplate(
            template="""Analyse la réponse fournie. Si elle commence par "D'après les documents fournis, il n'y a pas d'information précise", réponds UNIQUEMENT par "insatisfaisante". Sinon, réponds UNIQUEMENT par "satisfaisante".
Réponse à évaluer: "{answer}"
Critique (satisfaisante/insatisfaisante):""",
            input_variables=["answer"],
        )

    # --- Graph Nodes ---
    
    def _condense_question_node(self, state: AgentState) -> Dict[str, Any]:
        """First node: takes chat history and a new question to create a standalone question."""
        logger.info("🧠 Condensing question with history...")
        if not state.chat_history:
            logger.info("No history, question remains unchanged.")
            return {"query": state.query}

        history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in state.chat_history])
        chain = self.condense_question_prompt | self.llm_classifier
        new_query = chain.invoke({"chat_history": history_str, "query": state.query})
        logger.info(f"New standalone question: {new_query}")
        return {"query": new_query}

    def _relevance_check_node(self, state: AgentState) -> Dict[str, Any]:
        """Checks if the (now standalone) question is relevant to the agent's domain."""
        logger.info("🛡️  Checking question relevance...")
        chain = self.relevance_prompt | self.llm_classifier
        relevance_result = chain.invoke({"query": state.query}).strip().lower()
        logger.info(f"🔎 Relevance check result: {relevance_result}")
        return {"relevance": "off_topic" if "non_pertinente" in relevance_result else "on_topic"}

    def _off_topic_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates a canned response for off-topic questions."""
        logger.warning("❌ Question deemed off-topic. Ending process.")
        off_topic_message = "As an assistant specializing in economics and finance, I unfortunately cannot answer questions on other topics."
        return {"answer": off_topic_message, "source_documents": []}

    def _search_node(self, state: AgentState) -> Dict[str, Any]:
        """Performs a broad vector search on the local ChromaDB to retrieve candidate documents."""
        logger.info(f"🔍 Performing broad local search (k={self.config.INITIAL_SEARCH_K})...")
        retriever = self.vectordb.as_retriever(search_kwargs={"k": self.config.INITIAL_SEARCH_K})
        documents = retriever.invoke(state.query)
        return {"documents": documents, "tavily_results": [], "strategy_log": state.strategy_log + ["local_search"]}

    def _rerank_node(self, state: AgentState) -> Dict[str, Any]:
        """Re-ranks the retrieved documents using a Cross-Encoder model for higher relevance."""
        logger.info(f"🔁 Re-ranking {len(state.documents)} retrieved documents...")
        if not state.documents:
            return {"documents": []}

        pairs = [(state.query, doc.page_content) for doc in state.documents]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(scores, state.documents), key=lambda x: x[0], reverse=True)
        
        top_docs = [doc for score, doc in scored_docs[:self.config.RERANK_TOP_K]]
        logger.info(f"✅ Re-ranking complete. {len(top_docs)} documents kept.")
        return {"documents": top_docs}

    def _web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Performs a web search using Tavily for up-to-date or missing information."""
        logger.info("🕸️ Switching to web search strategy...")
        tavily_results = self.tavily_agent.search(state.query)
        return {"documents": state.documents, "tavily_results": tavily_results, "strategy_log": state.strategy_log + ["web_search"]}

    def _context_node(self, state: AgentState) -> Dict[str, Any]:
        """Builds the final context string from all retrieved and re-ranked documents."""
        logger.info("🏗️  Building context and sources...")
        source_documents = list(state.documents)
        if state.tavily_results:
            tavily_docs = [Document(page_content=res.get("content", ""), metadata={"source": res.get("url", "N/A")}) for res in state.tavily_results]
            source_documents.extend(tavily_docs)
            
        if not source_documents:
            return {"context": "No relevant context found.", "source_documents": []}
            
        # De-duplicate documents based on page content
        unique_contents = set()
        unique_documents = []
        for doc in source_documents:
            if doc.page_content not in unique_contents:
                unique_documents.append(doc)
                unique_contents.add(doc.page_content)
        
        context_parts = [f"Source [{i+1}] (from: {doc.metadata.get('source', 'unknown')}):\n{doc.page_content}" for i, doc in enumerate(unique_documents)]
        full_context = "\n\n---\n\n".join(context_parts)
        
        if len(full_context) > self.config.MAX_CONTEXT_LENGTH:
            full_context = full_context[:self.config.MAX_CONTEXT_LENGTH]
            logger.warning("Context was truncated to max length.")
            
        return {"context": full_context, "source_documents": unique_documents}

    def _answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Generates the final answer using the generator LLM and the prepared context."""
        logger.info("🤖 Generating final answer...")
        chain = self.answer_prompt | self.llm_generator
        answer = chain.invoke({"context": state.context, "query": state.query})
        return {"answer": answer}

    def _critique_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Critiques the generated answer to decide if a fallback (like web search) is needed."""
        logger.info("🤔 Critiquing the generated answer...")
        chain = self.critique_prompt | self.llm_classifier
        critique_result = chain.invoke({"answer": state.answer}).strip().lower()
        logger.info(f"🔎 Critique result: {critique_result}")
        return {"critique": "rerun" if "insatisfaisante" in critique_result else "end"}

    def _should_do_web_search(self, state: AgentState) -> str:
        """Determines if a web search is necessary after a failed local search."""
        logger.info("⚖️ Deciding whether to perform a web search...")
        if "web_search" in state.strategy_log:
            logger.info("Web search already performed. Ending.")
            return "end"
        
        logger.info("Critique was unsatisfactory. Attempting web search.")
        return "continue_to_web_search"

    # --- Graph Construction ---

    def _build_graph(self) -> CompiledStateGraph:
        """Builds the complete LangGraph workflow, connecting all nodes and defining the logic."""
        workflow = StateGraph(AgentState)
        
        # Add all nodes to the graph
        workflow.add_node("condense_question", self._condense_question_node)
        workflow.add_node("relevance_check", self._relevance_check_node)
        workflow.add_node("off_topic_answer", self._off_topic_answer_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("critique_answer", self._critique_answer_node)
        workflow.add_node("web_search", self._web_search_node)

        # Define the graph's edges and conditional logic
        workflow.set_entry_point("condense_question")
        workflow.add_edge("condense_question", "relevance_check")
        
        workflow.add_conditional_edges(
            "relevance_check", 
            lambda s: s.relevance, 
            {"on_topic": "search", "off_topic": "off_topic_answer"}
        )
        workflow.add_edge("off_topic_answer", END)
        
        workflow.add_edge("search", "rerank")
        workflow.add_edge("rerank", "context")
        workflow.add_edge("context", "answer")
        workflow.add_edge("answer", "critique_answer")

        workflow.add_conditional_edges(
            "critique_answer",
            self._should_do_web_search,
            {
                "continue_to_web_search": "web_search",
                "end": END,
            },
        )
        
        # After a web search, the new documents are added to the context and we try to answer again
        workflow.add_edge("web_search", "context")
        
        logger.info("✅ RAG graph compiled with conversational memory and re-ranking.")
        return workflow.compile()

    # --- Public Method ---

    def query(self, question: str, chat_history: List[Tuple[str, str]]) -> Dict:
        """
        Executes a query through the graph.

        Args:
            question (str): The user's new question.
            chat_history (List[Tuple[str, str]]): The history of the conversation.

        Returns:
            Dict: The final state of the graph after execution.
        """
        return self.graph.invoke({
            "query": question,
            "chat_history": chat_history
        })

# --- Main block for direct script execution (interactive terminal mode) ---
def interactive_mode(agent: RAGAgent):
    """Launches an interactive command-line chat session with the agent."""
    print("🚀 Agent RAG (v8) Initialized - Interactive Mode")
    print("Type 'quit', 'exit', or 'q' to end the session.")
    
    chat_history = []
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
            
            print("\n" + "="*60)
            final_state = agent.query(question, chat_history)
            
            final_answer = final_state.get("answer", "No answer was generated.")
            print("📝 Final Answer:\n")
            print(final_answer)
            
            # Update history for the next turn
            chat_history.append((question, final_answer))
            
            source_docs = final_state.get('source_documents', [])
            if source_docs:
                print("\n" + "-"*40)
                print("📚 Sources Used:")
                for i, doc in enumerate(source_docs):
                    source_name = doc.metadata.get('source', 'Unknown Source')
                    page = doc.metadata.get('page')
                    display_name = f"  [{i+1}] {source_name}"
                    if page is not None:
                        display_name += f" (Page: {int(page) + 1})"
                    print(display_name)
            
            print("="*60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred in interactive mode: {e}", exc_info=True)
            
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    if not os.path.exists(RAGConfig.CHROMA_DB_PATH):
        logger.error(f"Database not found at {RAGConfig.CHROMA_DB_PATH}. Please run index.py first.")
    else:
        rag_agent = RAGAgent()
        interactive_mode(rag_agent)