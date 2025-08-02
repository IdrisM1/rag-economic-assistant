import os
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple

# Charger les variables d'environnement du fichier .env
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
import logging
from tavily_agent import TavilySearchAgent
# NOUVEAU : Importation pour le re-ranking
from sentence_transformers.cross_encoder import CrossEncoder


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# MODIFIÉ : Configuration enrichie pour le re-ranking
class RAGConfig:
    """Configuration centralisée pour l'agent RAG."""
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_DB_PATH = "./chroma_db"
    
    LLM_GENERATOR_MODEL = "llama3:8b"
    LLM_CLASSIFIER_MODEL = "gemma:2b"
    
    # NOUVEAU : Paramètres pour le Re-Ranking
    INITIAL_SEARCH_K = 15  # Récupérer plus de documents au début
    RERANK_MODEL = "cross-encoder/ms-marco-minilm-l-6-v2" # Modèle léger et efficace
    RERANK_TOP_K = 4       # Garder les 4 meilleurs après reclassement
    
    MAX_CONTEXT_LENGTH = 16000

# MODIFIÉ : État de l'agent avec mémoire de conversation
class AgentState(BaseModel):
    """État de l'agent, circulant dans le graphe."""
    query: str = ""
    documents: List[Document] = []
    # NOUVEAU : Historique de la conversation
    chat_history: List[Tuple[str, str]] = [] # Liste de tuples (humain, ia)
    
    tavily_results: List[Dict] = []
    context: str = ""
    answer: str = ""
    source_documents: List[Document] = []
    strategy_log: List[str] = []
    critique: str = ""
    relevance: str = ""
    local_search_sufficiency: str = ""

class RAGAgent:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self._initialize_components()
        if "TAVILY_API_KEY" not in os.environ:
            raise ValueError("❌ Clé API Tavily (TAVILY_API_KEY) non trouvée.")
        self.tavily_agent = TavilySearchAgent(api_key=os.environ["TAVILY_API_KEY"])
        self.graph = self._build_graph()

    def _initialize_components(self):
        logger.info("Initialisation des composants...")
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
        self.vectordb = Chroma(persist_directory=self.config.CHROMA_DB_PATH, embedding_function=self.embedding_function)
        
        self.llm_generator = OllamaLLM(model=self.config.LLM_GENERATOR_MODEL, temperature=0.1)
        self.llm_classifier = OllamaLLM(model=self.config.LLM_CLASSIFIER_MODEL, temperature=0.0)

        logger.info(f"Chargement du modèle de re-ranking : {self.config.RERANK_MODEL}")
        self.reranker = CrossEncoder(self.config.RERANK_MODEL)
        
        self._initialize_prompts()
        logger.info("✅ Composants initialisés avec succès")

    def _initialize_prompts(self):
        """Définit tous les PromptTemplates utilisés par l'agent."""
        self.condense_question_prompt = PromptTemplate(
            template="""Étant donné l'historique de la conversation et une nouvelle question, reformule la nouvelle question pour en faire une question **autonome** et claire que l'on pourrait comprendre sans l'historique. NE RÉPONDS PAS à la question, contente-toi de la reformuler.

Historique de la conversation:
{chat_history}

Nouvelle Question: {query}

Question Autonome:""",
            input_variables=["chat_history", "query"],
        )
        
        self.relevance_prompt = PromptTemplate(
            template="""Analyse la question suivante. Ton unique tâche est de répondre en un seul mot, en français. Réponds UNIQUEMENT par "pertinente" si la question concerne l'économie, la finance ou les statistiques, ou par "non_pertinente" dans tous les autres cas.
Question: "{query}"
Réponse (un seul mot):""",
            input_variables=["query"],
        )
        
        self.evaluation_prompt = PromptTemplate(
            template="""Évalue si le contexte fourni est **directement pertinent** et **suffisant** pour répondre à la question. Réponds UNIQUEMENT par "suffisant" ou "insuffisant".
Le contexte est pertinent s'il aborde le sujet exact de la question. Par exemple, si la question porte sur la 'dette publique', le contexte doit parler de 'dette publique', et non juste de la 'Banque de France'.

Question: "{query}"
Contexte: "{context_preview}"
Réponse (suffisant/insuffisant):""",
            input_variables=["query", "context_preview"],
        )

        self.answer_prompt = PromptTemplate(
            template="""Tu es un analyste économique expert. Ton rôle est de synthétiser les informations du contexte fourni pour répondre à la question de l'utilisateur.

Contexte Fourni :
---
{context}
---

**Instructions IMPÉRATIVES :**
1.  **Adéquation stricte :** Base ta réponse UNIQUEMENT sur les informations contenues dans le contexte fourni. Ne fais aucune supposition et n'ajoute pas d'informations externes.
2.  **Honnêteté directe :** Si le contexte ne contient pas de réponse directe à la question, commence ta réponse par "D'après les documents fournis, il n'y a pas d'information précise concernant...".
3.  **Pertinence avant tout :** Si tu ne trouves pas de réponse directe, ne synthétise des informations connexes que si elles apportent un éclairage DIRECTEMENT pertinent à la question. **Ne tente JAMAIS de créer des liens entre des sujets non pertinents.**
4.  **Structure et clarté :** Cite tes sources après chaque information avec le format `[1]`, `[2]`, etc.
5.  **Conclusion concise :** Termine par une brève synthèse de 1 à 2 phrases.

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

    def _condense_question_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🧠 Condensation de la question avec l'historique...")
        if not state.chat_history:
            logger.info("Pas d'historique, la question reste inchangée.")
            return {"query": state.query}

        history_str = "\n".join([f"Humain: {q}\nIA: {a}" for q, a in state.chat_history])
        chain = self.condense_question_prompt | self.llm_classifier
        new_query = chain.invoke({"chat_history": history_str, "query": state.query})
        logger.info(f"Nouvelle question autonome : {new_query}")
        return {"query": new_query}

    def _relevance_check_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🛡️  Vérification de la pertinence de la question...")
        chain = self.relevance_prompt | self.llm_classifier
        relevance_result = chain.invoke({"query": state.query}).strip().lower()
        logger.info(f"🔎 Résultat de la vérification de pertinence : {relevance_result}")
        return {"relevance": "off_topic" if "non_pertinente" in relevance_result else "on_topic"}

    def _off_topic_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.warning("❌ Question jugée hors sujet. Fin du processus.")
        off_topic_message = "Je suis un assistant spécialisé en économie et finance. Je ne peux malheureusement pas répondre à cette question."
        return {"answer": off_topic_message, "source_documents": []}

    def _search_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"🔍 Recherche locale large (k={self.config.INITIAL_SEARCH_K})...")
        retriever = self.vectordb.as_retriever(search_kwargs={"k": self.config.INITIAL_SEARCH_K})
        documents = retriever.invoke(state.query)
        return {"documents": documents, "tavily_results": []}

    def _rerank_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"🔁 Reclassement des {len(state.documents)} documents trouvés...")
        if not state.documents:
            return {"documents": []}

        pairs = [(state.query, doc.page_content) for doc in state.documents]
        scores = self.reranker.predict(pairs)

        scored_docs = zip(scores, state.documents)
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        top_docs = [doc for score, doc in sorted_docs[:self.config.RERANK_TOP_K]]
        logger.info(f"✅ Reclassement terminé. {len(top_docs)} documents conservés.")
        return {"documents": top_docs}

    def _evaluate_local_results_node(self, state: AgentState) -> Dict[str, str]:
        logger.info("⚖️  Évaluation de la suffisance des résultats locaux...")
        if not state.documents:
            logger.warning("Aucun document local trouvé. La recherche web est nécessaire.")
            return {"local_search_sufficiency": "insufficient"}
        context_preview = "\n---\n".join([doc.page_content for doc in state.documents])
        chain = self.evaluation_prompt | self.llm_classifier
        result = chain.invoke({"query": state.query, "context_preview": context_preview[:4000]}).strip().lower()
        logger.info(f"🔎 Résultat de l'évaluation locale : {result}")
        return {"local_search_sufficiency": "insufficient" if "insuffisant" in result else "sufficient"}

    def _hybrid_search_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"🔍 Stratégie enrichie : Recherche hybride (locale k={self.config.INITIAL_SEARCH_K} + web)...")
        retriever = self.vectordb.as_retriever(search_kwargs={"k": self.config.INITIAL_SEARCH_K})
        local_docs = retriever.invoke(state.query)
        tavily_results = self.tavily_agent.search(state.query)
        # On passe les documents locaux au state pour qu'ils soient reclassés
        return {"documents": local_docs, "tavily_results": tavily_results}

    def _web_search_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(" G Basculant vers une stratégie de recherche web uniquement (web_search)...")
        tavily_results = self.tavily_agent.search(state.query)
        return {"documents": state.documents, "tavily_results": tavily_results}

    def _context_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🏗️  Construction du contexte et des sources...")
        source_documents = list(state.documents)
        if state.tavily_results:
            tavily_docs = [Document(page_content=res.get("content", ""), metadata={"source": res.get("url", "N/A")}) for res in state.tavily_results]
            source_documents.extend(tavily_docs)
        if not source_documents:
            return {"context": "Aucun contexte pertinent trouvé.", "source_documents": []}
        unique_contents = set()
        unique_documents = []
        for doc in source_documents:
            if doc.page_content not in unique_contents:
                unique_documents.append(doc)
                unique_contents.add(doc.page_content)
        context_parts = [f"Source [{i+1}] (de: {doc.metadata.get('source', 'inconnue')}):\n{doc.page_content}" for i, doc in enumerate(unique_documents)]
        full_context = "\n\n---\n\n".join(context_parts)
        if len(full_context) > self.config.MAX_CONTEXT_LENGTH:
            full_context = full_context[:self.config.MAX_CONTEXT_LENGTH]
            logger.warning("Le contexte a été tronqué.")
        return {"context": full_context, "source_documents": unique_documents}

    def _answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🤖 Génération de la réponse...")
        chain = self.answer_prompt | self.llm_generator
        answer = chain.invoke({"context": state.context, "query": state.query})
        return {"answer": answer}

    def _critique_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🤔 Critique de la réponse générée...")
        chain = self.critique_prompt | self.llm_classifier
        critique_result = chain.invoke({"answer": state.answer}).strip().lower()
        logger.info(f"🔎 Résultat de la critique : {critique_result}")
        return {"critique": "rerun" if "insatisfaisante" in critique_result else "end"}

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("condense_question", self._condense_question_node)
        workflow.add_node("relevance_check", self._relevance_check_node)
        workflow.add_node("off_topic_answer", self._off_topic_answer_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("evaluate_local_results", self._evaluate_local_results_node)
        workflow.add_node("hybrid_search", self._hybrid_search_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("critique_answer", self._critique_answer_node)
        workflow.add_node("web_search", self._web_search_node)

        workflow.set_entry_point("condense_question")
        workflow.add_edge("condense_question", "relevance_check")
        
        workflow.add_conditional_edges("relevance_check", lambda s: s.relevance, {"on_topic": "search", "off_topic": "off_topic_answer"})
        workflow.add_edge("off_topic_answer", END)
        
        workflow.add_edge("search", "rerank")
        workflow.add_edge("rerank", "evaluate_local_results")

        workflow.add_conditional_edges("evaluate_local_results", lambda s: s.local_search_sufficiency, {"sufficient": "context", "insufficient": "hybrid_search"})
        
        workflow.add_edge("hybrid_search", "rerank")

        workflow.add_edge("web_search", "context")
        workflow.add_edge("context", "answer")
        workflow.add_edge("answer", "critique_answer")

        def decide_next_step_after_critique(state: AgentState) -> str:
            if state.critique == "end": return "end"
            if "web_search" not in state.strategy_log:
                logger.info("Critique insatisfaisante, tentative avec une recherche web forcée.")
                return "rerun_with_web"
            logger.warning("Critique toujours insatisfaisante après la recherche web. Fin.")
            return "end"
        
        workflow.add_conditional_edges("critique_answer", decide_next_step_after_critique, {"rerun_with_web": "web_search", "end": END})
        
        logger.info("Graphe RAG compilé avec mémoire conversationnelle et re-ranking.")
        return workflow.compile()

    def query(self, question: str, chat_history: List[Tuple[str, str]]) -> Dict:
        return self.graph.invoke({
            "query": question,
            "chat_history": chat_history
        })

    def interactive_mode(self):
        """Lance une session de questions/réponses interactive avec l'agent."""
        print("🚀 Agent RAG (v7) initialisé - Conversationnel & Re-Ranking")
        print("Tapez 'quit', 'exit' ou 'q' pour quitter")
        
        chat_history = []

        while True:
            try:
                question = input("\n❓ Votre question : ").strip()
                if question.lower() in ['quit', 'exit', 'q']: break
                if not question: continue
                
                print("\n" + "="*60)
                final_state = self.query(question, chat_history)
                
                final_answer = final_state.get("answer", "Aucune réponse générée.")
                print("📝 Réponse Finale :\n")
                print(final_answer)
                
                chat_history.append((question, final_answer))
                
                source_docs = final_state.get('source_documents', [])
                if source_docs:
                    print("\n" + "-"*40)
                    print("📚 Sources utilisées pour générer cette réponse :")
                    for i, doc in enumerate(source_docs):
                        source_name = doc.metadata.get('source', 'Source locale inconnue')
                        page = doc.metadata.get('page')
                        display_name = f"  [{i+1}] {source_name}"
                        if page is not None: display_name += f" (Page: {int(page) + 1})"
                        print(display_name)
                
                print("="*60)
            except KeyboardInterrupt: break
            except Exception as e: 
                logger.error(f"❌ Erreur inattendue dans la boucle interactive: {e}", exc_info=True)
        print("\n👋 Au revoir !")

def main():
    if not os.path.exists(RAGConfig.CHROMA_DB_PATH):
        logger.error(f"Base de données {RAGConfig.CHROMA_DB_PATH} non trouvée.")
        return
    agent = RAGAgent()
    agent.interactive_mode()

if __name__ == "__main__":
    main()