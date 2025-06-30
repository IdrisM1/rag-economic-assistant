import os
from dotenv import load_dotenv
from typing import Dict, Any, List

# Charger les variables d'environnement du fichier .env
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
import logging
from tavily_agent import TavilySearchAgent
from langchain_huggingface import HuggingFaceEndpoint

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class RAGConfig:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_DB_PATH = "./chroma_db"
    LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    MAX_CONTEXT_LENGTH = 16000

# État de l'agent
class AgentState(BaseModel):
    query: str = ""
    documents: List[Document] = []
    tavily_results: List[Dict] = []
    context: str = ""
    answer: str = ""
    source_documents: List[Document] = []
    question_type: str = ""
    strategy_log: List[str] = []
    critique: str = ""
    relevance: str = ""

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
        self.llm = HuggingFaceEndpoint(
            repo_id="google/gemma-2-9b-it", # <-- NOUVEAU MODÈLE, TRÈS PERFORMANT
            temperature=0.2,
            max_new_tokens=1024,
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )
        logger.info("✅ Composants initialisés avec succès")
    
    def query(self, question: str) -> Dict[str, Any]:
        return self.graph.invoke({"query": question})

    def _relevance_check_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Vérifie si la question est pertinente par rapport au domaine d'expertise de l'agent.
        """
        logger.info("🛡️  Vérification de la pertinence de la question...")
        
        prompt = f"""Tu es le gardien d'un assistant IA spécialisé en économie, finance et statistiques. Évalue si la question de l'utilisateur est pertinente pour ce domaine.
Domaines pertinents: Macroéconomie, croissance (PIB), politique monétaire, inflation, taux d'intérêt, stabilité financière, dette, commerce mondial, emploi, rapports de l'OCDE, FMI, BCE, Fed, OMC.
Exemples pertinents: "Prévisions de croissance pour la France ?", "Comparer la politique de la BCE et de la Fed."
Exemples NON pertinents: "D'où vient le chocolat ?", "Recette de la quiche lorraine ?", "Qui a gagné la coupe du monde ?"
Instructions: Analyse la question. Réponds UNIQUEMENT par "pertinente" ou "non_pertinente".
Question: "{state.query}"
Évaluation:"""

        relevance_result = self.llm.invoke(prompt).strip().lower()
        logger.info(f"🔎 Résultat de la vérification de pertinence : {relevance_result}")

        # =================== CORRECTION DE LA LOGIQUE ===================
        # On vérifie si la réponse contient "non_pertinente". C'est non ambigu.
        if "non_pertinente" in relevance_result:
            return {"relevance": "off_topic"}
        else:
            return {"relevance": "on_topic"}
        # ================================================================

    def _off_topic_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.warning("❌ Question jugée hors sujet. Fin du processus.")
        off_topic_message = "Je suis un assistant spécialisé en économie et finance. Je ne peux malheureusement pas répondre aux questions sur des sujets généraux comme celui-ci."
        return {"answer": off_topic_message, "source_documents": []}

    def _classify_question_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🔎 Classification de la question (factuelle/ouverte)...")
        prompt = f"""Analyse la question et classifie-la comme "factuelle" ou "ouverte".
Une question **factuelle** demande une donnée précise (chiffre, date, définition).
Une question **ouverte** nécessite une synthèse, analyse ou comparaison.
Question: "{state.query}"
Classification:"""
        result = self.llm.invoke(prompt).strip().lower()
        question_type = "ouverte" if "ouverte" in result else "factuelle"
        logger.info(f"🔎 Type de question : {question_type}")
        return {"question_type": question_type}

    def _search_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"🔍 Stratégie 'search' (locale) avec k=5...")
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        documents = retriever.invoke(state.query)
        return {"documents": documents, "tavily_results": [], "strategy_log": state.strategy_log + ["search"]}

    def _hybrid_search_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"🔍 Stratégie 'hybrid_search' (locale k=10 + web)...")
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 10})
        local_docs = retriever.invoke(state.query)
        tavily_results = self.tavily_agent.search(state.query)
        return {"documents": local_docs, "tavily_results": tavily_results, "strategy_log": state.strategy_log + ["hybrid_search"]}

    def _web_search_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info(" G Basculant vers une stratégie de recherche web uniquement (web_search)...")
        tavily_results = self.tavily_agent.search(state.query)
        return {"documents": [], "tavily_results": tavily_results, "strategy_log": state.strategy_log + ["web_search"]}

    def _context_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🏗️  Construction du contexte et des sources...")
        # CORRECTION: On accède aux attributs avec la notation par point
        all_docs_from_state = state.documents
        tavily_results = state.tavily_results
        
        source_documents = []
        context_parts = []
        
        if all_docs_from_state:
            source_documents.extend(all_docs_from_state)

        if tavily_results:
            tavily_docs = [Document(page_content=res.get("content", ""), metadata={"source": res.get("url", "N/A")}) for res in tavily_results]
            source_documents.extend(tavily_docs)

        if not source_documents:
            return {"context": "Aucun contexte pertinent trouvé.", "source_documents": []}

        for i, doc in enumerate(source_documents):
            header = f"Source [{i+1}] (de: {doc.metadata.get('source', 'inconnue')})"
            context_parts.append(f"{header}:\n{doc.page_content}")
        
        full_context = "\n\n---\n\n".join(context_parts)
        if len(full_context) > self.config.MAX_CONTEXT_LENGTH:
            full_context = full_context[:self.config.MAX_CONTEXT_LENGTH]
            logger.warning("Le contexte a été tronqué.")

        return {"context": full_context, "source_documents": source_documents}

    def _answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🤖 Génération de la réponse...")

        # =================== DÉBUT DU NOUVEAU PROMPT "ANALYSTE" ===================
        prompt = f"""Tu es un analyste économique expert. Ton rôle est de synthétiser les informations du contexte fourni pour répondre à la question de l'utilisateur de la manière la plus complète et utile possible.

**Contexte Fourni :**
Le contexte est une liste de sources numérotées (ex. Source [1], Source [2], ...).
---
{state.context}
---

**Instructions impératives :**
1.  Analyse l'ensemble du contexte pour te forger une vue d'ensemble.
2.  Construis une réponse détaillée et bien structurée. Si les sources contiennent des données chiffrées (comme des taux par pays), présente-les sous forme de liste à puces pour plus de clarté.
3.  Cite le numéro de la source pour chaque information que tu utilises, en utilisant le format `[1]`, `[2]`, etc.
4.  Si le contexte ne permet pas de répondre à la totalité de la question, réponds à la partie pour laquelle tu as des informations et précise le périmètre (par ex., "Pour le mois de mai 2024, les données disponibles sont..."). Ne te contente pas de dire que tu ne sais pas.
5.  Si le contexte est totalement vide ou non pertinent, réponds UNIQUEMENT par : "Je ne dispose pas des informations nécessaires pour répondre à cette question.".

**Question :** {state.query}

**Réponse d'expert (détaillée, structurée et citant les sources numériques) :**"""
        # ==================== FIN DU NOUVEAU PROMPT "ANALYSTE" =====================

        answer = self.llm.invoke(prompt)
        return {"answer": answer}

    def _critique_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("🤔 Critique de la réponse générée...")
        prompt = f"""Évalue la réponse. Si elle indique ne pas savoir (ex. "Je ne dispose pas des informations..."), réponds "insatisfaisante". Sinon, réponds "satisfaisante".
Réponse à évaluer: "{state.answer}"
Critique:"""
        critique_result = self.llm.invoke(prompt).strip().lower()
        logger.info(f"🔎 Résultat de la critique : {critique_result}")
        return {"critique": "rerun" if "insatisfaisante" in critique_result else "end"}

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AgentState)

        # Ajout de tous les noeuds
        workflow.add_node("relevance_check", self._relevance_check_node)
        workflow.add_node("off_topic_answer", self._off_topic_answer_node)
        workflow.add_node("classify_question", self._classify_question_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("hybrid_search", self._hybrid_search_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("context", self._context_node)
        workflow.add_node("answer", self._answer_node)
        workflow.add_node("critique_answer", self._critique_answer_node)

        # --- CÂBLAGE FINAL ET CORRIGÉ DU GRAPHE ---
        
        # 1. Le point d'entrée est le gardien de pertinence
        workflow.set_entry_point("relevance_check")

        # 2. Aiguillage après le gardien
        # CORRECTION FINALE : On utilise la notation par point (x.relevance) car l'état est un objet Pydantic
        workflow.add_conditional_edges(
            "relevance_check",
            lambda x: x.relevance,
            {
                "on_topic": "classify_question",
                "off_topic": "off_topic_answer"
            }
        )
        
        # 3. La branche hors-sujet se termine immédiatement
        workflow.add_edge("off_topic_answer", END)

        # 4. Aiguillage pour la stratégie de recherche (si la question est pertinente)
        # CORRECTION FINALE : On utilise la notation par point (x.question_type)
        workflow.add_conditional_edges(
            "classify_question",
            lambda x: x.question_type,
            {
                "factuelle": "search",
                "ouverte": "hybrid_search"
            }
        )

        # 5. Connexions vers le contexte et la réponse
        workflow.add_edge("search", "context")
        workflow.add_edge("hybrid_search", "context")
        workflow.add_edge("web_search", "context")
        workflow.add_edge("context", "answer")
        
        # 6. Connexion vers la critique
        workflow.add_edge("answer", "critique_answer")

        # 7. Aiguillage pour la boucle de réflexion
        def decide_next_step(state: AgentState) -> str:
            if state.critique == "end":
                return "end"
            else:
                if "web_search" not in state.strategy_log:
                    return "rerun_with_web"
                else:
                    return "end"

        workflow.add_conditional_edges(
            "critique_answer",
            decide_next_step,
            {
                "rerun_with_web": "web_search",
                "end": END
            }
        )

        logger.info("Graphe RAG compilé avec la logique de routage finale et correcte.")
        return workflow.compile()

    def query(self, question: str) -> Dict[str, Any]:
        return self.graph.invoke({"query": question})

    def stream_query(self, question: str):
        """
        Exécute le graphe en mode streaming pour suivre la progression.
        """
        return self.graph.stream({"query": question})
        
    def interactive_mode(self):
        print("🚀 Agent RAG initialisé - Mode interactif")
        print("Tapez 'quit', 'exit' ou 'q' pour quitter")
        while True:
            try:
                question = input("\n❓ Votre question : ").strip()
                if question.lower() in ['quit', 'exit', 'q']: break
                if not question: continue
                
                print("\n" + "="*60)
                result = self.query(question)
                
                print("📝 Réponse Finale :\n")
                print(result.get("answer", "Aucune réponse générée."))
                
                # --- NOUVELLE LOGIQUE D'AFFICHAGE DES SOURCES ---
                source_docs = result.get('source_documents')
                if source_docs:
                    print("\n" + "-"*40)
                    print("📚 Sources utilisées pour générer cette réponse :")
                    for i, doc in enumerate(source_docs):
                        source_name = doc.metadata.get('source', 'Source inconnue')
                        page = doc.metadata.get('page')
                        display_name = f"  [{i+1}] {source_name}"
                        if page is not None:
                            display_name += f" (Page: {page})"
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