# Conversational Economic RAG Agent

This project is a sophisticated, conversational Retrieval-Augmented Generation (RAG) agent built with Python, LangChain, and Ollama. It ingests economic reports (e.g., from OECD, IMF, World Bank) and answers user questions based on the content of these documents. The agent is accessible through a user-friendly web interface created with Streamlit.

## Key Features

-   **Conversational Memory**: The agent remembers the context of the conversation, allowing for natural follow-up questions.
-   **Advanced RAG Pipeline**: Implements a modern RAG architecture using LangGraph for complex, stateful logic.
-   **Relevance Filtering**: A multi-step process to ensure high-quality answers:
    -   **Relevance Check**: An initial gate to filter out off-topic questions.
    -   **Re-Ranking**: Uses a Cross-Encoder model to re-rank retrieved documents for maximum relevance, significantly reducing hallucinations and irrelevant context.
-   **Web & Local Search**: Can perform a hybrid search, querying both the local document database (ChromaDB) and the web (via Tavily API) for comprehensive answers.
-   **Sourced Answers**: All information provided by the agent is cited with the source document and page number.
-   **Interactive UI**: A clean, modern chat interface built with Streamlit.

## Tech Stack

-   **Backend**: Python
-   **LLM Orchestration**: LangChain, LangGraph
-   **LLMs**: Ollama (serving local models like Llama 3 and Gemma)
-   **Vector Database**: ChromaDB
-   **Embeddings & Re-Ranking**: Sentence-Transformers
-   **Web Interface**: Streamlit
-   **Web Search**: Tavily API

---

## 🚀 Setup and Installation

Follow these steps to get the project running locally.

### 1. Prerequisites

-   Python 3.10+
-   [Ollama](https://ollama.com/) installed and running.
-   Git

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Configure Ollama Models

Pull the necessary LLMs that the agent uses. The agent is configured to use `llama3:8b` for generation and `gemma:2b` for classification tasks.

```bash
ollama pull llama3:8b
ollama pull gemma:2b
```

### 6. Set Up Environment Variables

Create a `.env` file in the root of the project directory by copying the example.

```bash
# On Windows (if you don't have a .env file yet)
copy .env.example .env
# On macOS/Linux
cp .env.example .env
```

Now, open the `.env` file and add your Tavily API key:

```env
# .env
TAVILY_API_KEY="your_tavily_api_key_here"
```

### 7. Add Your Documents

Place all your PDF reports into the `data/reports` directory. If these directories don't exist, create them.

### 8. Create the Vector Database

Run the indexing script to process your PDFs and create the local ChromaDB vector store. This only needs to be done once, or whenever you add new documents.

```bash
python index.py
```
This process might take a few minutes depending on the number and size of your documents.

---

## ▶️ Running the Application

1.  **Ensure Ollama is Running**: Make sure the Ollama application is running in the background.
2.  **Launch the Streamlit App**: Run the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

Your web browser will automatically open with the chat interface, ready for you to ask questions.