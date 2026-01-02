# Local RAG Chat

A RAG-based chatbot using Ollama models with Chroma DB for vector database and Document Reranking for local on-device LLM inference.

- **Ollama model used**: `gemma3:4b`

- **Embedding model used**: `nomic-embed-text:v1.5`

- **Document re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

### Note

This app is inspired by the work of [yankeexe](https://github.com/yankeexe) and their [reranker-based RAG app demo](https://github.com/yankeexe/llm-rag-with-reranker-demo).

**Features:**

- **Control model parameters**: Adjust Temperature, Top-P, and Max Tokens directly from the interface.

- **Editable System Prompt**: A dedicated configuration page to customize the AI's core instructions for each session.

---
### Setting up the App

**Clone the repo**:

```
git clone https://github.com/TRJgit/Local-RAG-Chat.git
cd Local-RAG-Chat
```

**Create a virtual environment (venv)**:

```
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on MacOS/Linux:
source venv/bin/activate
```

**Install all dependencies from requirements.txt**:

```
pip install -r requirements.txt
```

**Ollama Setup**:

- Download and install Ollama from the [official library](https://ollama.com/library).

- Pull a sample LLM (e.g., Gemma 3 4B):

```
ollama pull gemma3:4b
```

**Embedding Model and Reranker**:

The embedding model [`nomic-embed-text:v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) and the reranker model [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
will download automatically on the first run.

**Run the application**:

```
streamlit run app.py
```
