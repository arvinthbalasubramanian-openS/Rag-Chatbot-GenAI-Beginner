# RAG PDF Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask natural language questions about their content.

Built using:
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama (Mistral LLM)

---

# Features

- Upload PDF documents
- Extract and process PDF text
- Chunk large documents intelligently
- Generate semantic embeddings
- Store embeddings in a FAISS vector database
- Retrieve relevant document context
- Answer questions using a Large Language Model (LLM)
- Interactive Streamlit web interface
- Fully local LLM support using Ollama

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Streamlit | Web interface |
| LangChain | RAG pipeline |
| FAISS | Vector database |
| HuggingFace Embeddings | Semantic embeddings |
| Ollama | Local LLM serving |
| Mistral | Language model |
| pdfplumber | PDF text extraction |

---

# Project Workflow

1. User uploads a PDF document
2. PDF text is extracted using `pdfplumber`
3. Text is split into smaller chunks
4. HuggingFace embeddings are generated for each chunk
5. Embeddings are stored in a FAISS vector database
6. User asks questions
7. Relevant chunks are retrieved using semantic search
8. Retrieved context is passed to the Mistral LLM
9. AI-generated response is displayed in Streamlit

---

# Requirements

- Python 3.11 recommended
- Ollama installed locally
- Mistral model downloaded

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/arvinthbalasubramanian-openS/rag-chatbot.git
cd rag-chatbot