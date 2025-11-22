# 🧠 Privacy-First RAG Knowledge Assistant

## 📋 Executive Summary

This project implements a Full-Stack Retrieval-Augmented Generation (RAG) system designed for secure, offline document analysis. Unlike cloud-based solutions (e.g., GPT-4), this architecture runs entirely on-premise, ensuring data sovereignty and GDPR compliance.

The system leverages Ollama for local LLM inference (Llama 3), ChromaDB for vector storage, and a decoupled React + FastAPI architecture. It features conversational memory, allowing for context-aware follow-up questions, and provides precise source citations for every answer to mitigate hallucinations.

## ✨ Key Features

1. 🔒 Privacy-By-Design (Local Inference)

    Zero Data Leakage: No data is sent to external APIs (OpenAI/Anthropic). All processing happens locally.

    Tech: Uses Ollama running Meta Llama 3, integrated via langchain-community.

2. 🧠 Context-Aware Conversational Memory

    Problem Solved: Standard RAG systems treat every query in isolation.

    Solution: Implemented a History-Aware Retriever. The system reformulates the user's latest query based on the chat history before performing the vector search.

    Example: User asks "How much does it cost?" after discussing a specific product -> System searches for "Product X cost".

3. 🔍 Granular Source Citation

    Transparency: Every generated response includes a sources field pointing to the specific page numbers of the PDF used to generate the answer.

    Hallucination Mitigation: Users can verify the ground truth immediately.

4. ⚡ Modern Full-Stack Implementation

    Frontend: React (Vite) with a component-based architecture and responsive UI.

    Backend: FastAPI for high-performance, asynchronous request handling and auto-generated Swagger documentation.

## 🛠️ Tech Stack

<b>LLM Inference:</b>  Ollama (Llama3) -> Local Large Language Model

<b>Orchestration:</b>	LangChain (Python) ->	Chain management, Memory, Prompt Engineering

<b>Vector DB</b>	ChromaDB ->	Local embedding storage and similarity search

<b>Backend</b>	FastAPI (Python) ->	API Endpoint management

<b>Frontend</b>	React + Vite ->	User Interface

<b>Embeddings</b>	HuggingFace (All-MiniLM) ->	Converting text to vectors locally

## 🚀 Installation & Setup
### Prerequisites
Python 3.10+
Node.js & npm

Ollama installed and running (Download [here](https://ollama.com/download))

Open a terminal and pull the model (this might take a few minutes depending on connection):
```bash
ollama run llama3
```

Setup Backend
```bash
cd backend
pip install -r requirements.txt
```
Setup Frontend
```bash
cd frontend
npm install
```

### 🏃‍♂️ Usage
Start the Backend Server:
```bash
cd backend
python main.py
# Server running at http://localhost:8000
```
Start the Frontend Server:
```bash
cd frontend
npm run dev
# Client running at http://localhost:5173
```

## 🔮 Future Roadmap 
Automated Evaluation: Implementing RAGAS (RAG Assessment) to quantitatively measure Context Precision and Faithfulness.

Hybrid Search: Integrating Keyword Search (BM25) with Vector Search to improve retrieval on specific technical terms.

Re-Ranking: Adding a Cross-Encoder step (e.g., bge-reranker) to re-order retrieved documents for higher relevance before generation.

Dockerization: Creating a unified docker-compose.yml for easy deployment.
