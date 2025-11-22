# 🧠 Privacy-First RAG Knowledge Assistant

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![React](https://img.shields.io/badge/react-18.0+-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)

## 📋 Executive Summary

This project implements a Full-Stack Retrieval-Augmented Generation (RAG) system designed for secure, offline document analysis. Unlike cloud-based solutions (e.g., GPT-4), this architecture runs entirely on-premise, ensuring data sovereignty and GDPR compliance.

The system leverages Ollama for local LLM inference (Llama 3), ChromaDB for vector storage, and a decoupled React + FastAPI architecture. It features conversational memory, allowing for context-aware follow-up questions, and provides precise source citations for every answer to mitigate hallucinations.

## ✨ Key Features

1.  🔒 **Privacy-By-Design (Local Inference)**
    * **Zero Data Leakage:** No data is sent to external APIs (OpenAI/Anthropic). All processing happens locally.
    * **Tech:** Uses Ollama running Meta Llama 3, integrated via langchain-community.

2.  🐳 **Containerized Architecture**
    * **Dockerized:** Fully containerized application using **Docker Compose**.
    * **Reproducibility:** Ensures consistent deployment environments across different machines, eliminating "works on my machine" issues.

3.  🧠 **Context-Aware Conversational Memory**
    * **Problem Solved:** Standard RAG systems treat every query in isolation.
    * **Solution:** Implemented a **History-Aware Retriever**. The system reformulates the user's latest query based on the chat history before performing the vector search.
    * **Example:** User asks *"How much does it cost?"* after discussing a specific product -> System searches for *"Product X cost"*.

4.  🔍 **Granular Source Citation**
    * **Transparency:** Every generated response includes a `sources` field pointing to the specific page numbers of the PDF used to generate the answer.
    * **Hallucination Mitigation:** Users can verify the ground truth immediately.

5.  ⚡ **Modern Full-Stack Implementation**
    * **Frontend:** React (Vite) with a component-based architecture and responsive UI.
    * **Backend:** FastAPI for high-performance, asynchronous request handling and auto-generated Swagger documentation.

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Infrastructure** | **Docker & Compose** | Containerization & Orchestration |
| **LLM Inference** | Ollama (Llama 3) | Local Large Language Model |
| **Orchestration** | LangChain (Python) | Chain management, Memory, Prompt Engineering |
| **Vector DB** | ChromaDB | Local embedding storage and similarity search |
| **Backend** | FastAPI (Python) | API Endpoint management |
| **Frontend** | React + Vite | User Interface |
| **Embeddings** | HuggingFace (All-MiniLM) | Converting text to vectors locally

## 🚀 Installation & Setup
### Prerequisites
* **Docker Desktop** installed and running.
* **Ollama** installed on your host machine ([Download here](https://ollama.com/download)).


Open a terminal and pull the model (this might take a few minutes depending on connection):
```bash
ollama run llama3
```
### Option A: Quick Start (Docker) 🐳
The recommended way to run the project. This will start the Backend, Frontend, and Database automatically.
```bash
# Build and start containers
docker-compose up --build
```

### Option B: Manual / Development Setup 🛠️

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

Deep Linking: Making source citations interactive to jump directly to the specific paragraph in the PDF viewer.
