import uvicorn
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# LangChain Imports Aggiornati
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma

# Nuovi import per la gestione della memoria (LCEL)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

# --- CORS ---
origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATO GLOBALE ---
vectorstore = None

# --- MODELLI DATI ---
# Aggiorniamo la richiesta per accettare la history
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = [] # Lista di messaggi precedenti

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# --- ENDPOINT UPLOAD (Invariato) ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(temp_filename)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory="./chroma_db")
        return {"message": "File processato e memoria aggiornata!"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

# --- ENDPOINT CHAT (Con Memoria) ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global vectorstore
    
    # 1. Carica il VectorStore
    if vectorstore is None:
        if os.path.exists("./chroma_db"):
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        else:
            return ChatResponse(answer="Carica prima un PDF!", sources=[])

    # 2. Configura LLM
    llm = ChatOllama(model="llama3", temperature=0)
    
    # 3. Prepara la History nel formato LangChain
    chat_history = []
    for msg in request.history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    # --- STEP A: CREARE IL RETRIEVER CONSAPEVOLE DELLA STORIA ---
    # Questo prompt dice all'LLM: "Usa la chat history per capire cosa cerca davvero l'utente"
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Questo retriever prima "riformula" la domanda, poi cerca nel DB
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )

    # --- STEP B: CREARE LA CATENA DI RISPOSTA (QA) ---
    qa_system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.\n\n
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Uniamo tutto insieme
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 4. Esegui la catena
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": chat_history
    })
    
    # 5. Estrai le fonti
    sources = []
    if "context" in response:
        for doc in response["context"]:
            src = f"Pagina {doc.metadata.get('page', '?')}"
            if src not in sources:
                sources.append(src)

    return ChatResponse(answer=response["answer"], sources=sources)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)