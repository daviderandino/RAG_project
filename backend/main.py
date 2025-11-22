import uvicorn
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma

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

vectorstore = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[Message] = [] 

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/documents")
async def get_documents():
    """Restituisce la lista dei PDF salvati nella Knowledge Base."""
    if not os.path.exists("static"):
        return []
    
    files = []
    for filename in os.listdir("static"):
        if filename.endswith(".pdf"):
            files.append(filename)
    return files

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    
    # Usa il nome originale del file (così ne puoi caricare diversi)
    # Attenzione: pulisci il nome se ha spazi strani, ma per ora va bene così
    file_path = f"static/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)
        
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Qui la magia: Chroma AGGIUNGE i documenti al DB esistente, non lo cancella
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embedding, 
            persist_directory="./chroma_db"
        )
        
        return {
            "message": "Documento aggiunto alla Knowledge Base!", 
            "filename": file.filename,
            "pdf_url": f"http://localhost:8000/static/{file.filename}"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global vectorstore
    
    if vectorstore is None:
        if os.path.exists("./chroma_db"):
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        else:
            return ChatResponse(answer="Please load a PDF first!", sources=[])

    ollama_url = os.getenv("OLLAMA_BASE_URL","http://localhost:11434")

    llm = ChatOllama(model="llama3", ## phi3 is faster
                     temperature=0,
                     base_url = ollama_url) 
    
    chat_history = []
    for msg in request.history:
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))


    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )

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
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": chat_history
    })
    
    sources = []
    if "context" in response:
        for doc in response["context"]:
            src = f"Pagina {doc.metadata.get('page', '?')}"
            if src not in sources:
                sources.append(src)

    return ChatResponse(answer=response["answer"], sources=sources)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)