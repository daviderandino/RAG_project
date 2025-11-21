import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.chat_models import ChatOllama          
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import uvicorn

app = FastAPI()

origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        loader = PyPDFLoader(temp_filename)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(data)
        
        print("Generazione embeddings in corso... (potrebbe richiedere un attimo)")
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory="./chroma_db"
        )
        
        return {"message": f"File elaborato! Creati {len(splits)} chunk nel DB locale."}
        
    except Exception as e:
        print(e)
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global vectorstore
    
    if vectorstore is None:
        if os.path.exists("./chroma_db"):
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        else:
            return ChatResponse(answer="Per favore, carica prima un documento PDF.", sources=[])

    llm = ChatOllama(model="llama3", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    print("Ollama sta pensando...")
    result = qa_chain.invoke({"query": request.question})
    
    sources = []
    for doc in result['source_documents']:
        src = f"Pagina {doc.metadata.get('page', '?')}"
        if src not in sources:
            sources.append(src)
            
    return ChatResponse(
        answer=result['result'],
        sources=sources
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)