from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGChatbot
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip()

app = FastAPI(title="RAG API", version="1.0")

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Single chatbot instance
chatbot = RAGChatbot(api_key=GROQ_API_KEY)

# Request/Response models
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    num_sources: int

class LoadResponse(BaseModel):
    message: str
    num_chunks: int
    source: str

# Routes
@app.get("/")
def root():
    return {"message": "RAG API is running", "version": "1.0"}

@app.post("/load/text")
def load_text(payload: dict):
    """Load a document from pasted text"""
    text = payload.get("text", "")
    source = payload.get("source", "pasted_text")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    num_chunks = chatbot.load_document(text, source=source)
    if num_chunks == 0:
        return LoadResponse(
            message="Document already loaded",
            num_chunks=0,
            source=source
        )
    return LoadResponse(
        message="Document loaded successfully",
        num_chunks=num_chunks,
        source=source
    )

@app.post("/load/file")
async def load_file(file: UploadFile = File(...)):
    """Load a document from uploaded file"""
    if not file.filename.endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files supported"
        )
    contents = await file.read()
    if file.filename.endswith(".pdf"):
        import fitz
        pdf = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
    else:
        text = contents.decode("utf-8")

    num_chunks = chatbot.load_document(text, source=file.filename)
    if num_chunks == 0:
        return LoadResponse(
            message="Document already loaded",
            num_chunks=0,
            source=file.filename
        )
    return LoadResponse(
        message="Document loaded successfully",
        num_chunks=num_chunks,
        source=file.filename
    )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question about loaded documents"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    response, sources = chatbot.chat(request.question)
    source_texts = [doc.page_content[:100] for doc in sources]
    return ChatResponse(
        answer=response,
        sources=source_texts,
        num_sources=len(sources)
    )

@app.post("/reset")
def reset():
    """Reset the chatbot"""
    chatbot.reset()
    return {"message": "Chatbot reset successfully"}

@app.get("/status")
def status():
    """Check loaded documents"""
    return {
        "loaded_sources": list(chatbot.loaded_sources),
        "num_documents": len(chatbot.loaded_sources)
    }