from fastapi import FastAPI, Request, HTTPException,File, Form,UploadFile
from flask import jsonify
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests
import os
from pydantic import BaseModel
from RAG.memmory import ChatMemory
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from RAG.embedding import load_vector_store
import json
import tempfile
load_dotenv()
app = FastAPI(title="RAG Chatbot with Agent Integration")

from RAG.rag_engine import answer_with_rag
from RAG.memmory import ChatMemory


AGENT_API_URL = os.getenv("AGENT_API_URL")

memory = ChatMemory()

chat_history = []

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


@app.post("/chatWithRAG")
async def chat_endpoint(request: Request):
    payload = await request.json()
    session_id = payload.get("session_id", "default")
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # retrieve memory for session
    history = memory.get_messages(session_id)

    # run rag flow
    result = answer_with_rag(query, history)

    # persist
    print("SSsession_id",session_id)
    memory.add_message(session_id, query, result["answer"])

    return JSONResponse(result)


@app.post("/chat")
async def chat(req: ChatRequest):
    payload =req.dict()
    print("FFrrmmk",payload)
    session_id = payload.get("session_id")
    query = payload.get("query")
    # response = answer_with_rag(query, chat_history)
    chat_history.append({"user": query})

    print("NNJ",chat_history)
    testQueryBody={
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_ctx": 2048
  }
}
    
    # Step 3: Call your local agent API
    try:
        agent_resp = requests.post(AGENT_API_URL, json=testQueryBody)
        agent_output = agent_resp.json()
        bot_response = agent_output.get('agent_response', agent_output.get('message', {}).get('content', 'Fallback response'))
        memory.add_message(session_id, query, bot_response)
    except Exception as e:
        agent_output = {"error": str(e)}

    return {
        "user_query": query,
        # "rag_context": context,
        "agent_response": agent_output
    }
    
    
@app.get("/memory/{session_id}")
def get_chat_history(session_id: str):
    history = memory.get_messages(session_id)
    return {
        "session_id": session_id,
        "messages": history,
        "count": len(history)
    }
    
BASE_VECTOR_DIR = "data"
   
@app.post("/ingest_docs")    
async def ingest_docs(
    request: Request,
    text: str = Form(None),
    file: UploadFile = File(None)
    ):
    """
    Ingest documents into the vector store for a user.
    Body:
        userId: string
        text: string (optional)
        file: file upload (optional)
    """
    print("KKU",request)
    # text = request.form.get("text")
    # file = request.files.get("file")
    print("FFI",text)
    print("FFF",file)
    docs = []

    # Case 1 — text ingestion
    if text:
        docs.append(Document(page_content=text, metadata={"source": "text"}))

    # Case 2 — file ingestion
    if file:
    # Create safe temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        print("TTMP",tmp_path)    
    # Load file
    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
        print("PPDF LOADER")
    else:
        loader = TextLoader(tmp_path)
        print("TTEXT LOADER")

    docs.extend(loader.load())
    if not docs:
        return jsonify({"error": "No valid documents found"}), 400

    
    # Load vector store
    vs = load_vector_store()

    # Add and persist
    vs.add_documents(docs)
    storage = os.path.join(BASE_VECTOR_DIR, f"faiss_index")
    vs.save_local(storage)

    return JSONResponse(
        content={"message": "Documents added", "count": len(docs)},
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
