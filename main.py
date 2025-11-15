from fastapi import FastAPI, Request
from dotenv import load_dotenv
import requests
import os
from pydantic import BaseModel
from RAG.memmory import ChatMemory
load_dotenv()
app = FastAPI(title="RAG Chatbot with Agent Integration")

AGENT_API_URL = os.getenv("AGENT_API_URL")

memory = ChatMemory()

chat_history = []

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

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
        print("VVG",AGENT_API_URL,"NNH",testQueryBody)
        agent_resp = requests.post(AGENT_API_URL, json=testQueryBody)
        agent_output = agent_resp.json()
        bot_response = agent_output.get('agent_response', agent_output.get('message', {}).get('content', 'Fallback response'))
        # memory.add_message(payload.get('session_id'), payload['query'], bot_response)
        print("ZZX",bot_response)
    except Exception as e:
        agent_output = {"error": str(e)}

    return {
        # "user_query": query,
        # "rag_context": context,
        "agent_response": agent_output['message']['content']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
