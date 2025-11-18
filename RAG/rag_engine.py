import os
from RAG.vectorstore import get_retriever
from RAG.prompt import build_prompt,build_non_hallucinating_prompt
from agent.agent_client import call_local_agent
from RAG.compression import compress_documents
K = int(os.getenv("K", 3))



def answer_with_rag(query: str, history: list):
    """
    Main entry: retrieve docs, build prompt, call agent, return answer + sources.
    Fully updated for LangChain 0.2+ (uses retriever.invoke).
    """
    print("Flag 1")
    # --- Retrieve documents ---
    retriever = get_retriever(k=K)
    print("flag 2",retriever)
    try:
        docs = retriever(query)  # NEW API
        print("flag 3",docs)
    except Exception as e:
        print("Retriever failed:", e)
        docs = []
    print("flag 4")
    # --- Ensure docs is always a list ---
    if docs is None:
       print("NO DOCS") 
       prompt = f"You are a helpful assistant.\nUser question:\n{query}\nAnswer concisely."
       answer_text = call_local_agent(prompt)
       agent_resp = answer_text.get("bot_response", {})
       return agent_resp
    print("flag 5")
    # 2. Compress docs (query-focused)
    try:
        print("flag 6")
        compressed = compress_documents(docs, query)
        print("flag 7",compressed)
    except Exception as e:
        print("Compression failed:", e)
        compressed = docs
    # --- Build RAG prompt ---
    print("flag 8")
    if not compressed:
            print("flag 9")
            # no useful compressed context -> fallback to pure LLM
            prompt = f"You are a helpful assistant.\nUser question:\n{query}\nAnswer concisely."
            answer_text = call_local_agent(prompt)
            answer_text = answer_text.get("bot_response", {})
            print(answer_text,"vvtt")
            # return answer_text
            return {
                "answer": answer_text,
                "sources": "sources_trimmed"
            }
    # 3. Build prompt & get sources metadata
    # prompt_text, sources_meta = build_non_hallucinating_prompt(query, history, compressed)
    print("Compressed docs:", len(compressed))
    
    prompt_text, sources = build_prompt(query, history, docs)
    
    # --- Call your local agent ---
    agent_raw = call_local_agent(prompt_text)
    print("XXX",type(agent_raw),agent_raw)
    
    # --- Extract agent response ---
    agent_resp = agent_raw.get("bot_response", {})
    print("YYY",type(agent_resp),agent_resp)
    message = agent_resp

    if isinstance(message, dict):
        answer = message.get("content", "")
    else:
        answer = str(message)

    # --- Build trimmed sources ---
    sources_trimmed = []
    for s in sources:
        if isinstance(s, dict):
            sources_trimmed.append({
                "source": s.get("source", "unknown"),
                "snippet": s.get("snippet", "")
            })
        else:
            # in case build_prompt returns raw text
            sources_trimmed.append({"source": "unknown", "snippet": str(s)})

    return {
        "answer": answer,
        "sources": sources_trimmed
    }

# def answer_with_rag(query: str, history: list):
#     return {
#   "answer": "Based on your question, the Sanctuary Hope system stores patient records in MongoDB and uses a region-based rotation logic to assign questions every 7 days. The documents also show that each patient has a unique organizationId, which is used in all database filters.",
  
#   "sources": [
#     {
#       "source": "data/docs/healthcare_system.pdf",
#       "snippet": "The question rotation cycle is configured to run every Monday at 7 PM SGT... This ensures global scalability and reduces system load."
#     },
#     {
#       "source": "data/docs/patient_model.pdf",
#       "snippet": "All patient data includes an organizationId field. Any API request must filter by this organizationId during CRUD operations."
#     }
#   ]
# }
