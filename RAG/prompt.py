from typing import List, Dict, Any

def build_prompt(query: str, history: List[dict], docs: List[object]):
        """
        Build a prompt that includes recent chat history and retrieved documents.
        Returns (prompt_text, sources)
        """
    # history is list of {user:..., bot:...}
        history_text = ""
    # include last N turns
        for turn in history:
            history_text += f"User: {turn['user']}\nBot: {turn['bot']}\n"


        # docs is a list of Document objects (LangChain) with page_content and metadata
        sources = []
        context_text = ""
        for i, d in enumerate(docs):
            content = d.page_content
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source") or meta.get("id") or f"doc_{i}"
            sources.append({"source": src, "snippet": content[:800]})
            context_text += f"--- Source: {src} ---\n{content}\n\n"


        prompt = f"You are a helpful assistant. Use the context below to answer the user's question.\n\nContext:\n{context_text}\nConversation history:\n{history_text}\nUser question: {query}\n\nAnswer concisely and cite sources by their Source names."

        return prompt, sources



NON_HALLUCINATING_PROMPT = """
You are a retrieval-augmented assistant. Follow these rules strictly:
1) Use ONLY the provided context. Do NOT invent facts.
2) If the context does not contain the answer, say exactly:
   "I don't know based on the provided information."
3) Provide short factual answer and list sources used (brief).
4) Be concise.

Context:
{context}

Conversation History:
{history}

User question:
{question}

Answer:
"""

    
def build_non_hallucinating_prompt(query: str, history: List[Dict], docs: List[Any]) :
    """
    Build the final prompt_text and the source metadata list for returned docs.
    """
    # join sources with separators and small snippets
    assembled = []
    sources_meta = []
    for i, d in enumerate(docs):
        snippet = d.page_content
        assembled.append(f"[Source {i+1}]\n{snippet}")
        meta = getattr(d, "metadata", {}) or {}
        sources_meta.append({"source": meta.get("source", f"doc_{i+1}"), "id": meta.get("id", None)})
    context_text = "\n\n".join(assembled)
    history_text = "\n".join([f"User: {m.get('user')}\nBot: {m.get('bot')}" for m in history]) if history else ""
    prompt = NON_HALLUCINATING_PROMPT.format(context=context_text, history=history_text, question=query)
    return prompt, sources_meta
