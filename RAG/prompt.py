from typing import List

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