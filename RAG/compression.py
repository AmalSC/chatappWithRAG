
from typing import List, Any
from agent.agent_client import call_local_agent

# def compress_documents(docs, query, llm):
#     """
#     Takes retrieved docs and compresses them to only keep
#     the parts relevant to the query.
#     """
#     compressed_docs = []

#     for doc in docs:
#         prompt = f"""
# You are a document compression engine.

# Your goal: Keep ONLY the relevant information needed to answer a question.
# Remove unrelated sections, examples, fluff, filler, intros, conclusions.

# User Query:
# {query}

# Document:
# {doc.page_content}

# Return ONLY the relevant extracted text, nothing else.
# """
#         summary = llm.invoke(prompt)
#         doc.page_content = summary.strip()
#         compressed_docs.append(doc)

#     return compressed_docs




# -----------------------
# Document compression
# -----------------------
def compress_documents(docs: List[Any], query: str) -> List[Any]:
    """
    For each doc, produce a compressed version relevant to query using LLM.
    Mutates doc.page_content to the compressed text (returns new list).
    Keep it simple: ask LLM to return only relevant snippet(s).
    """
    compressed = []
    for doc in docs:
        prompt = f"""
You are a document compressor. Keep ONLY text that is directly relevant to answering the user's question.
User question:
{query}

Document:
{doc.page_content}

Return a short extract (1-3 paragraphs) containing only the relevant facts/lines that answer the question. If nothing relevant, return an empty string.
"""
        try:
            summary = call_local_agent(prompt)
        except Exception:
            summary = ""
        if summary:
            # create a shallow copy with compressed content (preserve metadata)
            doc.page_content = summary.strip()
            compressed.append(doc)
    return compressed
