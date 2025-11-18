from RAG.embedding import load_vector_store
from agent.agent_client import call_local_agent
from langchain_core.language_models import BaseLLM
# from langchain.schema import LLMResult
from typing import Any, List,Optional,Dict
from pydantic import Field
from RAG.cross_encoder import rerank

class CustomGeneration:
    def __init__(self, text: str, metadata: Optional[Dict] = None):
        self.text = text
        self.metadata = metadata or {}




class CustomLLMResult:
    print("CCCCCC")
    def __init__(self, generations: List[List[CustomGeneration]], extra_info: Optional[Dict] = None):
        self.generations = generations
        self.extra_info = extra_info or {}

    def get_texts(self) -> List[str]:
        print("DDDDDD")
        return [gen.text for sublist in self.generations for gen in sublist]

    # Add this method to satisfy LangChain
    def flatten(self) -> List['CustomGeneration']:
        """
        Flatten nested generations into a single list of CustomGeneration objects.
        """
        print("EEEEEE")
        return [gen for sublist in self.generations for gen in sublist]



class OllamaLLM(BaseLLM):
    model_name: str = "local-ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Single-prompt call (used by LangChain)
        """
        result = call_local_agent(prompt)
        resp = result.get("bot_response", {})
        if isinstance(resp, dict):
            return resp.get("content", "")
        return str(resp)




    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> CustomLLMResult:
        """
        Batch call (required by BaseLLM)
        """
        generations = []
        for p in prompts:
            text = self._call(p, stop)
            generations.append([CustomGeneration(text=text)])  # <-- use CustomGeneration
        return CustomLLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "ollama-local"
    
llm = OllamaLLM()




def generate_multi_queries(query: str, llm, num=4):
    prompt = f"""
You are an expert RAG Query Expansion Engine.

Generate {num} semantically diverse alternative search queries for the user question.

Rules:
- Each query must use different wording or phrasing.
- Vary specificity (broad, narrow).
- Include 1 query that expands context.
- Include 1 query that decomposes the intent.
- Return ONLY the queries, each on a new line.

User Query: {query}
"""
    raw = llm.invoke(prompt)
    queries = [q.strip() for q in raw.split("\n") if q.strip()]
    print("GQ:",queries)
    return queries[:num]





def get_retriever(k=3):
    """
    Custom Multi-Query Retriever for local models (Ollama).
    - Expands the query using generate_multi_queries()
    - Performs multiple similarity searches
    - Deduplicates results
    """
    print("flag ###2")
    vs = load_vector_store()
    base_retriever = vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.35, "k": 2}
    )





    def retrieve(query: str):
        # ---- 1. Generate multiple expanded queries ----
        expanded_queries = generate_multi_queries(query, llm)
        
        print("Expanded Queries:", expanded_queries)
        all_docs = []

        # ---- 2. Retrieve for each expanded query ----
        for q in expanded_queries:
            try:
                docs = base_retriever.invoke(q)
                print("Docs for query",q,":",docs)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Retriever error for query '{q}':", e)

        # ---- 3. Deduplicate results ----
        unique = {}
        for doc in all_docs:
            print("Doc:",doc)
            key = getattr(doc, "id", None) or doc.page_content[:50]
            if key not in unique:
                unique[key] = doc
        print("Unique Docs Count:", len(unique))
        docs = list(unique.values())
        print("iiopp",docs)
        docs = rerank(query, docs, top_k=k)
        
        print("///start from here docs are not returning",docs)
        return docs

    # return this inner function as the retriever
    return retrieve

