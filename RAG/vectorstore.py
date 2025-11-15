from RAG.embedding import load_vector_store

def get_retriever(k=3):
    vs = load_vector_store()
    retriever = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.00, "k": 3}
    )
    return retriever
