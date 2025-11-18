from sentence_transformers import CrossEncoder

# Load a cross-encoder model
cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-base")

def rerank(query, docs, top_k=5):
    """
    Re-rank retrieved documents using a cross-encoder.
    """
    if not docs:
        print("RERANK: No documents passed in.")
        return []

    print(f"RERANK: Received {len(docs)} docs")

    # Pair each doc with the query
    try:
        pairs = [(query, d.page_content) for d in docs]
    except Exception as e:
        print("RERANK ERROR: Invalid document type:", e)
        return docs  # fallback: return original docs

    # Try running the cross-encoder
    try:
        scores = cross_encoder.predict(pairs)
    except Exception as e:
        print("CrossEncoder FAILED:", e)
        print("Falling back to original ranking...")
        return docs  # fallback if model breaks

    if not isinstance(scores, list) or len(scores) == 0:
        print("CrossEncoder returned EMPTY scores. Falling back.")
        return docs

    # Attach scores
    for doc, score in zip(docs, scores):
        doc.relevance_score = float(score)

    ranked = sorted(docs, key=lambda x: x.relevance_score, reverse=True)

    print(f"RERANK: Returning top {min(top_k, len(ranked))}")
    return ranked[:top_k]
