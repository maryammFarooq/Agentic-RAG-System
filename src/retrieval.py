"""
Retrieval module to fetch top-k chunks from the global index.
"""

def retrieve_chunks(query: str, index, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Embeds the user query and retrieves the most relevant chunks from the global index.
    
    Args:
        query (str): The user's question.
        index (GlobalIndex): The loaded global corpus index object.
        top_k (int): The number of chunks to retrieve.
        
    Returns:
        List of tuples containing (chunk_text, similarity_score).
    """
    if not query.strip():
        return []

    # 1. Embed the query using the same SentenceTransformer model stored in the index [cite: 40]
    query_embedding = index.model.encode(query, convert_to_numpy=True)
    
    # 2. Search the index using the generated embedding to retrieve top-k chunks [cite: 39, 40]
    results = index.retrieve(query_embedding, top_k=top_k)
    
    return results

def format_retrieved_context(retrieved_results: list[tuple[str, float]]) -> str:
    """
    Helper function to format the retrieved chunks into a single string 
    that can be easily injected into an LLM prompt.
    """
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_results):
        # Including the score can be useful for debugging or strict confidence gating
        context_parts.append(f"[Source {i+1} | Score: {score:.4f}]:\n{chunk}")
        
    return "\n\n".join(context_parts)