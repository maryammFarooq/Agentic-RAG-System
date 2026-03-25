"""
RAG Fusion: multiple queries, retrieve from global index for each, merge ranked lists (e.g. RRF), generate answer.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (fused_ranked_passages, answer).
# Multiple queries -> retrieve ranked lists from global index -> fuse with RRF -> take top-k -> generate.
"""
RAG Fusion pipeline implementation.
"""
from typing import List, Tuple
from src.retrieval import retrieve_chunks, format_retrieved_context
from src.generation import generate_answer, client  # Reusing the initialized OpenAI client

def generate_query_variants(query: str, num_variants: int = 3) -> List[str]:
    """Uses the LLM to generate variations of the original user query."""
    prompt = (
        f"You are an expert search assistant. Your task is to generate {num_variants} "
        f"different search queries that explore the same underlying intent as the "
        f"following original query. Return ONLY the queries, one per line, without "
        f"numbering, bullet points, or introductory text.\n\nOriginal query: {query}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5, # Slightly higher temperature for creative variations
        )
        variants = response.choices[0].message.content.strip().split('\n')
        # Clean up any accidental numbering or bullet points the LLM might add
        clean_variants = [v.strip('- ').strip('1234567890. ') for v in variants if v.strip()]
        return clean_variants[:num_variants]
    except Exception as e:
        print(f"Error generating variants: {e}")
        return [] # Fallback to an empty list so the pipeline doesn't crash

def reciprocal_rank_fusion(results_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Merges multiple ranked lists of retrieved chunks using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    
    for results in results_lists:
        # results is a list of (chunk_text, similarity_score)
        for rank, (chunk, _) in enumerate(results):
            if chunk not in rrf_scores:
                rrf_scores[chunk] = 0.0
            
            # Rank is 0-indexed in our loop, so we add 1 for the mathematical formula
            rrf_scores[chunk] += 1.0 / (k + rank + 1)
            
    # Sort chunks by their new RRF score in descending order
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks

def run_rag_fusion(query: str, index, top_k: int = 5) -> Tuple[str, str, float]:
    """
    Executes the full RAG Fusion pipeline.
    
    Args:
        query (str): The original user question.
        index: The loaded GlobalIndex object.
        top_k (int): The final number of fused chunks to send to the LLM.
        
    Returns:
        Tuple containing: (generated_answer, retrieved_context_string, top_retrieval_score)
    """
    # 1. Generate query variants 
    variants = generate_query_variants(query)
    all_queries = [query] + variants
    
    # 2. Retrieve ranked lists for each query variant 
    all_results = []
    for q in all_queries:
        # Fetch slightly more per variant to ensure a good pool of chunks for fusion
        res = retrieve_chunks(q, index, top_k=top_k * 2) 
        all_results.append(res)
        
    # 3. Merge the ranked lists using Reciprocal Rank Fusion (RRF) 
    fused_results = reciprocal_rank_fusion(all_results)
    
    # 4. Take top-k from the fused list 
    final_top_k = fused_results[:top_k]
    
    # Format the chunks into a readable prompt context
    context_str = format_retrieved_context(final_top_k)
    
    # Extract the top score to display in the UI/Evaluation later
    top_score = final_top_k[0][1] if final_top_k else 0.0
    
    # 5. LLM generates answer from fused context 
    answer = generate_answer(query, context_str)
    
    return answer, context_str, top_score