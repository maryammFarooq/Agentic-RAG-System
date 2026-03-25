"""
Graph RAG: graph-augmented retrieval over the corpus (e.g. entity/relation graph or similarity graph), then generate.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# Build or use a graph over the corpus (entities, relations, or chunk similarity) ->
# retrieve in a graph-aware way (e.g. entity linking, subgraph, spread from seeds) ->
# convert selected graph neighborhood to text -> generate answer from that context.
"""
Graph RAG: graph-augmented retrieval over the corpus (e.g. entity/relation graph or similarity graph), then generate.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# Build or use a graph over the corpus (entities, relations, or chunk similarity) ->
# retrieve in a graph-aware way (e.g. entity linking, subgraph, spread from seeds) ->
# convert selected graph neighborhood to text -> generate answer from that context.

"""
Graph RAG pipeline implementation using a chunk-similarity graph.
"""
from typing import Tuple, List
from src.retrieval import retrieve_chunks, format_retrieved_context
from src.generation import generate_answer

def get_graph_neighborhood(query: str, index, seed_k: int = 3, hop_k: int = 2) -> List[Tuple[str, float]]:
    """
    Retrieves a graph neighborhood using vector similarity to define edges.
    1. Finds seed nodes (chunks) for the query.
    2. Spreads from seed nodes to find related neighbor nodes in the corpus.
    """
    # Step 1: Retrieve seed nodes (first hop based on the original query)
    seed_chunks = retrieve_chunks(query, index, top_k=seed_k)
    
    # Store the neighborhood in a dictionary to prevent duplicate nodes
    neighborhood = {chunk: score for chunk, score in seed_chunks}
    
    # Step 2: Expand to neighbors (second hop / spreading)
    for seed_text, seed_score in seed_chunks:
        # We query the index using the text of the seed chunk to find its "edges"
        # We fetch hop_k + 1 to account for the seed chunk retrieving itself
        neighbors = retrieve_chunks(seed_text, index, top_k=hop_k + 1)
        
        for neighbor_text, neighbor_score in neighbors:
            if neighbor_text not in neighborhood:
                # Calculate an edge weight/score penalty for the neighbor. 
                # A neighbor's relevance is bounded by how relevant the seed was.
                neighborhood[neighbor_text] = seed_score * neighbor_score

    # Sort the graph neighborhood by the combined confidence score in descending order
    sorted_neighborhood = sorted(neighborhood.items(), key=lambda x: x[1], reverse=True)
    return sorted_neighborhood

def run_graph_rag(query: str, index, top_k: int = 5) -> Tuple[str, str, float]:
    """
    Executes the Graph RAG pipeline using similarity-based node spreading.
    
    Args:
        query (str): The original user question.
        index: The loaded GlobalIndex object.
        top_k (int): The final number of nodes (chunks) to pass to the LLM.
        
    Returns:
        Tuple containing: (generated_answer, retrieved_context_string, top_retrieval_score)
    """
    # 1. Retrieve graph neighborhood (seed nodes + their neighbors)
    neighborhood_chunks = get_graph_neighborhood(query, index, seed_k=3, hop_k=2)
    
    # Trim the subgraph to the top_k nodes so we don't blow up the LLM token limit
    final_chunks = neighborhood_chunks[:top_k]
    
    # 2. Turn the selected graph neighborhood into text
    context_str = format_retrieved_context(final_chunks)
    
    # Extract the top score to display in the UI/Evaluation later
    top_score = final_chunks[0][1] if final_chunks else 0.0
    
    # 3. Pass it to the LLM for answer generation
    final_answer = generate_answer(query, context_str)
    
    return final_answer, context_str, top_score