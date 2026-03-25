"""
CRAG (Corrective RAG): assess retrieval confidence; use or correct retrieval based on it, then generate.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# Retrieve from global index -> assess confidence (e.g. NLI, consistency, or LLM judge) ->
# if high: use retrieved chunks for generation; if low: skip retrieval or use fallback -> generate answer.
"""
CRAG (Corrective RAG): assess retrieval confidence; use or correct retrieval based on it, then generate.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# Retrieve from global index -> assess confidence (e.g. NLI, consistency, or LLM judge) ->
# if high: use retrieved chunks for generation; if low: skip retrieval or use fallback -> generate answer.
"""
CRAG (Corrective RAG) pipeline implementation.
"""
from typing import Tuple
from src.retrieval import retrieve_chunks, format_retrieved_context
from src.generation import generate_answer, client

def evaluate_retrieval_confidence(query: str, context: str) -> float:
    """
    Uses an LLM judge to evaluate how relevant the retrieved context is to the query.
    Returns a confidence score between 0.0 and 1.0.
    """
    prompt = (
        "You are an expert grading system. Your task is to evaluate the relevance of "
        "the provided retrieved context to the user's question. \n"
        "If the context contains the exact answer or highly relevant information, score it a 10.\n"
        "If the context is completely irrelevant, ad copy, or off-topic, score it a 0.\n"
        "Respond strictly with a single integer between 0 and 10. Do not include any other text.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved Context:\n{context}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Zero temperature for strict, deterministic grading
            max_tokens=5
        )
        score_text = response.choices[0].message.content.strip()
        # Parse the integer and normalize it to a 0.0 - 1.0 float
        score = float(score_text) / 10.0
        return score
    except Exception as e:
        print(f"Error during confidence evaluation: {e}")
        # If the judge fails, default to a middle-ground score
        return 0.5

def run_crag(query: str, index, top_k: int = 5, confidence_threshold: float = 0.6) -> Tuple[str, str, float]:
    """
    Executes the full Corrective RAG (CRAG) pipeline.
    
    Args:
        query (str): The user's question.
        index: The loaded GlobalIndex object.
        top_k (int): The number of chunks to retrieve initially.
        confidence_threshold (float): The score (0.0 to 1.0) required to use the context.
        
    Returns:
        Tuple containing: (generated_answer, retrieved_context_string, confidence_score)
    """
    # 1. Embed query and retrieve from the global index as usual
    retrieved_results = retrieve_chunks(query, index, top_k=top_k)
    
    # Format the context so the judge can read it
    temp_context_str = format_retrieved_context(retrieved_results)
    
    # 2. Assess retrieval confidence (using our LLM judge)
    confidence_score = evaluate_retrieval_confidence(query, temp_context_str)
    
    # 3. Apply corrective logic based on the confidence score
    if confidence_score >= confidence_threshold:
        # Confidence is high: use the retrieved chunks for generation
        final_context = temp_context_str
    else:
        # Confidence is low: fallback strategy. 
        # The assignment suggests either skipping retrieval or web-style expansion.
        # We will skip retrieval and generate from the query alone to avoid hallucinations.
        final_context = "" 
        
    # 4. Generate the final answer. 
    # (Our src/generation.py already handles empty contexts correctly by adjusting its prompt)
    final_answer = generate_answer(query, final_context)
    
    return final_answer, final_context, confidence_score