"""
HyDE: generate hypothetical document, retrieve from global index by similarity to it, generate final answer.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (hypothetical_doc, retrieved_passages, answer).
# Generate hypothetical passage with LLM -> embed it -> index.retrieve(hypothetical_embedding, top_k) -> generate from retrieved chunks.
"""
HyDE: generate hypothetical document, retrieve from global index by similarity to it, generate final answer.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (hypothetical_doc, retrieved_passages, answer).
# Generate hypothetical passage with LLM -> embed it -> index.retrieve(hypothetical_embedding, top_k) -> generate from retrieved chunks.
"""
HyDE (Hypothetical Document Embeddings) pipeline implementation.
"""
from typing import Tuple
from src.retrieval import retrieve_chunks, format_retrieved_context
from src.generation import generate_answer, client 

def generate_hypothetical_document(query: str) -> str:
    """
    Prompts the LLM to generate a hypothetical 1-2 paragraph document 
    that might answer the user's query.
    """
    prompt = (
        "You are an expert content writer. Please write a 1-2 paragraph hypothetical "
        "document or article snippet that directly answers the following question. "
        "Write in a factual, informative tone as if this is a real web page. "
        "Do not include introductory filler or acknowledge the prompt.\n\n"
        f"Question: {query}"
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Keep it relatively grounded but allow some creative expansion
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating hypothetical document: {e}")
        # Fallback to the original query if generation fails so the pipeline can still proceed
        return query

def run_hyde(query: str, index, top_k: int = 5) -> Tuple[str, str, float]:
    """
    Executes the full HyDE pipeline.
    
    Args:
        query (str): The original user question.
        index: The loaded GlobalIndex object.
        top_k (int): The number of chunks to retrieve based on the hypothetical document.
        
    Returns:
        Tuple containing: (generated_answer, retrieved_context_string, top_retrieval_score)
    """
    # 1. LLM generates a hypothetical document that might contain the answer
    hypothetical_doc = generate_hypothetical_document(query)
    
    # 2 & 3. Embed the hypothetical doc (not the query) and retrieve top-k from the global index
    # Our retrieve_chunks function handles the embedding of whatever string is passed in
    retrieved_results = retrieve_chunks(hypothetical_doc, index, top_k=top_k)
    
    # Format the chunks into a readable prompt context
    context_str = format_retrieved_context(retrieved_results)
    
    # Extract the top score to display in the UI/Evaluation later
    top_score = retrieved_results[0][1] if retrieved_results else 0.0
    
    # 4. LLM generates the final answer from retrieved context
    final_answer = generate_answer(query, context_str)
    
    return final_answer, context_str, top_score