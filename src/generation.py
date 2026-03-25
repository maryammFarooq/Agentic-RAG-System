import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the client. 
# Make sure to set your OPENAI_API_KEY environment variable before running.
# If you are using a local model (e.g., Ollama), you can pass base_url="http://localhost:11434/v1"
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1" 
)

def generate_answer(query: str, context: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Generates an answer using an LLM based strictly on the provided context.
    
    Args:
        query (str): The user's original question.
        context (str): The formatted string of retrieved chunks.
        model (str): The LLM model to use for generation.
        
    Returns:
        str: The generated answer.
    """
    # If no context is provided (e.g., if CRAG decides to skip retrieval), 
    # we adjust the system prompt accordingly.
    if not context.strip():
        system_prompt = (
            "You are a helpful, smart assistant. Answer the user's question "
            "to the best of your general knowledge."
        )
        user_message = f"Question: {query}"
    else:
        # Crucial for CRAG and general RAG: Instruct the model to cite its sources!
        system_prompt = (
            "You are an expert factual assistant. Answer the user's question strictly "
            "using the provided context. If the answer is not contained in the context, "
            "say 'I do not have enough information to answer that.'\n\n"
            "IMPORTANT: You must include the source of the information used from the "
            "retrieved context and cite it at the end of the relevant sentences or at "
            "the end of your answer (e.g., [Source 1])."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2, # Low temperature for more grounded, factual answers
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while generating the answer."