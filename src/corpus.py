import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Import the specific functions from your data_loader
from src.data_loader import load_examples, get_passages_for_retrieval

class GlobalIndex:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initializes the embedding model and index storage."""
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def retrieve(self, query_embedding, top_k=5):
        """
        Retrieves the top-k most similar chunks given a query embedding.
        Returns a list of tuples: (chunk_text, score).
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Compute cosine similarities using numpy
        query_norm = np.linalg.norm(query_embedding)
        norms = np.linalg.norm(self.embeddings, axis=1)
        
        # Avoid division by zero
        valid_indices = (norms > 0) & (query_norm > 0)
        scores = np.zeros(self.embeddings.shape[0])
        
        if np.any(valid_indices):
            scores[valid_indices] = np.dot(self.embeddings[valid_indices], query_embedding) / (norms[valid_indices] * query_norm)

        # Get top_k indices sorted by highest score
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


def build_index(dataset_path, save_path=None, model_name='all-MiniLM-L6-v2'):
    """
    Builds the global corpus from the dataset, embeds all chunks, and creates the index.
    """
    print(f"Building index from {dataset_path}...")
    index = GlobalIndex(model_name)
    
    # 1 & 2. Load data and extract passages using your data_loader functions
    for example in load_examples(path=dataset_path):
        # We set use_snippet=True to grab 'page_snippet' as requested by the assignment
        passages = get_passages_for_retrieval(example, use_snippet=True)
        
        for snippet in passages:
            snippet = snippet.strip()
            if snippet:
                index.chunks.append(snippet)
                
    print(f"Extracted {len(index.chunks)} total chunks. Embedding now... (This might take a moment)")
                
    # 3. Embed all chunks 
    index.embeddings = index.model.encode(index.chunks, convert_to_numpy=True)
    
    # 4 & 5. Save the index to disk if a path is provided
    if save_path:
        print(f"Saving index to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'chunks': index.chunks, 
                'embeddings': index.embeddings
            }, f)
            
    return index


def load_index(index_path, model_name='all-MiniLM-L6-v2'):
    """
    Loads a previously saved index from disk.
    """
    print(f"Loading index from {index_path}...")
    index = GlobalIndex(model_name)
    
    with open(index_path, 'rb') as f:
        data = pickle.load(f)
        index.chunks = data['chunks']
        index.embeddings = data['embeddings']
        
    return index