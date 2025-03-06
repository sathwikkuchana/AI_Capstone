# src/indexing.py

import faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(chunks, embedding_model):
    """
    Compute embeddings for each chunk using the provided embedding model,
    and build a FAISS index for quick similarity search.
    
    Args:
        chunks (List[str]): List of text chunks.
        embedding_model: A SentenceTransformer model instance.
    
    Returns:
        index: The FAISS index.
        chunk_embeddings: The computed embeddings.
    """
    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(chunk_embeddings)
    return index, chunk_embeddings

def retrieve_chunks(query, embedding_model, index, chunks, top_k=3):
    """
    Given a query, compute its embedding, search the FAISS index,
    and return the top_k most similar chunks.
    
    Args:
        query (str): The query text.
        embedding_model: A SentenceTransformer model instance.
        index: The FAISS index.
        chunks (List[str]): List of text chunks.
        top_k (int): Number of top matches to retrieve.
    
    Returns:
        List[str]: Retrieved chunks.
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks
