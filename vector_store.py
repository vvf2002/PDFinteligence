import streamlit as st
from typing import List, Tuple
import numpy as np
import hashlib
import re

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class VectorStore:
    """Handles document vectorization and similarity search"""
    
    def __init__(self):
        self.embeddings = {}
        self.chunks = []
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        if not FAISS_AVAILABLE:
            st.warning("FAISS not available. Using simple text matching for search.")
    
    def create_vector_store(self, text_content: str):
        """
        Create a vector store from text content
        
        Args:
            text_content (str): The extracted text from PDF
            
        Returns:
            VectorStore: Self reference for method chaining
        """
        # Split text into chunks
        self.chunks = self._split_text_into_chunks(text_content)
        
        if not self.chunks:
            raise Exception("No text chunks created from the document")
        
        if FAISS_AVAILABLE:
            # Create simple embeddings using text hashing
            self._create_simple_embeddings()
        
        return self
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation
        
        Args:
            text (str): Input text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (sentence end, paragraph, etc.)
            break_point = end
            
            # Look for sentence endings within a reasonable range
            search_start = max(start + self.chunk_size - 200, start)
            search_end = min(end + 100, len(text))
            
            for i in range(search_end - 1, search_start - 1, -1):
                if text[i] in '.!?':
                    # Check if it's likely a sentence end
                    if i + 1 < len(text) and (text[i + 1].isspace() or text[i + 1].isupper()):
                        break_point = i + 1
                        break
            
            chunks.append(text[start:break_point].strip())
            
            # Calculate next start position with overlap
            start = break_point - self.chunk_overlap
            if start < 0:
                start = break_point
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        return chunks
    
    def _create_simple_embeddings(self):
        """Create simple hash-based embeddings for text similarity"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            # Create simple embeddings using character n-grams and word hashing
            embedding_dim = 384  # Standard embedding dimension
            embeddings_matrix = np.zeros((len(self.chunks), embedding_dim))
            
            for i, chunk in enumerate(self.chunks):
                embeddings_matrix[i] = self._text_to_vector(chunk, embedding_dim)
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            self.index.add(embeddings_matrix.astype('float32'))
            
        except Exception as e:
            st.warning(f"FAISS indexing failed, falling back to keyword search: {str(e)}")
            # Remove the index so we fall back to keyword search
            if hasattr(self, 'index'):
                delattr(self, 'index')
    
    def _text_to_vector(self, text: str, dim: int) -> np.ndarray:
        """
        Convert text to a simple vector representation
        
        Args:
            text (str): Input text
            dim (int): Vector dimension
            
        Returns:
            np.ndarray: Vector representation
        """
        # Simple hash-based embedding
        words = text.lower().split()
        vector = np.zeros(dim)
        
        # Use word hashing
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % dim
            vector[idx] += 1.0
        
        # Add character n-gram features
        for n in [2, 3]:
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n].lower()
                hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                idx = hash_val % dim
                vector[idx] += 0.5
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Find most similar chunks to the query
        
        Args:
            query (str): Search query
            k (int): Number of results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk, similarity_score) tuples
        """
        if not self.chunks:
            return []
        
        if FAISS_AVAILABLE and hasattr(self, 'index'):
            return self._faiss_search(query, k)
        else:
            return self._simple_keyword_search(query, k)
    
    def _faiss_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Search using FAISS vector similarity"""
        query_vector = self._text_to_vector(query, 384).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.index.search(query_vector.astype('float32'), min(k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def _simple_keyword_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Simple keyword-based search fallback"""
        query_words = set(query.lower().split())
        
        chunk_scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(chunk_words))
            union = len(query_words.union(chunk_words))
            
            if union > 0:
                similarity = intersection / union
                
                # Boost score if query appears as substring
                if query.lower() in chunk.lower():
                    similarity += 0.3
                
                chunk_scores.append((chunk, similarity))
        
        # Sort by similarity and return top k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:k]
    
    def get_all_chunks(self) -> List[str]:
        """Return all text chunks"""
        return self.chunks.copy()
    
    def get_chunk_count(self) -> int:
        """Return the number of chunks"""
        return len(self.chunks)
