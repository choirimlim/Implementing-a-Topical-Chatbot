"""
Embeddings module for the DocuChat application.
Handles document vectorization and vector store management.
"""

import os
import logging
import yaml
import json
import time
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple, Union
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentEmbeddings:
    """
    Manages document embeddings and vector storage for efficient retrieval.
    Uses Sentence Transformers for embedding generation and FAISS for storage.
    """
    
    def __init__(self, embedding_model=None, config_path: str = "config/config.yaml"):
        """
        Initialize the embeddings manager.
        
        Args:
            embedding_model: Pre-loaded embedding model (optional)
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embedding_dimensions = self.config['documents']['embedding']['dimensions']
        self.normalize_embeddings = self.config['documents']['embedding']['normalize']
        self.vector_store_path = self.config['documents']['embedding']['vector_store_path']
        
        # Create the vector store directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Store the embedding model
        self.embedding_model = embedding_model
        
        # Initialize FAISS index
        self.index = self._initialize_faiss_index()
        
        # Initialize document mapping (index to document ID)
        self.doc_mapping = {}
        self.chunk_mapping = {}
        
        # Load existing vector store if available
        self._load_vector_store()
        
        logger.info(f"Initialized DocumentEmbeddings with dimension {self.embedding_dimensions}")
    
    def _initialize_faiss_index(self) -> faiss.Index:
        """
        Initialize a FAISS index for vector storage.
        
        Returns:
            FAISS index instance
        """
        if self.normalize_embeddings:
            # For normalized vectors, use cosine similarity
            index = faiss.IndexFlatIP(self.embedding_dimensions)  # Inner product = cosine similarity for normalized vectors
        else:
            # For non-normalized vectors, use L2 distance
            index = faiss.IndexFlatL2(self.embedding_dimensions)
        
        return index
    
    def _load_vector_store(self) -> None:
        """
        Load existing vector store from disk if available.
        """
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        mapping_path = os.path.join(self.vector_store_path, "doc_mapping.pkl")
        chunk_mapping_path = os.path.join(self.vector_store_path, "chunk_mapping.pkl")
        
        # Check if vector store exists
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load document mapping
                with open(mapping_path, 'rb') as f:
                    self.doc_mapping = pickle.load(f)
                
                # Load chunk mapping if available
                if os.path.exists(chunk_mapping_path):
                    with open(chunk_mapping_path, 'rb') as f:
                        self.chunk_mapping = pickle.load(f)
                
                logger.info(f"Loaded existing vector store with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                # Re-initialize index if loading fails
                self.index = self._initialize_faiss_index()
                self.doc_mapping = {}
                self.chunk_mapping = {}
    
    def _save_vector_store(self) -> None:
        """
        Save the current vector store to disk.
        """
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        mapping_path = os.path.join(self.vector_store_path, "doc_mapping.pkl")
        chunk_mapping_path = os.path.join(self.vector_store_path, "chunk_mapping.pkl")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save document mapping
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.doc_mapping, f)
            
            # Save chunk mapping
            with open(chunk_mapping_path, 'wb') as f:
                pickle.dump(self.chunk_mapping, f)
            
            logger.info(f"Saved vector store with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Please provide a model during initialization.")
        
        # Generate embedding
        embedding = self.embedding_model.encode(text)
        
        # Normalize if configured
        if self.normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def add_document_chunks(self, doc_id: str, chunks: List[str], metadata: Dict[str, Any]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            doc_id: Unique identifier for the document
            chunks: List of document chunks to embed
            metadata: Document metadata
        """
        if not chunks:
            logger.warning(f"No chunks to add for document {doc_id}")
            return
        
        # Generate embeddings for each chunk
        embeddings = []
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
            
            # Map the index to the document and chunk
            idx = self.index.ntotal + i
            self.doc_mapping[idx] = doc_id
            self.chunk_mapping[idx] = {
                "doc_id": doc_id,
                "chunk_idx": i,
                "chunk_text": chunk,
                "metadata": metadata
            }
        
        # Convert to numpy array
        embeddings_array = np.vstack(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        logger.info(f"Added {len(chunks)} chunks for document {doc_id} to vector store")
        
        # Save the updated vector store
        self._save_vector_store()
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and its chunks from the vector store.
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if document was found and removed, False otherwise
        """
        # Find indices to remove
        indices_to_remove = [idx for idx, did in self.doc_mapping.items() if did == doc_id]
        
        if not indices_to_remove:
            logger.warning(f"Document {doc_id} not found in vector store")
            return False
        
        # FAISS doesn't support direct removal, so we need to rebuild the index
        # Get all vectors that we want to keep
        retained_vectors = []
        retained_mapping = {}
        retained_chunk_mapping = {}
        
        for i in range(self.index.ntotal):
            if i not in indices_to_remove:
                # Get the vector
                vector = np.array([self.index.reconstruct(i)])
                retained_vectors.append(vector)
                
                # Update mappings with new indices
                new_idx = len(retained_vectors) - 1
                retained_mapping[new_idx] = self.doc_mapping[i]
                retained_chunk_mapping[new_idx] = self.chunk_mapping[i]
        
        # Reinitialize the index
        self.index = self._initialize_faiss_index()
        
        # Add retained vectors back to the index
        if retained_vectors:
            retained_vectors_array = np.vstack(retained_vectors).astype('float32')
            self.index.add(retained_vectors_array)
        
        # Update mappings
        self.doc_mapping = retained_mapping
        self.chunk_mapping = retained_chunk_mapping
        
        logger.info(f"Removed {len(indices_to_remove)} chunks for document {doc_id} from vector store")
        
        # Save the updated vector store
        self._save_vector_store()
        
        return True
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for document chunks similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of dictionaries containing search results with metadata
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty, no results to return")
            return []
        
        # Generate embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Reshape for FAISS
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            # Skip results below threshold (note: for IP, higher is better; for L2, lower is better)
            if self.normalize_embeddings:  # Using inner product (cosine similarity)
                similarity = distances[0][i]
                if similarity < threshold:
                    continue
            else:  # Using L2 distance
                # Convert L2 distance to similarity score (0 to 1)
                similarity = 1.0 / (1.0 + distances[0][i])
                if similarity < threshold:
                    continue
            
            # Get chunk info
            chunk_info = self.chunk_mapping.get(int(idx), {})
            
            # Add to results
            results.append({
                "doc_id": chunk_info.get("doc_id", "unknown"),
                "chunk_idx": chunk_info.get("chunk_idx", 0),
                "text": chunk_info.get("chunk_text", ""),
                "metadata": chunk_info.get("metadata", {}),
                "similarity": float(similarity)
            })
        
        return results
