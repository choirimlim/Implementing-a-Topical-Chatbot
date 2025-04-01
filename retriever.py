"""
Retriever module for the DocuChat application.
Handles retrieving relevant context from document store for queries.
"""

import os
import logging
import yaml
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Retrieves relevant document chunks for user queries.
    Supports basic and advanced retrieval strategies.
    """
    
    def __init__(self, embeddings_manager, config_path: str = "config/config.yaml"):
        """
        Initialize the retriever with configuration.
        
        Args:
            embeddings_manager: DocumentEmbeddings instance
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.embeddings = embeddings_manager
        self.top_k = self.config['retrieval']['top_k']
        self.similarity_threshold = self.config['retrieval']['similarity_threshold']
        self.reranker_enabled = self.config['retrieval']['reranker_enabled']
        
        # Initialize reranker if enabled
        self.reranker = None
        if self.reranker_enabled:
            self._initialize_reranker()
        
        logger.info(f"Initialized DocumentRetriever with top_k={self.top_k}, threshold={self.similarity_threshold}")
    
    def _initialize_reranker(self) -> None:
        """
        Initialize a cross-encoder reranker for improved retrieval precision.
        """
        try:
            from sentence_transformers import CrossEncoder
            
            reranker_model = self.config['retrieval']['reranker_model']
            self.reranker = CrossEncoder(reranker_model)
            
            logger.info(f"Initialized reranker: {reranker_model}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {str(e)}")
            self.reranker_enabled = False
    
    def retrieve(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for the query.
        
        Args:
            query: User query text
            top_k: Number of results to retrieve (overrides config)
            threshold: Similarity threshold (overrides config)
            
        Returns:
            List of dictionaries containing retrieval results with metadata
        """
        # Use provided values or fall back to config defaults
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        # Initial retrieval using vector similarity
        results = self.embeddings.search(query, top_k=top_k * 2 if self.reranker_enabled else top_k, threshold=threshold)
        
        if not results:
            logger.info(f"No results found for query: {query}")
            return []
        
        # Apply reranking if enabled
        if self.reranker_enabled and self.reranker and len(results) > 1:
            return self._rerank_results(query, results, top_k)
        
        logger.info(f"Retrieved {len(results)} results for query: {query}")
        return results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using a cross-encoder model.
        
        Args:
            query: Original query text
            results: Initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of results
        """
        # Prepare pairs for reranking
        pairs = [(query, result["text"]) for result in results]
        
        # Compute reranker scores
        reranker_scores = self.reranker.predict(pairs)
        
        # Add scores to results
        for i, score in enumerate(reranker_scores):
            results[i]["reranker_score"] = float(score)
        
        # Sort by reranker score (descending)
        reranked_results = sorted(results, key=lambda x: x["reranker_score"], reverse=True)
        
        logger.info(f"Reranked {len(results)} results for query: {query}")
        return reranked_results[:top_k]
    
    def format_context(self, results: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
        """
        Format retrieval results into a context string for the LLM.
        
        Args:
            results: Retrieval results from retrieve()
            max_tokens: Approximate maximum tokens to include
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        # Sort results by document and chunk index to maintain original document order
        sorted_results = sorted(results, key=lambda x: (x["doc_id"], x["chunk_idx"]))
        
        # Format context with metadata and text
        context_parts = []
        total_chars = 0
        
        for result in sorted_results:
            # Estimate tokens (rough approximation)
            text_chars = len(result["text"])
            if total_chars + text_chars > max_tokens * 4:  # Rough char-to-token ratio
                break
            
            # Get document metadata
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "Unknown document")
            
            # Format context piece
            context_part = f"\n[Document: {filename}]\n{result['text']}\n"
            context_parts.append(context_part)
            
            total_chars += text_chars
        
        # Combine all parts into a single context string
        full_context = "\n".join(context_parts)
        
        return full_context
    
    def retrieve_and_format(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None, max_tokens: int = 3000) -> str:
        """
        Combine retrieval and formatting into a single operation.
        
        Args:
            query: User query text
            top_k: Number of results to retrieve
            threshold: Similarity threshold
            max_tokens: Approximate maximum tokens to include
            
        Returns:
            Formatted context string
        """
        # Retrieve relevant documents
        results = self.retrieve(query, top_k=top_k, threshold=threshold)
        
        # Format context
        context = self.format_context(results, max_tokens=max_tokens)
        
        return context
