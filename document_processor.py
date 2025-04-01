"""
Document processing module for the DocuChat application.
Handles document loading, cleaning, and chunking.
"""

import os
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import yaml
import re
import json

# Document loading libraries
import pypdf
import docx2txt

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Processes documents for ingestion into the retrieval system.
    Handles multiple file formats and chunks documents into manageable sizes.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the document processor with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.chunk_size = self.config['documents']['processor']['chunk_size']
        self.chunk_overlap = self.config['documents']['processor']['chunk_overlap']
        self.supported_file_types = self.config['documents']['processor']['supported_file_types']
        
        # Create data directories if they don't exist
        os.makedirs("data/documents", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        logger.info(f"Initialized DocumentProcessor with chunk size {self.chunk_size} and overlap {self.chunk_overlap}")
    
    def load_document(self, file_path: str) -> str:
        """
        Load document content from various file formats.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            String containing the document content
            
        Raises:
            ValueError: If the file format is not supported
        """
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension not in self.supported_file_types:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if file_extension == 'pdf':
            return self._load_pdf(file_path)
        elif file_extension == 'txt':
            return self._load_txt(file_path)
        elif file_extension == 'docx':
            return self._load_docx(file_path)
        elif file_extension == 'md':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_pdf(self, file_path: str) -> str:
        """Load and extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        return text
    
    def _load_txt(self, file_path: str) -> str:
        """Load text from a plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_docx(self, file_path: str) -> str:
        """Load and extract text from a DOCX file."""
        return docx2txt.process(file_path)
    
    def clean_text(self, text: str) -> str:
        """
        Clean the document text by removing extra whitespace and fixing common issues.
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text
        """
        # Remove excessive newlines and whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', text)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        # Fix broken sentences (where sentences are split across lines)
        cleaned = re.sub(r'(?<!\n)(\n)(?!\n)', ' ', cleaned)
        
        # Remove headers, footers, and page numbers (simplified approach)
        cleaned = re.sub(r'\n\d+\n', '\n', cleaned)
        
        return cleaned.strip()
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Split the document into overlapping chunks of specified size.
        
        Args:
            text: Document text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds the chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Calculate overlap - take last n characters
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    # Try to break at a paragraph boundary for overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Find the last paragraph break within the overlap region
                    last_para_break = overlap_text.rfind('\n\n')
                    
                    if last_para_break != -1:
                        # Start with overlap from last paragraph break
                        current_chunk = current_chunk[-(self.chunk_overlap - last_para_break):]
                    else:
                        # No paragraph break, just use character overlap
                        current_chunk = current_chunk[-self.chunk_overlap:]
                else:
                    current_chunk = ""
            
            # Add the paragraph to the current chunk
            if current_chunk and not current_chunk.endswith('\n\n'):
                current_chunk += '\n\n'
            current_chunk += paragraph
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_document(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process a document from start to finish: load, clean, and chunk.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple containing:
                - List of document chunks
                - Metadata dictionary with document information
        """
        logger.info(f"Processing document: {file_path}")
        
        # Extract filename for metadata
        filename = os.path.basename(file_path)
        file_extension = filename.split('.')[-1].lower()
        
        # Load and clean the document
        raw_text = self.load_document(file_path)
        cleaned_text = self.clean_text(raw_text)
        
        # Create document chunks
        chunks = self.chunk_document(cleaned_text)
        
        # Create metadata
        metadata = {
            "filename": filename,
            "file_type": file_extension,
            "chunk_count": len(chunks),
            "total_length": len(cleaned_text),
            "processed_date": None  # Will be filled by the calling function
        }
        
        logger.info(f"Document processed into {len(chunks)} chunks")
        
        return chunks, metadata
    
    def save_processed_document(self, doc_id: str, chunks: List[str], metadata: Dict[str, Any]) -> str:
        """
        Save processed document chunks and metadata to disk.
        
        Args:
            doc_id: Unique identifier for the document
            chunks: List of document chunks
            metadata: Document metadata
            
        Returns:
            Path to the saved document directory
        """
        doc_dir = f"data/processed/documents/{doc_id}"
        os.makedirs(doc_dir, exist_ok=True)
        
        # Save chunks
        with open(f"{doc_dir}/chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        with open(f"{doc_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed document to {doc_dir}")
        
        return doc_dir
