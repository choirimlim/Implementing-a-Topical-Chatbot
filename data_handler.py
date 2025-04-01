"""
Data handler module for the DocuChat application.
Manages datasets for training and evaluation.
"""

import os
import logging
import yaml
import json
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles data preparation for RL training.
    Manages datasets, prompts, and example formatting.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the data handler.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = os.path.join("data", "training")
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Initialized DataHandler")
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load a dataset from a JSON file.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            List of examples (dictionaries)
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded dataset with {len(dataset)} examples from {dataset_path}")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], dataset_path: str) -> None:
        """
        Save a dataset to a JSON file.
        
        Args:
            dataset: Dataset to save
            dataset_path: Path to save the dataset
        """
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset with {len(dataset)} examples to {dataset_path}")
    
    def create_training_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Create a training prompt from a question and optional context.
        
        Args:
            question: User question
            context: Retrieved document context (optional)
            
        Returns:
            Formatted prompt string
        """
        # Base system prompt
        system_prompt = "You are a helpful assistant that provides accurate and detailed answers based on the provided documents. Please use <think>...</think> tags to show your reasoning process."
        
        # Structure the prompt
        prompt_parts = [
            f"<s>[INST] {system_prompt}\n\n"
        ]
        
        # Add context if provided
        if context:
            prompt_parts.append(f"Here are relevant documents to help answer the question:\n\n{context}\n\n")
        
        # Add the question
        prompt_parts.append(f"Question: {question} [/INST]\n\n")
        
        return "".join(prompt_parts)
    
    def prepare_document_training_data(self, documents_path: str, questions_path: str, split: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Prepare training and evaluation datasets from documents and questions.
        
        Args:
            documents_path: Path to processed documents directory
            questions_path: Path to questions JSON file
            split: Train/eval split ratio
            
        Returns:
            Tuple of (training_data, evaluation_data)
        """
        # Load questions
        questions = self.load_dataset(questions_path)
        
        # Load document retriever
        from docuchat.core.retriever import DocumentRetriever
        from docuchat.core.embeddings import DocumentEmbeddings
        from docuchat.core.model_loader import ModelLoader
        
        # Initialize components
        model_loader = ModelLoader()
        embedding_model = model_loader.load_embedding_model()
        embeddings = DocumentEmbeddings(embedding_model)
        retriever = DocumentRetriever(embeddings)
        
        # Prepare datasets
        dataset = []
        
        for question_item in questions:
            question = question_item["question"]
            answer = question_item.get("answer")
            
            # Retrieve relevant context
            context = retriever.retrieve_and_format(question)
            
            # Create prompt
            prompt = self.create_training_prompt(question, context)
            
            # Add to dataset
            dataset.append({
                "prompt": prompt,
                "question": question,
                "context": context,
                "answer": answer
            })
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        # Split into train and eval
        split_idx = int(len(dataset) * split)
        train_data = dataset[:split_idx]
        eval_data = dataset[split_idx:]
        
        # Save datasets
        train_path = os.path.join(self.data_dir, "train_data.json")
        eval_path = os.path.join(self.data_dir, "eval_data.json")
        
        self.save_dataset(train_data, train_path)
        self.save_dataset(eval_data, eval_path)
        
        logger.info(f"Prepared {len(train_data)} training examples and {len(eval_data)} evaluation examples")
        
        return train_data, eval_data
    
    def create_synthetic_qa_dataset(self, documents_path: str, model, tokenizer, num_examples: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic question-answer pairs from documents.
        
        Args:
            documents_path: Path to processed documents directory
            model: Language model for generation
            tokenizer: Tokenizer for the model
            num_examples: Number of examples to generate
            
        Returns:
            List of generated QA pairs
        """
        # This is an advanced feature that would generate synthetic questions from documents
        # For simplicity, we'll just create a stub implementation
        
        logger.info(f"Creating synthetic QA dataset with {num_examples} examples - this is a placeholder")
        
        # Load document chunks
        from docuchat.core.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        # Get all document files
        doc_files = []
        for root, _, files in os.walk(documents_path):
            for file in files:
                if file.lower().endswith(tuple(processor.supported_file_types)):
                    doc_files.append(os.path.join(root, file))
        
        # Select a subset of documents
        if len(doc_files) > num_examples:
            doc_files = random.sample(doc_files, num_examples)
        
        synthetic_dataset = []
        
        for file_path in doc_files:
            try:
                # Load the document
                doc_text = processor.load_document(file_path)
                
                # Create a prompt for question generation
                prompt = f"""
                Generate a question and answer pair based on the following text:
                
                {doc_text[:2000]}  # Limit to first 2000 chars for simplicity
                
                Format your response as:
                Question: [your question here]
                Answer: [your answer here]
                """
                
                # Generate question-answer pair
                # In a real implementation, we would use the model to generate QA pairs
                # For now, just create placeholder data
                
                question = f"What is the main topic of the document '{os.path.basename(file_path)}'?"
                answer = "The document discusses various aspects of the main subject."
                
                synthetic_dataset.append({
                    "question": question,
                    "answer": answer,
                    "source_document": file_path
                })
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        # Save the synthetic dataset
        output_path = os.path.join(self.data_dir, "synthetic_qa.json")
        self.save_dataset(synthetic_dataset, output_path)
        
        return synthetic_dataset
    
    def augment_with_user_feedback(self, base_dataset: List[Dict[str, Any]], feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Augment training data with user feedback.
        
        Args:
            base_dataset: Original training dataset
            feedback_data: User feedback data
            
        Returns:
            Augmented dataset
        """
        # Merge datasets, prioritizing user feedback
        augmented_dataset = base_dataset.copy()
        
        # Create a map of existing questions
        question_map = {item["question"]: i for i, item in enumerate(augmented_dataset)}
        
        # Add or update with feedback data
        for feedback_item in feedback_data:
            question = feedback_item["question"]
            
            if question in question_map:
                # Update existing item
                idx = question_map[question]
                augmented_dataset[idx].update(feedback_item)
            else:
                # Add new item
                augmented_dataset.append(feedback_item)
        
        # Save augmented dataset
        output_path = os.path.join(self.data_dir, f"augmented_data_{int(time.time())}.json")
        self.save_dataset(augmented_dataset, output_path)
        
        logger.info(f"Augmented dataset with {len(feedback_data)} feedback items, total size: {len(augmented_dataset)}")
        
        return augmented_dataset
