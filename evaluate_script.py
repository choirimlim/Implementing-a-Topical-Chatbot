#!/usr/bin/env python
"""
Evaluation script for DocuChat model.
Evaluates model performance on a test set and generates metrics.
"""

import os
import sys
import logging
import argparse
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from docuchat.core.model_loader import ModelLoader
from docuchat.core.reward_model import RewardModel
from docuchat.training.data_handler import DataHandler
from docuchat.core.retriever import DocumentRetriever
from docuchat.core.embeddings import DocumentEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DocuChat model")
    
    parser.add_argument("--model_name", type=str, help="Model name or checkpoint path to evaluate")
    parser.add_argument("--test_set", type=str, help="Path to test dataset")
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to evaluate")
    
    return parser.parse_args()

def calculate_metrics(predictions, references, reward_model):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of model predictions
        references: List of reference answers
        reward_model: Reward model instance
        
    Returns:
        Dictionary of metrics
    """
    # Initialize metrics
    metrics = {
        "accuracy": 0.0,
        "format_compliance": 0.0,
        "reward_score": 0.0,
    }
    
    if not predictions:
        return metrics
    
    # Calculate metrics
    accuracies = []
    format_scores = []
    reward_scores = []
    
    for pred, ref in zip(predictions, references):
        # Accuracy (1 if correct, 0 if incorrect)
        accuracy = reward_model.calculate_accuracy_reward(pred, ref)
        accuracies.append(accuracy)
        
        # Format compliance
        format_score = reward_model.calculate_format_reward(pred)
        format_scores.append(format_score)
        
        # Combined reward
        reward_score = reward_model.calculate_combined_reward(pred, ref)
        reward_scores.append(reward_score)
    
    # Compute averages
    metrics["accuracy"] = np.mean(accuracies)
    metrics["format_compliance"] = np.mean(format_scores)
    metrics["reward_score"] = np.mean(reward_scores)
    
    return metrics

def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.model_name:
        config['model']['base_model_name'] = args.model_name
    
    # Set up output file
    output_file = args.output_file or f"evaluation_results_{int(datetime.now().timestamp())}.json"
    
    # Initialize components
    model_loader = ModelLoader(args.config_path)
    reward_model = RewardModel(args.config_path)
    data_handler = DataHandler(args.config_path)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model']['base_model_name']}")
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    # Load test dataset
    test_set_path = args.test_set or os.path.join("data", "training", "eval_data.json")
    
    if not os.path.exists(test_set_path):
        logger.error(f"Test set file not found: {test_set_path}")
        sys.exit(1)
    
    with open(test_set_path, 'r') as f:
        test_data = json.load(f)
    
    # Limit number of examples if specified
    if args.max_examples and args.max_examples < len(test_data):
        test_data = test_data[:args.max_examples]
    
    logger.info(f"Evaluating on {len(test_data)} examples")
    
    # Set up retriever for context
    embedding_model = model_loader.load_embedding_model()
    embeddings = DocumentEmbeddings(embedding_model)
    retriever = DocumentRetriever(embeddings)
    
    # Run evaluation
    predictions = []
    references = []
    
    for example in tqdm(test_data, desc="Evaluating"):
        try:
            # Get query and expected answer
            query = example.get("question")
            expected_answer = example.get("answer")
            
            if not query:
                continue
            
            # Retrieve context if not already provided
            context = example.get("context")
            if not context:
                context = retriever.retrieve_and_format(query)
            
            # Create prompt
            prompt = data_handler.create_training_prompt(query, context)
            
            # Generate response
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                model = model.cuda()
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=config['model']['max_new_tokens'],
                    temperature=config['model']['temperature'],
                    top_p=config['model']['top_p'],
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode output (skip input prompt tokens)
            response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
            
            # Add to results
            predictions.append(response)
            references.append(expected_answer)
        
        except Exception as e:
            logger.error(f"Error processing example: {str(e)}")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references, reward_model)
    
    # Add additional info
    metrics["model_name"] = config['model']['base_model_name']
    metrics["test_set"] = test_set_path
    metrics["num_examples"] = len(predictions)
    metrics["timestamp"] = datetime.now().isoformat()
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump({
            "metrics": metrics,
            "predictions": [{"prediction": p, "reference": r} for p, r in zip(predictions, references)]
        }, f, indent=2)
    
    # Print summary
    logger.info(f"Evaluation completed. Results saved to {output_file}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Format compliance: {metrics['format_compliance']:.4f}")
    logger.info(f"Reward score: {metrics['reward_score']:.4f}")

if __name__ == "__main__":
    main()
