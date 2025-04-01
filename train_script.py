#!/usr/bin/env python
"""
Training script for DocuChat model.
Allows command-line training with various options.
"""

import os
import sys
import logging
import argparse
import yaml
import json
import torch
from datetime import datetime

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from docuchat.core.model_loader import ModelLoader
from docuchat.core.reward_model import RewardModel
from docuchat.training.data_handler import DataHandler
from docuchat.training.rl_trainer import GRPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train DocuChat model with RL")
    
    parser.add_argument("--model_name", type=str, help="Model name to train")
    parser.add_argument("--documents_path", type=str, help="Path to processed documents")
    parser.add_argument("--questions_path", type=str, help="Path to questions dataset")
    parser.add_argument("--output_dir", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--config_path", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--steps", type=int, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.model_name:
        config['model']['base_model_name'] = args.model_name
    
    if args.epochs:
        config['rl']['training']['num_train_epochs'] = args.epochs
    
    if args.steps:
        config['rl']['training']['max_steps'] = args.steps
    
    if args.learning_rate:
        config['rl']['training']['learning_rate'] = args.learning_rate
    
    if args.batch_size:
        config['rl']['training']['per_device_train_batch_size'] = args.batch_size
    
    # Set up output directory
    output_dir = args.output_dir or os.path.join("data", "trained_models", f"model_{int(datetime.now().timestamp())}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    model_loader = ModelLoader(args.config_path)
    reward_model = RewardModel(args.config_path)
    data_handler = DataHandler(args.config_path)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {config['model']['base_model_name']}")
    train_model = model_loader.prepare_model_for_training()
    tokenizer = model_loader.tokenizer or model_loader.load_model_and_tokenizer()[1]
    
    # Prepare dataset
    questions_path = args.questions_path or os.path.join("data", "training", "questions.json")
    documents_path = args.documents_path or "data/processed/documents"
    
    if not os.path.exists(questions_path):
        logger.error(f"Questions file not found: {questions_path}")
        sys.exit(1)
    
    logger.info(f"Preparing training data from {documents_path} and {questions_path}")
    train_data, eval_data = data_handler.prepare_document_training_data(
        documents_path=documents_path,
        questions_path=questions_path
    )
    
    # Create trainer
    trainer = GRPOTrainer(
        model=train_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        config_path=args.config_path,
        output_dir=output_dir
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=config['rl']['training']['learning_rate']
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['rl']['training']['max_steps']
    )
    
    # Train the model
    logger.info("Starting training")
    metrics = trainer.train(train_data, optimizer, scheduler)
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {output_dir}")
    logger.info(f"Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
