"""
Model loading module for the DocuChat application.
Handles loading LLMs and embedding models with optimizations.
"""

import os
import logging
import yaml
import torch
from typing import Dict, Any, Optional, Union, List

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Loads and manages language models for inference and training.
    Supports quantization, LoRA, and other optimizations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the model loader with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['base_model_name']
        self.device = self._resolve_device()
        self.load_in_8bit = self.config['model']['load_in_8bit']
        self.load_in_4bit = self.config['model']['load_in_4bit']
        
        logger.info(f"Initialized ModelLoader for {self.model_name} on {self.device}")
    
    def _resolve_device(self) -> str:
        """
        Resolve the device setting from the config.
        If 'auto', automatically select the best available device.
        
        Returns:
            Device string ('cuda', 'cpu', or 'mps')
        """
        device_config = self.config['model']['device']
        
        if device_config != 'auto':
            return device_config
        
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create a BitsAndBytesConfig for model quantization.
        
        Returns:
            BitsAndBytesConfig object or None if no quantization is enabled
        """
        if not (self.load_in_8bit or self.load_in_4bit):
            return None
        
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        return None
    
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the model and tokenizer with appropriate optimizations.
        
        Returns:
            Tuple containing:
                - Model instance
                - Tokenizer instance
        """
        logger.info(f"Loading model {self.model_name}")
        
        # Prepare quantization config if enabled
        quantization_config = self._get_quantization_config()
        
        # Load the model with appropriate settings
        model_kwargs = {
            "device_map": self.device if self.device != 'cpu' else None,
            "trust_remote_code": True,
        }
        
        # Add quantization config if available
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Enable flash attention if available and configured
        if self.config['model']['use_flash_attention']:
            try:
                from transformers.utils import is_flash_attn_available
                if is_flash_attn_available():
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning("Flash Attention not available, using default attention implementation")
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token if not defined (important for proper batching)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "</s>"
        
        logger.info(f"Successfully loaded model and tokenizer")
        
        return model, tokenizer
    
    def load_embedding_model(self):
        """
        Load a sentence embedding model for document vectorization.
        
        Returns:
            Sentence transformer model for generating embeddings
        """
        from sentence_transformers import SentenceTransformer
        
        embedding_model_name = self.config['model']['embedding_model']
        logger.info(f"Loading embedding model {embedding_model_name}")
        
        model = SentenceTransformer(embedding_model_name)
        
        # Move to appropriate device
        if self.device == 'cuda' and torch.cuda.is_available():
            model = model.to(torch.device('cuda'))
        elif self.device == 'mps' and torch.backends.mps.is_available():
            model = model.to(torch.device('mps'))
        
        return model
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get generation configuration parameters from the config file.
        
        Returns:
            Dictionary of generation parameters
        """
        return {
            "max_new_tokens": self.config['model']['max_new_tokens'],
            "temperature": self.config['model']['temperature'],
            "top_p": self.config['model']['top_p'],
            # Add more parameters as needed
        }
    
    def prepare_model_for_training(self) -> PreTrainedModel:
        """
        Prepare the model for RL training, applying necessary optimizations.
        
        Returns:
            Model ready for training
        """
        logger.info(f"Preparing model {self.model_name} for training")
        
        # Load the base model with minimal optimizations to save memory
        model_kwargs = {
            "device_map": self.device if self.device != 'cpu' else None,
            "trust_remote_code": True,
        }
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['rl']['training']['gradient_checkpointing']:
            model_kwargs["use_gradient_checkpointing"] = True
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Apply PEFT/LoRA if configured
        if self.config['rl']['training']['use_peft']:
            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            
            # Prepare model for PEFT if using quantization
            if self.load_in_8bit or self.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            
            # Define LoRA configuration
            lora_config = LoraConfig(
                r=16,                    # Rank
                lora_alpha=32,           # Alpha parameter for LoRA scaling
                target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # Adjust based on model architecture
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            logger.info("Applied LoRA to model for training")
        
        return model
