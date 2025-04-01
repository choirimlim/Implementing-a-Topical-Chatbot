"""
Reward model module for the DocuChat application.
Implements reward functions for reinforcement learning.
"""

import re
import logging
import yaml
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class RewardModel:
    """
    Implements reward functions for RL training.
    Provides rule-based and heuristic rewards without requiring a neural reward model.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the reward model with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load reward weights from config
        self.accuracy_weight = self.config['rl']['rewards']['accuracy_weight']
        self.format_weight = self.config['rl']['rewards']['format_weight']
        self.cosine_weight = self.config['rl']['rewards']['cosine_weight']
        
        logger.info(f"Initialized RewardModel with weights: accuracy={self.accuracy_weight}, format={self.format_weight}, cosine={self.cosine_weight}")
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the final answer from a model response.
        Looks for answers in a boxed format: \\boxed{answer}.
        
        Args:
            text: Model-generated text response
            
        Returns:
            Extracted answer or None if not found
        """
        # Try to find an answer with \boxed{} format (mathematics)
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, text)
        
        if matches:
            return matches[-1].strip()  # Return the last boxed answer
        
        # If no boxed answer, look for a direct answer format
        answer_pattern = r'(?:answer|result)[:\s]+([^\n.]+)'
        matches = re.findall(answer_pattern, text, re.IGNORECASE)
        
        if matches:
            return matches[-1].strip()
        
        return None
    
    def _check_thinking_format(self, text: str) -> bool:
        """
        Check if the response follows the structured thinking format.
        Looks for <think>...</think> tags.
        
        Args:
            text: Model-generated text response
            
        Returns:
            True if the response contains proper thinking structure
        """
        # Check for thinking tags
        has_opening_tag = "<think>" in text.lower()
        has_closing_tag = "</think>" in text.lower()
        
        return has_opening_tag and has_closing_tag
    
    def _cosine_length_reward(self, text: str, optimal_length: int = 1500, max_penalty: float = 0.5) -> float:
        """
        Calculate a cosine-based reward that penalizes responses that are too short or too long.
        Provides a smooth curve that peaks at the optimal length.
        
        Args:
            text: Model-generated text response
            optimal_length: Target character length for responses
            max_penalty: Maximum penalty for very short or very long responses
            
        Returns:
            Cosine length reward between 1-max_penalty and 1.0
        """
        text_length = len(text)
        
        # Calculate ratio of actual to optimal length
        length_ratio = text_length / optimal_length
        
        # Apply cosine function to get a smooth curve
        # cos(0) = 1 (optimal), cos(pi/2) = 0 (far from optimal)
        cosine_value = math.cos(min(abs(math.log(length_ratio)) * 1.5, math.pi/2))
        
        # Scale to range [1-max_penalty, 1]
        scaled_reward = 1.0 - (max_penalty * (1.0 - cosine_value))
        
        return scaled_reward
    
    def calculate_accuracy_reward(self, response: str, expected_answer: str) -> float:
        """
        Calculate a reward based on answer accuracy.
        
        Args:
            response: Model-generated text response
            expected_answer: Expected correct answer
            
        Returns:
            Binary reward (1.0 for correct, 0.0 for incorrect)
        """
        # Extract the answer from the response
        extracted_answer = self._extract_answer(response)
        
        if not extracted_answer:
            logger.info(f"No answer found in response")
            return 0.0
        
        # Clean and normalize answers for comparison
        clean_extracted = self._normalize_answer(extracted_answer)
        clean_expected = self._normalize_answer(expected_answer)
        
        # Check for exact match
        if clean_extracted == clean_expected:
            logger.info(f"Answer correct: {extracted_answer}")
            return 1.0
        
        # Check for approximate numerical match
        if self._is_numerical_match(clean_extracted, clean_expected):
            logger.info(f"Approximate numerical match: {extracted_answer} ≈ {expected_answer}")
            return 1.0
        
        logger.info(f"Answer incorrect: {extracted_answer} != {expected_answer}")
        return 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer for comparison.
        
        Args:
            answer: Answer string
            
        Returns:
            Normalized answer string
        """
        # Convert to lowercase
        normalized = answer.lower()
        
        # Remove whitespace
        normalized = re.sub(r'\s+', '', normalized)
        
        # Remove punctuation
        normalized = re.sub(r'[.,;:\'"\(\)\[\]\{\}]', '', normalized)
        
        return normalized
    
    def _is_numerical_match(self, answer1: str, answer2: str) -> bool:
        """
        Check if two answers are numerically equivalent.
        
        Args:
            answer1: First answer string
            answer2: Second answer string
            
        Returns:
            True if answers match numerically
        """
        try:
            # Try to convert both to floats
            num1 = float(answer1)
            num2 = float(answer2)
            
            # Check for approximate equality (relative tolerance)
            return math.isclose(num1, num2, rel_tol=1e-9)
        except ValueError:
            # Not numerical values
            return False
    
    def calculate_format_reward(self, response: str) -> float:
        """
        Calculate a reward based on response format structure.
        
        Args:
            response: Model-generated text response
            
        Returns:
            Format compliance reward (1.0 for compliant, 0.0 for non-compliant)
        """
        has_proper_format = self._check_thinking_format(response)
        
        if has_proper_format:
            logger.info("Response has proper thinking format")
            return 1.0
        else:
            logger.info("Response does not have proper thinking format")
            return 0.0
    
    def calculate_cosine_reward(self, response: str, optimal_length: int = 1500) -> float:
        """
        Calculate a cosine-based length reward.
        
        Args:
            response: Model-generated text response
            optimal_length: Target character length for responses
            
        Returns:
            Cosine length reward between 0.5 and 1.0
        """
        cosine_reward = self._cosine_length_reward(response, optimal_length)
        
        logger.info(f"Cosine length reward: {cosine_reward:.4f} (length: {len(response)} chars)")
        
        return cosine_reward
    
    def calculate_combined_reward(self, response: str, expected_answer: Optional[str] = None) -> float:
        """
        Calculate a combined reward based on multiple criteria.
        
        Args:
            response: Model-generated text response
            expected_answer: Expected correct answer (optional)
            
        Returns:
            Combined weighted reward
        """
        # Calculate individual rewards
        format_reward = self.calculate_format_reward(response) if self.format_weight > 0 else 0.0
        cosine_reward = self.calculate_cosine_reward(response) if self.cosine_weight > 0 else 0.0
        
        # Calculate accuracy reward if expected answer is provided
        accuracy_reward = 0.0
        if expected_answer and self.accuracy_weight > 0:
            accuracy_reward = self.calculate_accuracy_reward(response, expected_answer)
        
        # Combine rewards with weights
        total_weight = 0.0
        weighted_reward = 0.0
        
        if self.format_weight > 0:
            weighted_reward += self.format_weight * format_reward
            total_weight += self.format_weight
        
        if self.cosine_weight > 0:
            weighted_reward += self.cosine_weight * cosine_reward
            total_weight += self.cosine_weight
        
        if expected_answer and self.accuracy_weight > 0:
            weighted_reward += self.accuracy_weight * accuracy_reward
            total_weight += self.accuracy_weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_reward = weighted_reward / total_weight
        else:
            combined_reward = 0.0
        
        logger.info(f"Combined reward: {combined_reward:.4f}")
        
        return combined_reward
