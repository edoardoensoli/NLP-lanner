#!/usr/bin/env python3
"""
Enhanced model management with comprehensive fallback system and rate limit handling
"""

import time
import random

def get_available_models():
    """
    Returns a comprehensive list of available models with intelligent fallbacks
    """
    # Primary models (usually available and high quality)
    primary_models = [
        'DeepSeek-R1',
        'Phi-3-medium-4k-instruct',
        'Phi-3-mini-4k-instruct', 
        'Llama-3.2-11B-Vision-Instruct'
    ]
    
    # Secondary fallback models (good performance, often less rate-limited)
    secondary_models = [
        'Llama-3.2-3B-Instruct',
        'Llama-3.2-1B-Instruct', 
        'Phi-3-small-8k-instruct',
        'Phi-3-small-128k-instruct',
        'Meta-Llama-3-8B-Instruct'
    ]
    
    # Tertiary fallback models (reliable alternatives)
    tertiary_models = [
        'gpt-3.5-turbo',
        'gpt-4o-mini',
        'Meta-Llama-3.1-8B-Instruct',
        'Meta-Llama-3.1-70B-Instruct',
        'CodeLlama-7b-Instruct-hf',
        'CodeLlama-13b-Instruct-hf'
    ]
    
    # Emergency fallback models (when others are rate-limited)
    emergency_models = [
        'Mistral-7B-Instruct-v0.1',
        'Mistral-7B-Instruct-v0.2',
        'AI21-Jamba-Instruct',
        'Cohere-command-r'
    ]
    
    # Return comprehensive list
    return primary_models + secondary_models + tertiary_models + emergency_models

def get_model_priority_list():
    """
    Returns models ordered by preference for ski planning tasks
    """
    return [
        # Best reasoning models first
        'DeepSeek-R1',
        'Phi-3-medium-4k-instruct',
        'Llama-3.2-11B-Vision-Instruct',
        
        # Good balance models
        'Phi-3-mini-4k-instruct',
        'Llama-3.2-3B-Instruct',
        'Phi-3-small-8k-instruct',
        
        # Lightweight fallbacks
        'Llama-3.2-1B-Instruct',
        'Phi-3-small-128k-instruct',
        
        # External API fallbacks (if available)
        'gpt-3.5-turbo',
        'gpt-4o-mini',
        'Meta-Llama-3-8B-Instruct',
        'Meta-Llama-3-70B-Instruct'
    ]

def get_rate_limit_safe_models():
    """
    Returns models that typically have higher rate limits or are less used
    """
    return [
        'Phi-3-mini-4k-instruct',
        'Phi-3-small-8k-instruct',
        'Phi-3-small-128k-instruct',
        'Llama-3.2-1B-Instruct',
        'Llama-3.2-3B-Instruct',
        'CodeLlama-7b-Instruct-hf',
        'Mistral-7B-Instruct-v0.1',
        'AI21-Jamba-Instruct'
    ]

def get_intelligent_fallback_sequence(max_models=8):
    """
    Get an intelligent sequence of models to try, mixing high-quality and safe options
    """
    # Start with a mix of high-quality and rate-limit-safe models
    priority_models = [
        'DeepSeek-R1',  # High quality
        'Phi-3-mini-4k-instruct',  # Rate-limit safe
        'Phi-3-medium-4k-instruct',  # High quality  
        'Llama-3.2-3B-Instruct',  # Rate-limit safe
        'Llama-3.2-11B-Vision-Instruct',  # High quality
        'Phi-3-small-8k-instruct',  # Rate-limit safe
        'Meta-Llama-3-8B-Instruct',  # Fallback
        'CodeLlama-7b-Instruct-hf'  # Emergency fallback
    ]
    
    return priority_models[:max_models]

def should_increase_delay(attempt_count):
    """
    Determine if we should increase delay between attempts
    """
    return attempt_count > 2

def get_adaptive_delay(attempt_count):
    """
    Get adaptive delay that increases with attempts to handle rate limits
    """
    if attempt_count <= 1:
        return 5  # 5 seconds for first retry
    elif attempt_count <= 3:
        return 15  # 15 seconds for subsequent retries
    else:
        return 30  # 30 seconds for later retries
    return [
        'Phi-3-mini-4k-instruct',
        'Phi-3-small-8k-instruct', 
        'Llama-3.2-1B-Instruct',
        'Llama-3.2-3B-Instruct',
        'Meta-Llama-3-8B-Instruct'
    ]

def get_model_characteristics():
    """
    Returns characteristics of each model for intelligent selection
    """
    return {
        'DeepSeek-R1': {
            'reasoning': 'excellent',
            'speed': 'slow',
            'rate_limit_risk': 'high',
            'context_length': '4k'
        },
        'Phi-3-mini-4k-instruct': {
            'reasoning': 'good',
            'speed': 'fast',
            'rate_limit_risk': 'low',
            'context_length': '4k'
        },
        'Phi-3-medium-4k-instruct': {
            'reasoning': 'very good',
            'speed': 'medium',
            'rate_limit_risk': 'medium',
            'context_length': '4k'
        },
        'Llama-3.2-11B-Vision-Instruct': {
            'reasoning': 'very good',
            'speed': 'medium',
            'rate_limit_risk': 'medium',
            'context_length': '4k'
        },
        'Llama-3.2-3B-Instruct': {
            'reasoning': 'good',
            'speed': 'fast',
            'rate_limit_risk': 'low',
            'context_length': '4k'
        },
        'Llama-3.2-1B-Instruct': {
            'reasoning': 'fair',
            'speed': 'very fast',
            'rate_limit_risk': 'very low',
            'context_length': '4k'
        }
    }

# Test the model availability
if __name__ == "__main__":
    print("Available Models:")
    for i, model in enumerate(get_available_models(), 1):
        print(f"  {i:2d}. {model}")
    
    print(f"\nTotal models available: {len(get_available_models())}")
    print(f"Priority order: {get_model_priority_list()[:5]}...")
    print(f"Rate-limit safe models: {get_rate_limit_safe_models()[:3]}...")
