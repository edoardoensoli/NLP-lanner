#!/usr/bin/env python3
"""
Script di test per verificare che le API key di OpenAI funzionino correttamente
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import OPENAI_API_KEY
    print("Config.py loaded correctly")
    print(f"API Key present (first 10 characters): {OPENAI_API_KEY[:10]}...")
except ImportError as e:
    print(f"Error loading config.py: {e}")
    sys.exit(1)

try:
    from openai_func import GPT_response
    print("openai_func imported correctly")
except ImportError as e:
    print(f"Error importing openai_func: {e}")
    sys.exit(1)

# Test a simple API call
print("\nTesting OpenAI API call...")
try:
    response = GPT_response("Say 'Hello, API is working!' in Italian", "gpt-3.5-turbo")
    print(f"API call successful!")
    print(f"Response: {response}")
except Exception as e:
    print(f"API call failed: {e}")
    sys.exit(1)

print("\nAll tests passed! API keys are working correctly.")
