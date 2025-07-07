#!/usr/bin/env python3
"""
Simple test script to verify OpenAI API integration with ski planner
"""

import sys
import os
import json

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import OpenAI functions
from openai_func import GPT_response

def test_openai_api():
    """Test OpenAI API with a simple query"""
    print("🔥 TESTING REAL OPENAI API")
    print("=" * 50)
    
    # Test query
    test_query = "Plan a 5-day ski trip to Zermatt for 2 people. I want to rent a car and our budget is 4000 euros."
    
    print(f"Query: {test_query}")
    print("-" * 50)
    
    try:
        # Test JSON conversion
        json_prompt = f"""Convert this ski trip query to JSON format:
        
Query: {test_query}

Please provide a JSON response with fields like: destination, days, people, budget, car_type, etc.

JSON:"""
        
        print("🚀 Making OpenAI API call for JSON conversion...")
        json_response = GPT_response(json_prompt, "gpt-3.5-turbo")
        
        print("✅ OpenAI API Response:")
        print(json_response)
        print("-" * 50)
        
        # Try to parse the JSON
        try:
            parsed_json = json.loads(json_response.replace('```json', '').replace('```', '').strip())
            print("✅ JSON parsing successful!")
            print(f"Parsed data: {json.dumps(parsed_json, indent=2)}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_api()
    if success:
        print("\n🎉 Real OpenAI API integration works!")
        print("Now we can proceed with Z3 vs Gurobi benchmarking.")
    else:
        print("\n❌ OpenAI API integration failed.")
        print("Please check your API key in config.py")
