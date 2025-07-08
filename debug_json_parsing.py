#!/usr/bin/env python3
"""
Debug script to check query JSON parsing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai_func import OpenAI_API

def test_query_parsing():
    """Test how the query is parsed into JSON"""
    
    query = "Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment"
    
    print("🔍 Testing Query JSON Parsing")
    print(f"Query: {query}")
    print()
    
    # Create LLM client
    llm_client = OpenAI_API(
        model_name="Phi-3-mini-4k-instruct",
        api_type="github_models",
        fallback_models=["Phi-3-medium-4k-instruct"]
    )
    
    # Test Z3 JSON parsing
    print("🔍 Z3 JSON Parsing:")
    try:
        z3_prompt = f"""
Convert the following natural language query into a JSON object for a ski trip planner:

Query: {query}

The JSON should have the following structure:
{{
  "domain": "ski",
  "destination": "resort_name",
  "days": 3,
  "people": 2,
  "budget": 1500,
  "special_requirements": ["car rental", "equipment", etc.]
}}

Return only the JSON object, no other text.
"""
        
        z3_response = llm_client._query_api(z3_prompt)
        print(f"Z3 JSON: {z3_response}")
    except Exception as e:
        print(f"Z3 JSON Error: {e}")
    
    print()
    
    # Test Gurobi JSON parsing  
    print("🔍 Gurobi JSON Parsing:")
    try:
        gurobi_prompt = f"""
Convert the following natural language query into a JSON object for a ski trip planner:

Query: {query}

The JSON should have the following structure:
- domain: "ski"
- destination: the destination name
- days: number of days
- people: number of people
- budget: budget amount in euros
- special_requirements: list of special requirements (car rental, equipment, etc.)

Return only the JSON object, no other text.
"""
        
        gurobi_response = llm_client._query_api(gurobi_prompt)
        print(f"Gurobi JSON: {gurobi_response}")
    except Exception as e:
        print(f"Gurobi JSON Error: {e}")

if __name__ == "__main__":
    test_query_parsing()
