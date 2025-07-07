#!/usr/bin/env python3
# Test fuzzy matching for resort names

import pandas as pd
import json
from test_skiplanner_openai import OpenAILLM

def test_fuzzy_matching():
    """Test fuzzy matching for resort names"""
    
    # Test specific resort queries
    test_queries = [
        "Plan a 3-day ski trip to Livigno for 2 people, budget 1500 euros",
        "Plan a 5-day ski trip to Cortina d'Ampezzo for 4 people, budget 3000 euros",
        "Plan a 7-day ski trip to Val d'Isère for 6 people, budget 5000 euros",
        "Plan a 4-day ski trip to Zermatt for 3 people, budget 2500 euros",
        "Plan a 6-day ski trip to St. Moritz for 2 people, budget 4000 euros"
    ]
    
    # Initialize OpenAI LLM
    llm = OpenAILLM(verbose=True)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎿 FUZZY MATCH TEST {i}: {query}")
        print("-" * 70)
        
        # Parse the query with fuzzy matching
        try:
            json_str = llm._manual_parse_query(query)
            parsed_json = json.loads(json_str)
            
            # Display results
            country = parsed_json.get('destination_country', 'N/A')
            available_resorts = parsed_json.get('available_resorts', [])
            primary_destination = parsed_json.get('destination', 'N/A')
            
            print(f"✅ Country/Region: {country}")
            print(f"✅ Available resorts: {len(available_resorts)}")
            print(f"✅ Primary destination: {primary_destination}")
            
            if available_resorts:
                print(f"✅ Resort list: {', '.join(available_resorts)}")
            else:
                print(f"❌ No resorts found!")
                
        except Exception as e:
            print(f"❌ Error parsing query: {e}")
    
    print("\n" + "="*70)
    print("🎯 Fuzzy matching test completed!")

if __name__ == "__main__":
    test_fuzzy_matching()
