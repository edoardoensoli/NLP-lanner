#!/usr/bin/env python3
# Test script to verify dynamic country loading

import pandas as pd
import json
from test_skiplanner_openai import OpenAILLM

def test_dynamic_countries():
    """Test that the system can load all countries and resorts dynamically"""
    
    # Load all countries from CSV
    df = pd.read_csv('dataset_ski/resorts/resorts.csv')
    all_countries = df['Country'].unique().tolist()
    
    print(f"🌍 Countries available in dataset: {len(all_countries)}")
    for country in sorted(all_countries):
        country_resorts = df[df['Country'] == country]['Resort'].unique().tolist()
        print(f"  {country}: {len(country_resorts)} resorts")
    
    print("\n" + "="*60)
    
    # Test queries for different countries
    test_queries = [
        "Plan a 5-day ski trip to Italy for 2 people with budget 2000 euros",
        "Plan a 7-day ski trip to Norway for 4 people with budget 3500 euros", 
        "Plan a 3-day ski trip to Austria for 2 people with budget 1500 euros",
        "Plan a 6-day ski trip to Switzerland for 3 people with budget 4000 euros",
        "Plan a 4-day ski trip to France for 2 people with budget 2500 euros",
        "Plan a 5-day ski trip to Canada for 4 people with budget 3000 euros",
        "Plan a 3-day ski trip to Germany for 2 people with budget 1800 euros",
        "Plan a 7-day ski trip to United States for 3 people with budget 4500 euros",
        "Plan a 4-day ski trip to Chile for 2 people with budget 2200 euros",
        "Plan a 6-day ski trip to New Zealand for 3 people with budget 3800 euros"
    ]
    
    # Initialize OpenAI LLM
    llm = OpenAILLM(verbose=True)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🎿 TEST {i}: {query}")
        print("-" * 50)
        
        # Parse the query
        try:
            json_str = llm._manual_parse_query(query)
            parsed_json = json.loads(json_str)
            
            # Display results
            country = parsed_json.get('destination_country', 'N/A')
            available_resorts = parsed_json.get('available_resorts', [])
            primary_destination = parsed_json.get('destination', 'N/A')
            
            print(f"✅ Country detected: {country}")
            print(f"✅ Available resorts: {len(available_resorts)}")
            print(f"✅ Primary destination: {primary_destination}")
            
            if len(available_resorts) > 1:
                print(f"✅ All resorts for {country}: {', '.join(available_resorts[:5])}")
                if len(available_resorts) > 5:
                    print(f"    ... and {len(available_resorts) - 5} more")
            else:
                print(f"⚠️  Only {len(available_resorts)} resort(s) found")
                
        except Exception as e:
            print(f"❌ Error parsing query: {e}")
    
    print("\n" + "="*60)
    print("🎯 Dynamic country loading test completed!")

if __name__ == "__main__":
    test_dynamic_countries()
