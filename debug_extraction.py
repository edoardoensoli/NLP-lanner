#!/usr/bin/env python3
"""Debug script for query parameter extraction"""

import sys
import os
sys.path.append(os.getcwd())

from ski_planner_metrics import QueryParameterExtractor

def test_query_extraction():
    extractor = QueryParameterExtractor()
    
    # Test queries
    test_queries = [
        "Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros",
        "Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment",
        "Plan a ski trip to Zermatt for 4 people, need car rental",
        "Simple ski trip to Cortina for 2 people"
    ]
    
    print('=== QUERY PARAMETER EXTRACTION DEBUG ===')
    
    for i, query in enumerate(test_queries, 1):
        print(f'\n{i}. Query: "{query}"')
        params = extractor.extract_parameters(query)
        
        print(f'   car_required: {params.get("car_required", False)}')
        print(f'   equipment_required: {params.get("equipment_required", False)}')
        print(f'   special_requirements: {params.get("special_requirements", [])}')
        
        # Show the logic that determines these flags
        query_lower = query.lower()
        car_patterns = [
            r'car rental', r'rent.*car', r'suv', r'sedan', r'pick up',
            r'cabriolet', r'electric', r'hybrid', r'diesel', r'petrol'
        ]
        equipment_patterns = [
            r'equipment', r'ski.*rental', r'rent.*ski', r'ski.*equipment',
            r'rental.*equipment', r'rent.*equipment', r'rent.*skis'
        ]
        
        print(f'   Car patterns found: {[p for p in car_patterns if p.replace(".*", " ") in query_lower]}')
        print(f'   Equipment patterns found: {[p for p in equipment_patterns if p.replace(".*", " ") in query_lower]}')

if __name__ == "__main__":
    test_query_extraction()
