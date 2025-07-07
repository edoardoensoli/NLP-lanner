#!/usr/bin/env python3
"""
Test equipment detection logic independently
"""
import re

def test_equipment_detection():
    # Test queries
    test_cases = [
        ("Plan a 3-day ski trip to Whistler for 2 people with accommodation and car rental", False),
        ("Plan a 3-day ski trip to Whistler for 2 people with accommodation, car rental, and ski equipment rental", True),
        ("Plan a ski trip with equipment rental", True),
        ("Plan a ski trip with ski rental", True),
        ("Plan a ski trip with rent equipment", True),
        ("Plan a ski trip with rental equipment", True),
        ("Plan a ski trip with car rental only", False),
        ("Plan a ski trip with hotel rental", False),
        ("Plan a ski trip to rent skis", True),
    ]
    
    # Equipment detection patterns (same as in the code)
    ski_equipment_patterns = [
        r'\bequipment\b',
        r'\bski\s+rental\b',
        r'\brent\s+ski\b',
        r'\bski\s+equipment\b',
        r'\brental\s+equipment\b',
        r'\brent\s+equipment\b',
        r'\brent\s+skis\b'
    ]
    
    print("🧪 Testing Equipment Detection Logic")
    print("=" * 50)
    
    for query, expected in test_cases:
        query_lower = query.lower()
        equipment_requested = any(re.search(pattern, query_lower) for pattern in ski_equipment_patterns)
        
        status = "✅ PASS" if equipment_requested == expected else "❌ FAIL"
        print(f"{status} | Expected: {expected:5} | Detected: {equipment_requested:5} | Query: {query}")
    
    print("\n" + "=" * 50)
    print("Equipment detection test completed!")

if __name__ == "__main__":
    test_equipment_detection()
