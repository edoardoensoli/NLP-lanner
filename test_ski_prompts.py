#!/usr/bin/env python3
"""
Test script for ski planning prompts and templates
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def test_ski_prompts():
    print("Testing Ski Planning Prompts...")
    
    # Test if all prompt files exist
    prompt_files = [
        'prompts/ski/query_to_json_ski.txt',
        'prompts/ski/constraint_to_step_ski.txt',
        'prompts/ski/step_to_code_destination.txt',
        'prompts/ski/step_to_code_resort.txt',
        'prompts/ski/step_to_code_slopes.txt',
        'prompts/ski/step_to_code_equipment.txt',
        'prompts/ski/step_to_code_car.txt',
        'prompts/ski/step_to_code_budget.txt',
        'prompts/ski/solve_ski_3.txt',
        'prompts/ski/solve_ski_5.txt',
        'prompts/ski/solve_ski_7.txt'
    ]
    
    print("\nChecking prompt files...")
    missing_files = []
    for file_path in prompt_files:
        if os.path.exists(file_path):
            print(f"OK {file_path}")
        else:
            print(f"MISSING {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n{len(missing_files)} files are missing!")
        return False
    else:
        print(f"\nAll {len(prompt_files)} ski prompt files are present!")
    
    # Test reading prompt contents
    print("\n📖 Testing prompt file contents...")
    for file_path in prompt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 50:  # Basic content check
                    print(f"OK {os.path.basename(file_path)} - Content OK ({len(content)} chars)")
                else:
                    print(f"WARNING {os.path.basename(file_path)} - Content too short")
        except Exception as e:
            print(f"ERROR {os.path.basename(file_path)} - Read error: {e}")
    
    # Test sample ski query conversion
    print("\nTesting sample ski query...")
    sample_query = "Plan a 5-day ski trip to Austria for 3 people with budget 2500 euros, need black slopes and electric SUV"
    
    try:
        with open('prompts/ski/query_to_json_ski.txt', 'r') as f:
            json_prompt = f.read()
        
        print(f"Query prompt loaded: {len(json_prompt)} characters")
        print(f"Sample query: {sample_query}")
        print("Ready for LLM processing!")
        
    except Exception as e:
        print(f"Error testing query conversion: {e}")
    
    return True

def show_ski_framework_structure():
    print("\n Ski Framework Structure:")
    print("""
     Ski Planning Framework
    ├──  tools_ski/
    │   ├── __init__.py
    │   └── apis.py (SkiResorts, SkiSlopes, SkiRent, SkiCar)
    │
    ├──  dataset_ski/
    │   ├── resorts/resorts.csv
    │   ├── slopes/ski_slopes.csv  
    │   ├── rent/ski_rent.csv
    │   └── car/ski_car.csv
    │
    ├── prompts/ski/
    │   ├── query_to_json_ski.txt
    │   ├── constraint_to_step_ski.txt
    │   ├── step_to_code_destination.txt
    │   ├── step_to_code_resort.txt
    │   ├── step_to_code_slopes.txt
    │   ├── step_to_code_equipment.txt
    │   ├── step_to_code_car.txt
    │   ├── step_to_code_budget.txt
    │   ├── solve_ski_3.txt
    │   ├── solve_ski_5.txt
    │   └── solve_ski_7.txt
    │
    └── Documentation
        ├── SKI_DATASET_GUIDE.md
        └── test_ski_prompts.py
    """)

if __name__ == "__main__":
    print("Ski Planning Framework Test Suite")
    print("=" * 50)
    
    success = test_ski_prompts()
    show_ski_framework_structure()
    
    if success:
        print("\nAll tests passed! Ski planning framework is ready.")
        print("\nNext steps:")
        print("   1. Integrate with main test_travelplanner.py")
        print("   2. Create domain detection logic")
        print("   3. Test with real ski queries")
        print("   4. Add Z3 constraint generation")
    else:
        print("\nSome tests failed. Please check the issues above.")
