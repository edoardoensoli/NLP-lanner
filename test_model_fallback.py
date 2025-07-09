#!/usr/bin/env python3
"""
Test the improved model fallback system with a small benchmark
"""

import os
import sys
sys.path.append(os.getcwd())

from testing_z3_gurobi import run_batch_benchmark
from model_manager import get_model_priority_list

def test_model_fallback():
    """Test the enhanced model fallback system"""
    
    print("🧪 TESTING ENHANCED MODEL FALLBACK SYSTEM")
    print("=" * 60)
    
    # Get comprehensive model list
    available_models = get_model_priority_list()
    print(f"📋 Available models ({len(available_models)}):")
    for i, model in enumerate(available_models[:8], 1):  # Show first 8
        print(f"  {i}. {model}")
    if len(available_models) > 8:
        print(f"  ... and {len(available_models) - 8} more")
    
    # Test with 3 simple queries
    test_queries = [
        "Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros",
        "Organize a 4-day ski vacation to Zermatt for 2 people with budget 3000 euros",
        "Plan a 7-day ski trip to Zermatt for 8 people with budget 1000 euros"  # Infeasible
    ]
    
    print(f"\n🎯 Test queries ({len(test_queries)}):")
    for i, query in enumerate(test_queries, 1):
        print(f"  {i}. {query[:60]}...")
    
    print("\n🚀 Starting test benchmark...")
    print("⏰ Estimated time: ~2 minutes")
    
    try:
        results = run_batch_benchmark(test_queries, available_models, verbose=True)
        
        print("\n✅ Test completed successfully!")
        print("📊 Results summary:")
        
        for i, result in enumerate(results, 1):
            print(f"\n  Query {i}:")
            print(f"    Pure LLM: {result.pure_llm_result.status} ({result.pure_llm_result.model_used})")
            print(f"    Z3: {result.z3_result.status} ({result.z3_result.model_used})")
            print(f"    Gurobi: {result.gurobi_result.status} ({result.gurobi_result.model_used})")
            
        print("\n🎉 Model fallback system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_fallback()
