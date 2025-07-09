#!/usr/bin/env python3
"""
Test script for the enhanced model fallback system
"""

import sys
import os
sys.path.append(os.getcwd())

from testing_z3_gurobi import run_batch_benchmark
from model_manager import get_intelligent_fallback_sequence

def test_enhanced_fallback():
    """Test the enhanced fallback system with a few queries"""
    
    print("🧪 TESTING ENHANCED MODEL FALLBACK SYSTEM")
    print("=" * 50)
    
    # Small set of test queries
    test_queries = [
        "Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros",
        "Organize a 4-day ski vacation to Zermatt for 2 people with budget 3000 euros and car rental",
        "Plan a 5-day ski trip to Chamonix for 4 people with budget 4500 euros and equipment rental"
    ]
    
    print(f"Test queries: {len(test_queries)}")
    
    # Get enhanced model sequence
    available_models = get_intelligent_fallback_sequence(max_models=6)
    print(f"Model sequence: {available_models}")
    
    print("\n🚀 Starting test benchmark...")
    
    # Run benchmark
    results = run_batch_benchmark(test_queries, available_models, verbose=True)
    
    print("\n✅ Test completed!")
    print(f"Results: {len(results)} benchmarks completed")
    
    # Check success rates
    successful_results = [r for r in results if r.pure_llm_result.success and r.z3_result.success and r.gurobi_result.success]
    success_rate = len(successful_results) / len(results) * 100
    
    print(f"Success rate: {success_rate:.1f}% ({len(successful_results)}/{len(results)})")
    
    if success_rate >= 80:
        print("🎉 Test passed! Enhanced fallback system working well.")
        print("✅ Ready to run comprehensive benchmark.")
        return True
    else:
        print("⚠️  Test shows issues. Need to investigate further.")
        return False

if __name__ == "__main__":
    success = test_enhanced_fallback()
    
    if success:
        print("\n🔄 Would you like to run the full comprehensive benchmark? (15 queries)")
        print("This will take approximately 10-15 minutes with enhanced delays.")
    else:
        print("\n🔧 Please fix the fallback system before running the full benchmark.")
