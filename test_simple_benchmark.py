#!/usr/bin/env python3
"""
Simple test of the three-way benchmarking functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pure_llm():
    """Test the Pure LLM functionality in isolation"""
    try:
        from testing_z3_gurobi import LLMClient, pipeline_ski_pure_llm
        
        print("Testing Pure LLM functionality...")
        
        # Test query
        query = "Plan a 3-day ski trip to Livigno for 2 people with budget 1500 euros"
        
        # Test Pure LLM
        result = pipeline_ski_pure_llm(
            query=query,
            mode="test",
            model_name="gpt-4o-mini",
            index=1,
            model_version="gpt-4o-mini",
            verbose=True
        )
        
        if result:
            print("✅ Pure LLM test successful!")
            print("Result preview:", result[:200] + "...")
            return True
        else:
            print("❌ Pure LLM test failed - no result")
            return False
            
    except Exception as e:
        print(f"❌ Pure LLM test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_classes():
    """Test the benchmark result classes"""
    try:
        from testing_z3_gurobi import SkiBenchmarkResult, ThreeWayBenchmarkComparison
        
        print("Testing benchmark classes...")
        
        # Create test results
        pure_llm_result = SkiBenchmarkResult("Pure LLM")
        pure_llm_result.success = True
        pure_llm_result.execution_time = 2.5
        pure_llm_result.total_cost = 1200.0
        
        z3_result = SkiBenchmarkResult("Z3")
        z3_result.success = True
        z3_result.execution_time = 5.2
        z3_result.total_cost = 1150.0
        
        gurobi_result = SkiBenchmarkResult("Gurobi")
        gurobi_result.success = True
        gurobi_result.execution_time = 3.8
        gurobi_result.total_cost = 1180.0
        
        # Test comparison
        comparison = ThreeWayBenchmarkComparison(
            "Test query",
            pure_llm_result,
            z3_result,
            gurobi_result
        )
        
        analysis = comparison.analyze_comparison()
        
        print(f"✅ Benchmark classes test successful!")
        print(f"Winner: {analysis['winner']}")
        print(f"Success count: {analysis['success_summary']['success_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark classes test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 SIMPLE BENCHMARK TESTING")
    print("=" * 50)
    
    tests = [
        ("Pure LLM functionality", test_pure_llm),
        ("Benchmark classes", test_benchmark_classes),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 TEST RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    main()
