#!/usr/bin/env python3
"""
Comprehensive benchmark for ski trip planning system
Creates 15 queries (10 feasible, 5 infeasible) with varying difficulty levels
"""

import os
import sys
sys.path.append(os.getcwd())

from testing_z3_gurobi import run_batch_benchmark
import json
from datetime import datetime

def create_benchmark_queries():
    """Create 15 carefully designed queries for comprehensive evaluation"""
    
    # FEASIBLE QUERIES (10 total)
    # Easy queries (3) - Basic requirements, low constraints
    easy_queries = [
        "Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros",
        "Organize a 4-day ski vacation to Zermatt for 2 people with budget 3000 euros", 
        "Book a 2-day ski trip to Cortina d'Ampezzo for 1 person with budget 1000 euros"
    ]
    
    # Medium queries (4) - Additional requirements, moderate constraints
    medium_queries = [
        "Plan a 5-day ski trip to Chamonix for 4 people with budget 4500 euros and car rental",
        "Organize a 3-day ski vacation to St. Moritz for 3 people with budget 2500 euros and equipment rental",
        "Book a 6-day ski adventure to Val d'Isère for 2 people with budget 3500 euros, need SUV and equipment",
        "Plan a 4-day ski trip to Innsbruck for 3 people with budget 2800 euros and intermediate slopes"
    ]
    
    # Hard queries (3) - Complex requirements, tight constraints
    hard_queries = [
        "Organize a 7-day ski vacation to Cortina d'Ampezzo for 6 people with budget 5000 euros, need car rental and equipment rental",
        "Plan a 5-day ski trip to Zermatt for 4 people with budget 4000 euros, need SUV rental and ski equipment for advanced slopes",
        "Book a 8-day ski adventure to Chamonix for 5 people with budget 6000 euros, need car rental, equipment, and beginner slopes"
    ]
    
    # INFEASIBLE QUERIES (5 total)
    # Budget too low, impossible group sizes, conflicting requirements
    infeasible_queries = [
        "Plan a 7-day ski trip to Zermatt for 8 people with budget 1000 euros",  # Budget way too low
        "Organize a 10-day ski vacation to St. Moritz for 6 people with budget 2000 euros and car rental",  # Impossible budget
        "Book a 5-day ski trip to Chamonix for 10 people with budget 1500 euros and equipment rental",  # Group too large, budget too low
        "Plan a 14-day ski adventure to Val d'Isère for 4 people with budget 2500 euros and SUV rental",  # Duration too long, budget insufficient
        "Organize a 6-day ski trip to Cortina d'Ampezzo for 7 people with budget 3000 euros, need car and equipment"  # Borderline impossible
    ]
    
    # Combine all queries with metadata
    all_queries = []
    
    # Add easy queries
    for i, query in enumerate(easy_queries, 1):
        all_queries.append({
            'id': f'E{i}',
            'query': query,
            'difficulty': 'Easy',
            'expected_feasible': True,
            'description': 'Basic requirements, generous budget'
        })
    
    # Add medium queries
    for i, query in enumerate(medium_queries, 1):
        all_queries.append({
            'id': f'M{i}',
            'query': query,
            'difficulty': 'Medium',
            'expected_feasible': True,
            'description': 'Additional services, moderate constraints'
        })
    
    # Add hard queries
    for i, query in enumerate(hard_queries, 1):
        all_queries.append({
            'id': f'H{i}',
            'query': query,
            'difficulty': 'Hard',
            'expected_feasible': True,
            'description': 'Complex requirements, tight budget'
        })
    
    # Add infeasible queries
    for i, query in enumerate(infeasible_queries, 1):
        all_queries.append({
            'id': f'I{i}',
            'query': query,
            'difficulty': 'Infeasible',
            'expected_feasible': False,
            'description': 'Impossible constraints, testing suggestion quality'
        })
    
    return all_queries

def run_comprehensive_benchmark():
    """Run the comprehensive benchmark and save results"""
    
    print("🎿 COMPREHENSIVE SKI PLANNER BENCHMARK")
    print("=" * 60)
    print("📋 Query Distribution:")
    print("  • Easy queries (E1-E3): 3 queries")
    print("  • Medium queries (M1-M4): 4 queries")  
    print("  • Hard queries (H1-H3): 3 queries")
    print("  • Infeasible queries (I1-I5): 5 queries")
    print("  • Total: 15 queries")
    print("=" * 60)
    
    # Create queries
    benchmark_queries = create_benchmark_queries()
    
    # Extract just the query strings for the benchmark system
    query_strings = [q['query'] for q in benchmark_queries]
    
    # Run all 15 queries for comprehensive evaluation
    # query_strings = query_strings[:2]  # Commented out for full benchmark
    # benchmark_queries = benchmark_queries[:2]  # Commented out for full benchmark
    
    # Run batch benchmark
    print(f"\n🚀 Starting comprehensive benchmark with {len(query_strings)} queries...")
    print("⏰ Estimated time: ~5 minutes")
    
    # Get available models with comprehensive fallback
    try:
        from model_manager import get_intelligent_fallback_sequence, get_rate_limit_safe_models
        # Use intelligent fallback sequence for better model selection
        available_models = get_intelligent_fallback_sequence(max_models=10)
        print(f"Using intelligent model sequence: {available_models[:3]}... (showing first 3 of {len(available_models)})")
        safe_models = get_rate_limit_safe_models()
        print(f"Rate-limit safe fallbacks available: {len(safe_models)} models")
    except Exception as e:
        print(f"Warning: Could not load enhanced model manager: {e}")
        # Comprehensive fallback list
        available_models = [
            'DeepSeek-R1', 
            'Phi-3-mini-4k-instruct', 
            'Phi-3-medium-4k-instruct',
            'Llama-3.2-11B-Vision-Instruct',
            'Llama-3.2-3B-Instruct',
            'Llama-3.2-1B-Instruct',
            'Phi-3-small-8k-instruct',
            'Phi-3-small-128k-instruct',
            'gpt-3.5-turbo',
            'Meta-Llama-3-8B-Instruct'
        ]
        print(f"Using fallback models: {available_models[:3]}... (showing first 3 of {len(available_models)})")
    
    # Run the benchmark
    results = run_batch_benchmark(query_strings, available_models, verbose=True)
    
    # Enhance results with query metadata
    enhanced_results = []
    for i, result in enumerate(results):
        enhanced_result = result.to_dict()
        enhanced_result.update(benchmark_queries[i])
        enhanced_results.append(enhanced_result)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results/comprehensive_benchmark_{timestamp}.json"
    
    comprehensive_data = {
        'benchmark_info': {
            'title': 'Comprehensive Ski Planner Benchmark',
            'timestamp': timestamp,
            'total_queries': len(query_strings),
            'query_distribution': {
                'easy': 3,
                'medium': 4, 
                'hard': 3,
                'infeasible': 5
            },
            'metrics_evaluated': [
                'Final Pass Rate',
                'Delivery Rate', 
                'Hard Constraint Pass Rate (Micro/Macro)',
                'Commonsense Pass Rate',
                'Interactive Plan Repair Success',
                'Optimality Score',
                'Runtime Efficiency'
            ]
        },
        'results': enhanced_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    
    print(f"\n💾 Comprehensive benchmark results saved to: {output_file}")
    
    # Generate summary statistics
    generate_benchmark_summary(enhanced_results)
    
    return output_file

def generate_benchmark_summary(results):
    """Generate summary statistics for the benchmark"""
    
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY STATISTICS")
    print("=" * 60)
    
    # Group by difficulty
    easy_results = [r for r in results if r['difficulty'] == 'Easy']
    medium_results = [r for r in results if r['difficulty'] == 'Medium']
    hard_results = [r for r in results if r['difficulty'] == 'Hard']
    infeasible_results = [r for r in results if r['difficulty'] == 'Infeasible']
    
    def calculate_avg_metrics(result_set, planner_name):
        """Calculate average metrics for a set of results"""
        if not result_set:
            return {}
            
        metrics_key = f'{planner_name.lower()}_metrics'
        total_final_pass = sum(r.get(metrics_key, {}).get('final_pass_rate', 0) for r in result_set)
        total_delivery = sum(r.get(metrics_key, {}).get('delivery_rate', 0) for r in result_set)
        total_hard_micro = sum(r.get(metrics_key, {}).get('hard_constraint_pass_rate_micro', 0) for r in result_set)
        total_optimality = sum(r.get(metrics_key, {}).get('optimality', 0) for r in result_set)
        total_runtime = sum(r.get(metrics_key, {}).get('runtime', 0) for r in result_set)
        
        count = len(result_set)
        return {
            'final_pass_rate': total_final_pass / count,
            'delivery_rate': total_delivery / count,
            'hard_constraint_micro': total_hard_micro / count,
            'optimality': total_optimality / count,
            'avg_runtime': total_runtime / count
        }
    
    # Print summary by difficulty
    for difficulty, result_set in [('Easy', easy_results), ('Medium', medium_results), 
                                   ('Hard', hard_results), ('Infeasible', infeasible_results)]:
        print(f"\n{difficulty.upper()} QUERIES ({len(result_set)} queries):")
        
        for planner in ['Pure_LLM', 'Z3', 'Gurobi']:
            metrics = calculate_avg_metrics(result_set, planner)
            if metrics:
                print(f"  {planner}:")
                print(f"    Final Pass Rate: {metrics['final_pass_rate']:.1%}")
                print(f"    Hard Constraint (Micro): {metrics['hard_constraint_micro']:.1%}")
                print(f"    Optimality: {metrics['optimality']:.2f}")
                print(f"    Avg Runtime: {metrics['avg_runtime']:.2f}s")
    
    print("\n" + "=" * 60)
    print("🎯 READY FOR OVERLEAF TABLE GENERATION")
    print("=" * 60)

if __name__ == "__main__":
    # Run the comprehensive benchmark
    output_file = run_comprehensive_benchmark()
    
    print("\n✅ Comprehensive benchmark completed!")
    print(f"📄 Results file: {output_file}")
    print("🔗 Use this file to generate the Overleaf table")
