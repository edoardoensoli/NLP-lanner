#!/usr/bin/env python3
"""
Benchmark script for ski trip planners.
Tests LLM, Z3, and Gurobi approaches individually with metrics evaluation.
Usage: python ski_planner_benchmark.py --method [llm|z3|gurobi] --queries queries.txt
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the metrics system
from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

# Import the different planners
try:
    from test_skiplanner_z3_extended import pipeline_ski_z3_extended
except ImportError:
    print("Warning: Z3 planner not available")
    pipeline_ski_z3_extended = None

try:
    from test_skiplanner_gurobi_extended import pipeline_ski_gurobi_extended
except ImportError:
    print("Warning: Gurobi planner not available")
    pipeline_ski_gurobi_extended = None

try:
    from test_skiplanner_llm_extended import pipeline_ski_extended as pipeline_ski_llm_extended
except ImportError:
    print("Warning: LLM planner not available")
    pipeline_ski_llm_extended = None


class SkiPlannerBenchmark:
    """Benchmark runner for ski planner methods"""
    
    def __init__(self, method: str):
        self.method = method.lower()
        self.metrics = SkiPlannerMetrics()
        self.results = []
        
        # Validate method
        if self.method not in ['llm', 'z3', 'gurobi']:
            raise ValueError(f"Invalid method: {method}. Must be one of: llm, z3, gurobi")
        
        # Check if method is available
        if self.method == 'z3' and pipeline_ski_z3_extended is None:
            raise ImportError("Z3 planner not available")
        elif self.method == 'gurobi' and pipeline_ski_gurobi_extended is None:
            raise ImportError("Gurobi planner not available")
        elif self.method == 'llm' and pipeline_ski_llm_extended is None:
            raise ImportError("LLM planner not available")
        
        # Load datasets for LLM method
        self.llm_datasets = None
        self.llm_client = None
        if self.method == 'llm':
            try:
                import pandas as pd
                from test_skiplanner_llm_extended import LLMClient
                
                # Load datasets
                resorts_df = pd.read_csv('dataset_ski/resorts/resorts.csv') if os.path.exists('dataset_ski/resorts/resorts.csv') else None
                car_df = pd.read_csv('dataset_ski/car/ski_car.csv') if os.path.exists('dataset_ski/car/ski_car.csv') else None
                equipment_df = pd.read_csv('dataset_ski/rent/ski_rent.csv') if os.path.exists('dataset_ski/rent/ski_rent.csv') else None
                slopes_df = pd.read_csv('dataset_ski/slopes/ski_slopes.csv') if os.path.exists('dataset_ski/slopes/ski_slopes.csv') else None
                
                self.llm_datasets = {
                    'resorts': resorts_df,
                    'cars': car_df,
                    'equipment': equipment_df,
                    'slopes': slopes_df
                }
                
                # Initialize LLM client
                self.llm_client = LLMClient()
                
            except Exception as e:
                raise ImportError(f"Failed to initialize LLM planner: {e}")
    
    def load_queries(self, queries_file: str) -> List[str]:
        """Load queries from file"""
        if not os.path.exists(queries_file):
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    queries.append(line)
        
        print(f"Loaded {len(queries)} queries from {queries_file}")
        return queries
    
    def run_single_query(self, query: str, query_index: int) -> PlannerResult:
        """Run a single query with the selected method"""
        print(f"\n{'='*80}")
        print(f"QUERY {query_index + 1}: {self.method.upper()}")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Method: {self.method}")
        
        start_time = time.time()
        result_type = "failed"
        plan_text = ""
        suggestions = ""
        cost = None
        resort_name = None
        
        try:
            if self.method == 'z3':
                result = pipeline_ski_z3_extended(query, verbose=True)
            elif self.method == 'gurobi':
                result = pipeline_ski_gurobi_extended(query, verbose=True)
            elif self.method == 'llm':
                result = pipeline_ski_llm_extended(
                    query, 
                    self.llm_client,
                    self.llm_datasets['resorts'],
                    self.llm_datasets['cars'],
                    self.llm_datasets['equipment'],
                    self.llm_datasets['slopes']
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            runtime = time.time() - start_time
            
            # Parse result
            if result is None:
                result_type = "failed"
                plan_text = "No result returned"
            elif isinstance(result, str):
                plan_text = result
                if "INFEASIBLE" in result:
                    result_type = "infeasible"
                    # Extract suggestions if available
                    if "Suggestion:" in result:
                        suggestions = result.split("Suggestion:")[-1].strip()
                elif any(keyword in result for keyword in ["PLAN", "COST", "RESORT"]):
                    result_type = "optimal"
                    # Extract cost if available
                    import re
                    cost_match = re.search(r'TOTAL COST:\s*€(\d+(?:\.\d+)?)', result)
                    if cost_match:
                        cost = float(cost_match.group(1))
                    
                    # Extract resort name if available
                    resort_match = re.search(r'(?:Selected|SELECTED) (?:Resort|RESORT):\s*(.+?)(?:\n|$)', result)
                    if resort_match:
                        resort_name = resort_match.group(1).strip()
                        if resort_name.startswith('-'):
                            resort_name = resort_name[1:].strip()
                else:
                    result_type = "failed"
            else:
                result_type = "failed"
                plan_text = f"Unexpected result type: {type(result)}"
            
        except Exception as e:
            runtime = time.time() - start_time
            result_type = "failed"
            plan_text = f"Error: {str(e)}"
            suggestions = f"Technical error occurred: {str(e)}"
            print(f"❌ Error processing query: {e}")
            traceback.print_exc()
        
        # Create PlannerResult
        planner_result = PlannerResult(
            planner_name=self.method.upper(),
            query=query,
            result_type=result_type,
            plan_text=plan_text,
            cost=cost,
            runtime=runtime,
            resort_name=resort_name,
            suggestions=suggestions
        )
        
        print(f"\n📊 RESULT SUMMARY:")
        print(f"- Status: {result_type}")
        print(f"- Runtime: {runtime:.2f}s")
        if cost:
            print(f"- Cost: €{cost:.2f}")
        if resort_name:
            print(f"- Resort: {resort_name}")
        
        return planner_result
    
    def run_benchmark(self, queries: List[str]) -> Dict[str, Any]:
        """Run benchmark on all queries"""
        print(f"\n🚀 STARTING BENCHMARK: {self.method.upper()}")
        print(f"Total queries: {len(queries)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            try:
                result = self.run_single_query(query, i)
                self.metrics.add_result(result)
                self.results.append(result)
            except KeyboardInterrupt:
                print(f"\n⚠️ Benchmark interrupted by user at query {i+1}")
                break
            except Exception as e:
                print(f"❌ Failed to process query {i+1}: {e}")
                # Create a failed result
                failed_result = PlannerResult(
                    planner_name=self.method.upper(),
                    query=query,
                    result_type="failed",
                    plan_text=f"Benchmark error: {str(e)}",
                    runtime=0.0
                )
                self.metrics.add_result(failed_result)
                self.results.append(failed_result)
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        successful_queries = len([r for r in self.results if r.result_type == "optimal"])
        infeasible_queries = len([r for r in self.results if r.result_type == "infeasible"])
        failed_queries = len([r for r in self.results if r.result_type == "failed"])
        
        total_cost = sum(r.cost for r in self.results if r.cost is not None)
        avg_runtime = sum(r.runtime for r in self.results if r.runtime is not None) / len(self.results) if self.results else 0
        
        summary = {
            'method': self.method,
            'total_queries': len(queries),
            'processed_queries': len(self.results),
            'successful_queries': successful_queries,
            'infeasible_queries': infeasible_queries,
            'failed_queries': failed_queries,
            'total_runtime': total_time,
            'average_runtime_per_query': avg_runtime,
            'total_cost': total_cost,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n🏁 BENCHMARK COMPLETED: {self.method.upper()}")
        print(f"- Total runtime: {total_time:.2f}s")
        print(f"- Processed queries: {len(self.results)}/{len(queries)}")
        print(f"- Successful: {successful_queries}")
        print(f"- Infeasible: {infeasible_queries}")
        print(f"- Failed: {failed_queries}")
        print(f"- Success rate: {successful_queries/len(self.results)*100:.1f}%")
        
        return summary
    
    def save_results(self, output_dir: str = "results") -> str:
        """Save benchmark results to files"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"ski_planner_benchmark_{self.method}_{timestamp}"
        
        # Save detailed report
        report_file = os.path.join(output_dir, f"{base_filename}_report.md")
        report = self.metrics.generate_report(report_file)
        
        # Save CSV data
        csv_file = os.path.join(output_dir, f"{base_filename}_data.csv")
        self.metrics.export_to_csv(csv_file)
        
        # Save JSON summary
        json_file = os.path.join(output_dir, f"{base_filename}_summary.json")
        aggregated_metrics = self.metrics.get_aggregated_metrics(self.method.upper())
        
        summary_data = {
            'method': self.method,
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(self.results),
            'aggregated_metrics': aggregated_metrics,
            'individual_results': [
                {
                    'query': r.query,
                    'result_type': r.result_type,
                    'cost': r.cost,
                    'runtime': r.runtime,
                    'resort_name': r.resort_name
                }
                for r in self.results
            ]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 RESULTS SAVED:")
        print(f"- Report: {report_file}")
        print(f"- Data: {csv_file}")
        print(f"- Summary: {json_file}")
        
        return base_filename


def create_sample_queries_file():
    """Create a sample queries file for testing"""
    sample_queries = [
        "Plan a 3-day ski trip to Italy for 2 people with budget 1500 euros. We need intermediate slopes.",
        "Plan a 5-day ski trip to Switzerland for 4 people with budget 3000 euros. We need a SUV with diesel fuel and ski equipment.",
        "Plan a 2-day ski trip to Austria for 1 person with budget 800 euros. We prefer beginner slopes.",
        "Plan a 4-day ski trip to France for 3 people with budget 2500 euros. We need car rental and equipment including skis and boots.",
        "Plan a 7-day ski trip to Italy for 2 people with budget 5000 euros. We need luxury accommodation and advanced slopes.",
        "Plan a 1-day ski trip to Switzerland for 6 people with budget 500 euros.",  # Should be infeasible
        "Plan a 3-day ski trip to Val Gardena for 2 people with budget 2000 euros. We need SUV rental and full ski equipment.",
        "Plan a 4-day ski trip to Zermatt for 2 people with budget 4000 euros. We need electric car and intermediate slopes.",
    ]
    
    with open('sample_queries.txt', 'w', encoding='utf-8') as f:
        f.write("# Sample ski trip queries for benchmarking\n")
        f.write("# Format: one query per line, lines starting with # are comments\n\n")
        for query in sample_queries:
            f.write(f"{query}\n")
    
    print("Created sample_queries.txt with example queries")


def main():
    """Main function to run the benchmark"""
    parser = argparse.ArgumentParser(description="Benchmark ski trip planners")
    parser.add_argument('--method', '-m', choices=['llm', 'z3', 'gurobi'], required=True,
                       help='Planning method to test')
    parser.add_argument('--queries', '-q', type=str, required=True,
                       help='File containing queries to test (one per line)')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample queries file and exit')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_queries_file()
        return
    
    try:
        # Initialize benchmark
        benchmark = SkiPlannerBenchmark(args.method)
        
        # Load queries
        queries = benchmark.load_queries(args.queries)
        
        if not queries:
            print("❌ No valid queries found in the file")
            return
        
        # Run benchmark
        summary = benchmark.run_benchmark(queries)
        
        # Save results
        base_filename = benchmark.save_results(args.output)
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"Results saved with base filename: {base_filename}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
