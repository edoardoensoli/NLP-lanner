#!/usr/bin/env python3
"""
Debug script to identify why Z3 has different constraint evaluation than Gurobi
"""

import json
from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

def debug_constraint_evaluation():
    """Debug the constraint evaluation discrepancy between Z3 and Gurobi"""
    
    # Load the benchmark results to get the actual outputs
    with open('benchmark_results/benchmark_20250709_173723.json', 'r') as f:
        data = json.load(f)
    
    print("=== DEBUGGING CONSTRAINT EVALUATION DISCREPANCY ===")
    print(f"Query: {data['query']}")
    print()
    
    # Get Z3 and Gurobi results
    z3_result = data['z3_result']
    gurobi_result = data['gurobi_result']
    
    print("Z3 Result:")
    print(f"  Result Type: {z3_result['status']}")
    print(f"  Cost: {z3_result.get('total_cost', 'N/A')}")
    print(f"  Plan Text Preview: {z3_result['plan_text'][:200]}...")
    print()
    
    print("Gurobi Result:")
    print(f"  Result Type: {gurobi_result['status']}")
    print(f"  Cost: {gurobi_result.get('total_cost', 'N/A')}")
    print(f"  Plan Text Preview: {gurobi_result['plan_text'][:200]}...")
    print()
    
    # Create PlannerResult objects
    z3_planner_result = PlannerResult(
        planner_name="Z3",
        query=data['query'],
        result_type=z3_result['status'],
        plan_text=z3_result['plan_text'],
        suggestions=z3_result.get('suggestions', None)
    )
    z3_planner_result.runtime = z3_result.get('execution_time', 0)
    
    gurobi_planner_result = PlannerResult(
        planner_name="Gurobi",
        query=data['query'],
        result_type=gurobi_result['status'],
        plan_text=gurobi_result['plan_text'],
        suggestions=gurobi_result.get('suggestions', None)
    )
    gurobi_planner_result.runtime = gurobi_result.get('execution_time', 0)
    
    # Evaluate both with the metrics system
    metrics = SkiPlannerMetrics()
    
    print("=== Z3 PLAN PARSING ===")
    print(f"Resort: {z3_planner_result.resort_name}")
    print(f"Cost: {z3_planner_result.cost}")
    print(f"Accommodation: {z3_planner_result.accommodation}")
    print(f"Car Rental: {z3_planner_result.car_rental}")
    print(f"Equipment Rental: {z3_planner_result.equipment_rental}")
    print()
    
    print("=== GUROBI PLAN PARSING ===")
    print(f"Resort: {gurobi_planner_result.resort_name}")
    print(f"Cost: {gurobi_planner_result.cost}")
    print(f"Accommodation: {gurobi_planner_result.accommodation}")
    print(f"Car Rental: {gurobi_planner_result.car_rental}")
    print(f"Equipment Rental: {gurobi_planner_result.equipment_rental}")
    print()
    
    # Evaluate metrics
    z3_metrics = metrics.evaluate_single_result(z3_planner_result)
    gurobi_metrics = metrics.evaluate_single_result(gurobi_planner_result)
    
    print("=== Z3 CONSTRAINT EVALUATION ===")
    print(f"Hard Constraint Micro: {z3_metrics['hard_constraint_micro'].value:.2f}")
    print(f"Hard Constraint Macro: {z3_metrics['hard_constraint_macro'].value:.2f}")
    print(f"Details: {z3_metrics['hard_constraint_micro'].details}")
    print()
    
    print("=== GUROBI CONSTRAINT EVALUATION ===")
    print(f"Hard Constraint Micro: {gurobi_metrics['hard_constraint_micro'].value:.2f}")
    print(f"Hard Constraint Macro: {gurobi_metrics['hard_constraint_macro'].value:.2f}")
    print(f"Details: {gurobi_metrics['hard_constraint_micro'].details}")
    print()
    
    # Check query parameter extraction
    query_params = metrics.query_extractor.extract_parameters(data['query'])
    print("=== QUERY PARAMETER EXTRACTION ===")
    for key, value in query_params.items():
        print(f"  {key}: {value}")
    print()

if __name__ == "__main__":
    debug_constraint_evaluation()
