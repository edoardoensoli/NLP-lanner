#!/usr/bin/env python3
"""Test script for the fixed metrics system"""

from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

def test_fixed_metrics():
    print('=== TESTING FIXED METRICS SYSTEM ===')
    
    # Initialize metrics system
    metrics = SkiPlannerMetrics()
    
    # Test with a realistic Gurobi result that has budget constraint violation
    print('\n1. Test Gurobi Result with Budget Violation:')
    result_gurobi = PlannerResult(
        planner_name='Gurobi',
        query='Plan a 3-day ski trip to Livigno for 2 people with a budget of 1000 euros, need car rental and equipment',
        result_type='optimal',
        plan_text='''### GUROBI SKI TRIP PLAN
**Selected Resort:** Livigno Carosello/Mottolino
**Total Cost:** €1200.00
#### Car Rental
- Rented Car: Sedan (Petrol) for 3 days.
#### Equipment Rental
- Rented Equipment for 2 people for 3 days:
  - Skis (x2)
  - Boots (x2)
''',
        cost=1200.0,  # Over budget!
        runtime=2.5
    )
    
    # Test the evaluate_all_metrics method with fixes
    eval_result_gurobi = metrics.evaluate_all_metrics(
        query=result_gurobi.query,
        plan_text=result_gurobi.plan_text,
        execution_time=result_gurobi.runtime,
        success=True,
        feasible=True,
        status='optimal',
        total_cost=result_gurobi.cost,
        model_used='Gurobi'
    )
    
    print('Gurobi Constraint Details:')
    for constraint, value in eval_result_gurobi['constraint_details'].items():
        print(f'  {constraint}: {value}')
    
    print(f'Micro Pass Rate: {eval_result_gurobi["hard_constraint_pass_rate_micro"]:.2f}')
    print(f'Macro Pass Rate: {eval_result_gurobi["hard_constraint_pass_rate_macro"]:.2f}')
    print(f'Cost: €{eval_result_gurobi["cost"]}')
    print(f'Optimality Score: {eval_result_gurobi["optimality"]:.2f}')
    
    # Test with Z3 result that misses car rental
    print('\n2. Test Z3 Result Missing Car Rental:')
    result_z3 = PlannerResult(
        planner_name='Z3',
        query='Plan a 3-day ski trip to Zermatt for 2 people with a budget of 2000 euros, need car rental and equipment',
        result_type='optimal',
        plan_text='''Z3 SKI TRIP PLAN:
SELECTED RESORT: Zermatt Matterhorn
TOTAL COST: €1800.00
EQUIPMENT RENTAL:
- Skis: €49/day
- Boots: €26/day
CAR RENTAL:
- No car rental selected
''',
        cost=1800.0,
        runtime=3.2
    )
    
    eval_result_z3 = metrics.evaluate_all_metrics(
        query=result_z3.query,
        plan_text=result_z3.plan_text,
        execution_time=result_z3.runtime,
        success=True,
        feasible=True,
        status='optimal',
        total_cost=result_z3.cost,
        model_used='Z3'
    )
    
    print('Z3 Constraint Details:')
    for constraint, value in eval_result_z3['constraint_details'].items():
        print(f'  {constraint}: {value}')
    
    print(f'Micro Pass Rate: {eval_result_z3["hard_constraint_pass_rate_micro"]:.2f}')
    print(f'Macro Pass Rate: {eval_result_z3["hard_constraint_pass_rate_macro"]:.2f}')
    print(f'Cost: €{eval_result_z3["cost"]}')
    print(f'Optimality Score: {eval_result_z3["optimality"]:.2f}')
    
    # Test cost baseline functionality
    print('\n3. Test Destination Cost Baselines:')
    test_destinations = ['Livigno', 'Zermatt', 'Switzerland', 'Italy', 'Unknown Resort']
    for dest in test_destinations:
        baseline = metrics._get_destination_cost_baseline(dest, 3, 2)  # 3 days, 2 people
        print(f'  {dest}: €{baseline:.2f} baseline cost')
    
    return {
        'gurobi_budget_passed': eval_result_gurobi['constraint_details']['budget_constraint'],
        'gurobi_micro_rate': eval_result_gurobi['hard_constraint_pass_rate_micro'],
        'z3_car_passed': eval_result_z3['constraint_details']['car_constraint'],
        'z3_micro_rate': eval_result_z3['hard_constraint_pass_rate_micro']
    }

if __name__ == "__main__":
    results = test_fixed_metrics()
    print('\n=== SUMMARY ===')
    print(f'Gurobi budget constraint properly detected as: {results["gurobi_budget_passed"]} (should be False)')
    print(f'Gurobi micro pass rate: {results["gurobi_micro_rate"]:.2f}')
    print(f'Z3 car constraint properly detected as: {results["z3_car_passed"]} (should be False)')
    print(f'Z3 micro pass rate: {results["z3_micro_rate"]:.2f}')
