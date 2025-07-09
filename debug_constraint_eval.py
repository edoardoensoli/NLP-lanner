#!/usr/bin/env python3
"""Debug script for constraint evaluation logic"""

import sys
import os
sys.path.append(os.getcwd())

from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

def debug_constraint_evaluation():
    print('=== CONSTRAINT EVALUATION DEBUG ===')
    
    # Recreate the exact Z3 result from the benchmark
    z3_result = PlannerResult(
        planner_name='Z3',
        query='Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros',
        result_type='optimal',
        plan_text='''Z3 SKI TRIP PLAN:

DESTINATION: Livigno
DURATION: 3 days
PEOPLE: 2
BUDGET: €2000

SELECTED RESORT: Livigno  Carosello/?Mottolino

RESORT DETAILS:
- Resort: Livigno  Carosello/?Mottolino
- Price per day: €243
- Beds available: 4
- Rating: 4.6/5

EQUIPMENT RENTAL:
- No equipment rental selected

CAR RENTAL:
- No car rental selected

COST BREAKDOWN:
- Accommodation: €729.00
- Equipment: €0.00
- Car rental: €0.00
- TOTAL COST: €729.00

BUDGET STATUS: ✅ Within budget''',
        cost=729.0,
        runtime=6.18
    )
    
    # Initialize metrics system
    metrics = SkiPlannerMetrics()
    
    # Test the evaluate_all_metrics method
    print('Testing evaluate_all_metrics method...')
    eval_result = metrics.evaluate_all_metrics(
        query=z3_result.query,
        plan_text=z3_result.plan_text,
        execution_time=z3_result.runtime,
        success=True,
        feasible=True,
        status='optimal',
        total_cost=z3_result.cost,
        model_used='Z3'
    )
    
    print(f'Query: "{z3_result.query}"')
    print('Constraint Details from evaluate_all_metrics:')
    for constraint, value in eval_result['constraint_details'].items():
        print(f'  {constraint}: {value}')
    
    # Now let's manually trace through the logic
    print('\n=== MANUAL CONSTRAINT EVALUATION TRACE ===')
    
    # Extract query parameters
    query_params = metrics.query_extractor.extract_parameters(z3_result.query)
    print(f'Extracted query parameters:')
    for key, value in query_params.items():
        if key in ['car_required', 'equipment_required', 'budget']:
            print(f'  {key}: {value}')
    
    # Check individual constraint evaluation logic
    plan_text = z3_result.plan_text
    total_cost = z3_result.cost
    
    print(f'\nPlan text analysis:')
    print(f'  Plan mentions "No car rental selected": {"No car rental selected" in plan_text}')
    print(f'  Plan mentions "No equipment rental selected": {"No equipment rental selected" in plan_text}')
    
    # Budget constraint
    budget_satisfied = True
    if query_params.get('budget') and total_cost > 0:
        budget_satisfied = total_cost <= query_params['budget']
    print(f'  Budget satisfied: {budget_satisfied} (€{total_cost} vs €{query_params.get("budget")})')
    
    # Car constraint evaluation logic from our code
    car_required = query_params.get('car_required', False)
    car_satisfied = True
    if car_required:
        car_satisfied = plan_text and ('car rental' in plan_text.lower() or 'rented car' in plan_text.lower()) and 'no car rental' not in plan_text.lower()
    
    print(f'  Car required: {car_required}')
    print(f'  Car satisfied: {car_satisfied}')
    
    # Equipment constraint evaluation logic from our code
    equipment_required = query_params.get('equipment_required', False)
    equipment_satisfied = True
    if equipment_required:
        equipment_satisfied = plan_text and ('equipment' in plan_text.lower() or 'skis' in plan_text.lower() or 'boots' in plan_text.lower()) and 'no equipment' not in plan_text.lower()
    
    print(f'  Equipment required: {equipment_required}')
    print(f'  Equipment satisfied: {equipment_satisfied}')

if __name__ == "__main__":
    debug_constraint_evaluation()
