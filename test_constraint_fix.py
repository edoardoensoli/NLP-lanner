#!/usr/bin/env python3
"""Test the fixed constraint evaluation with required vs optional services"""

import sys
import os
sys.path.append(os.getcwd())

from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

def test_required_vs_optional():
    print('=== TESTING REQUIRED VS OPTIONAL CONSTRAINTS ===')
    
    metrics = SkiPlannerMetrics()
    
    # Test 1: Query with NO car/equipment requirements (like the benchmark)
    print('\n1. Query WITHOUT car/equipment requirements:')
    result1 = metrics.evaluate_all_metrics(
        query='Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros',
        plan_text='Z3 SKI TRIP PLAN:\nSELECTED RESORT: Livigno\nEQUIPMENT RENTAL:\n- No equipment rental selected\nCAR RENTAL:\n- No car rental selected\nTOTAL COST: €729.00',
        execution_time=6.18,
        success=True,
        feasible=True,
        status='optimal',
        total_cost=729.0,
        model_used='Z3'
    )
    
    print('  Constraints in result:')
    for constraint, value in result1['constraint_details'].items():
        print(f'    {constraint}: {value}')
    print(f'  Micro pass rate: {result1["hard_constraint_pass_rate_micro"]:.2f}')
    print(f'  Macro pass rate: {result1["hard_constraint_pass_rate_macro"]:.2f}')
    
    # Test 2: Query WITH car/equipment requirements
    print('\n2. Query WITH car/equipment requirements:')
    result2 = metrics.evaluate_all_metrics(
        query='Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment',
        plan_text='Z3 SKI TRIP PLAN:\nSELECTED RESORT: Livigno\nEQUIPMENT RENTAL:\n- No equipment rental selected\nCAR RENTAL:\n- No car rental selected\nTOTAL COST: €729.00',
        execution_time=6.18,
        success=True,
        feasible=True,
        status='optimal',
        total_cost=729.0,
        model_used='Z3'
    )
    
    print('  Constraints in result:')
    for constraint, value in result2['constraint_details'].items():
        print(f'    {constraint}: {value}')
    print(f'  Micro pass rate: {result2["hard_constraint_pass_rate_micro"]:.2f}')
    print(f'  Macro pass rate: {result2["hard_constraint_pass_rate_macro"]:.2f}')
    
    # Test 3: Query WITH requirements that ARE satisfied
    print('\n3. Query WITH requirements that ARE satisfied:')
    result3 = metrics.evaluate_all_metrics(
        query='Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment',
        plan_text='Z3 SKI TRIP PLAN:\nSELECTED RESORT: Livigno\nEQUIPMENT RENTAL:\n- Skis: €49/day\n- Boots: €26/day\nCAR RENTAL:\n- Rented Car: Sedan (Petrol) for 3 days\nTOTAL COST: €1200.00',
        execution_time=6.18,
        success=True,
        feasible=True,
        status='optimal',
        total_cost=1200.0,
        model_used='Z3'
    )
    
    print('  Constraints in result:')
    for constraint, value in result3['constraint_details'].items():
        print(f'    {constraint}: {value}')
    print(f'  Micro pass rate: {result3["hard_constraint_pass_rate_micro"]:.2f}')
    print(f'  Macro pass rate: {result3["hard_constraint_pass_rate_macro"]:.2f}')

if __name__ == "__main__":
    test_required_vs_optional()
