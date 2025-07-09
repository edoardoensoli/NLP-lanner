#!/usr/bin/env python3
"""
Test script to verify micro/macro hard constraint calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

def test_micro_macro_logic():
    """Test the micro and macro hard constraint calculations"""
    
    print("🧪 Testing Micro vs Macro Hard Constraint Calculations")
    print("=" * 60)
    
    metrics = SkiPlannerMetrics()
    
    # Test Case 1: All constraints satisfied (budget, car, equipment)
    print("\n📋 Test Case 1: All constraints satisfied")
    print("-" * 40)
    
    result1 = PlannerResult(
        planner_name="Z3",
        query="Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment",
        result_type="optimal",
        plan_text="""### Z3 SKI TRIP PLAN
**Query:** Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment
**Result:** Optimal solution found!
**Selected Resort:** Livigno Resort
**Total Cost:** €1400.00
**Cost Breakdown:**
- Accommodation: €900.00
- Car Rental: €300.00
- Equipment Rental: €200.00

#### Accommodation
- Stay at Livigno Resort for 3 days.

#### Car Rental
- Rented Car: Toyota RAV4 (SUV, Petrol) for 3 days.

#### Equipment Rental
- Rented Equipment for 2 people for 3 days:
  - Skis (x2)
  - Boots (x2)
  - Helmet (x2)
  - Poles (x2)
""",
        runtime=2.3
    )
    
    metrics_result1 = metrics.evaluate_single_result(result1)
    
    print(f"Query: {result1.query}")
    print(f"Total Cost: €{result1.cost:.2f}")
    print(f"Micro Hard Constraint Pass Rate: {metrics_result1['hard_constraint_micro'].value:.2f}")
    print(f"Macro Hard Constraint Pass Rate: {metrics_result1['hard_constraint_macro'].value:.2f}")
    print(f"Micro Details: {metrics_result1['hard_constraint_micro'].details}")
    print(f"Macro Details: {metrics_result1['hard_constraint_macro'].details}")
    
    # Test Case 2: Budget constraint violated (but other constraints satisfied)
    print("\n📋 Test Case 2: Budget constraint violated")
    print("-" * 40)
    
    result2 = PlannerResult(
        planner_name="Gurobi",
        query="Plan a 3-day ski trip to Livigno for 2 people with a budget of 1200 euros, need car rental and equipment",
        result_type="optimal",
        plan_text="""### GUROBI SKI TRIP PLAN
**Query:** Plan a 3-day ski trip to Livigno for 2 people with a budget of 1200 euros, need car rental and equipment
**Result:** Optimal solution found!
**Selected Resort:** Livigno Resort
**Total Cost:** €1400.00
**Cost Breakdown:**
- Accommodation: €900.00
- Car Rental: €300.00
- Equipment Rental: €200.00

#### Accommodation
- Stay at Livigno Resort for 3 days.

#### Car Rental
- Rented Car: Toyota RAV4 (SUV, Petrol) for 3 days.

#### Equipment Rental
- Rented Equipment for 2 people for 3 days:
  - Skis (x2)
  - Boots (x2)
  - Helmet (x2)
  - Poles (x2)
""",
        runtime=2.1
    )
    
    metrics_result2 = metrics.evaluate_single_result(result2)
    
    print(f"Query: {result2.query}")
    print(f"Total Cost: €{result2.cost:.2f}")
    print(f"Micro Hard Constraint Pass Rate: {metrics_result2['hard_constraint_micro'].value:.2f}")
    print(f"Macro Hard Constraint Pass Rate: {metrics_result2['hard_constraint_macro'].value:.2f}")
    print(f"Micro Details: {metrics_result2['hard_constraint_micro'].details}")
    print(f"Macro Details: {metrics_result2['hard_constraint_macro'].details}")
    
    # Test Case 3: Car rental missing (but budget OK)
    print("\n📋 Test Case 3: Car rental missing")
    print("-" * 40)
    
    result3 = PlannerResult(
        planner_name="Z3",
        query="Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment",
        result_type="optimal",
        plan_text="""### Z3 SKI TRIP PLAN
**Query:** Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment
**Result:** Optimal solution found!
**Selected Resort:** Livigno Resort
**Total Cost:** €1100.00
**Cost Breakdown:**
- Accommodation: €900.00
- Equipment Rental: €200.00

#### Accommodation
- Stay at Livigno Resort for 3 days.

#### Car Rental
- No car rental

#### Equipment Rental
- Rented Equipment for 2 people for 3 days:
  - Skis (x2)
  - Boots (x2)
  - Helmet (x2)
  - Poles (x2)
""",
        runtime=1.8
    )
    
    metrics_result3 = metrics.evaluate_single_result(result3)
    
    print(f"Query: {result3.query}")
    print(f"Total Cost: €{result3.cost:.2f}")
    print(f"Micro Hard Constraint Pass Rate: {metrics_result3['hard_constraint_micro'].value:.2f}")
    print(f"Macro Hard Constraint Pass Rate: {metrics_result3['hard_constraint_macro'].value:.2f}")
    print(f"Micro Details: {metrics_result3['hard_constraint_micro'].details}")
    print(f"Macro Details: {metrics_result3['hard_constraint_macro'].details}")
    
    # Test Case 4: Infeasible query (no optimal solution)
    print("\n📋 Test Case 4: Infeasible query")
    print("-" * 40)
    
    result4 = PlannerResult(
        planner_name="Gurobi",
        query="Plan a 2-day ski trip to Zermatt for 4 people with a budget of 100 euros",
        result_type="infeasible",
        plan_text="The query is infeasible.",
        suggestions="Suggestion: try relaxing some constraints, for example, the budget of €100 is too low for the requested trip.",
        runtime=1.5
    )
    
    metrics_result4 = metrics.evaluate_single_result(result4)
    
    print(f"Query: {result4.query}")
    print(f"Result Type: {result4.result_type}")
    print(f"Micro Hard Constraint Pass Rate: {metrics_result4['hard_constraint_micro'].value:.2f}")
    print(f"Macro Hard Constraint Pass Rate: {metrics_result4['hard_constraint_macro'].value:.2f}")
    print(f"Micro Details: {metrics_result4['hard_constraint_micro'].details}")
    print(f"Macro Details: {metrics_result4['hard_constraint_macro'].details}")
    
    print("\n✅ Test Summary:")
    print("- Case 1: All constraints satisfied → Micro: 1.0, Macro: 1.0")
    print("- Case 2: Budget violated → Micro: 0.75 (3/4), Macro: 0.0")
    print("- Case 3: Car rental missing → Micro: 0.75 (3/4), Macro: 0.0")
    print("- Case 4: Infeasible → Micro: 0.0, Macro: 0.0")
    
    print("\n🎯 Expected Behavior:")
    print("- Micro: Individual constraint pass rate (0.0-1.0)")
    print("- Macro: All constraints must pass (0.0 or 1.0)")

if __name__ == "__main__":
    test_micro_macro_logic()
