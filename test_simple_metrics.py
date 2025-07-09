"""
Simple test script to demonstrate the ski planner metrics system.
This script creates various test scenarios to show how the metrics work.
"""

from ski_planner_metrics import SkiPlannerMetrics, PlannerResult

def test_simple_metrics():
    """Test the metrics system with simple examples"""
    print("=== SKI PLANNER METRICS TEST ===\n")
    
    # Initialize the metrics system
    metrics = SkiPlannerMetrics()
    
    # Test Case 1: Perfect Gurobi result
    print("1. Testing PERFECT GUROBI RESULT:")
    result1 = PlannerResult(
        planner_name="Gurobi",
        query="Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment",
        result_type="optimal",
        plan_text="""### GUROBI SKI TRIP PLAN
**Selected Resort:** Livigno Resort
**Total Cost:** €1200.00

#### Accommodation
- Stay at Livigno Resort for 3 days.

#### Car Rental
- Rented Car: Toyota RAV4 (SUV, Petrol) for 3 days.

#### Equipment Rental
- Rented Equipment for 2 people for 3 days: Skis, Boots, Helmet, Poles
""",
        runtime=2.5
    )
    
    metrics.add_result(result1)
    single_metrics = metrics.evaluate_single_result(result1)
    
    print(f"Query: {result1.query}")
    print(f"Result: {result1.result_type}")
    if result1.cost:
        print(f"Cost: €{result1.cost:.2f}")
    else:
        print("Cost: Not extracted from plan text")
    print(f"Runtime: {result1.runtime:.2f}s")
    print("\nMetrics:")
    for name, metric in single_metrics.items():
        print(f"  {name}: {metric}")
    print("\n" + "="*50 + "\n")
    
    # Test Case 2: Infeasible Z3 result
    print("2. Testing INFEASIBLE Z3 RESULT:")
    result2 = PlannerResult(
        planner_name="Z3",
        query="Plan a 2-day ski trip to Zermatt for 4 people with a budget of 300 euros",
        result_type="infeasible",
        plan_text="The query is infeasible due to budget constraints.",
        suggestions="Try increasing the budget to at least €800 or reduce the number of people to 2. Consider staying for only 1 day.",
        runtime=1.2
    )
    
    metrics.add_result(result2)
    single_metrics = metrics.evaluate_single_result(result2)
    
    print(f"Query: {result2.query}")
    print(f"Result: {result2.result_type}")
    print(f"Runtime: {result2.runtime:.2f}s")
    print(f"Suggestions: {result2.suggestions}")
    print("\nMetrics:")
    for name, metric in single_metrics.items():
        print(f"  {name}: {metric}")
    print("\n" + "="*50 + "\n")
    
    # Test Case 3: Failed LLM result
    print("3. Testing FAILED LLM RESULT:")
    result3 = PlannerResult(
        planner_name="Pure LLM",
        query="Plan a 5-day ski trip to Davos for 3 people with unlimited budget",
        result_type="failed",
        plan_text="Error: System timeout occurred during plan generation.",
        suggestions="",
        runtime=35.0
    )
    
    metrics.add_result(result3)
    single_metrics = metrics.evaluate_single_result(result3)
    
    print(f"Query: {result3.query}")
    print(f"Result: {result3.result_type}")
    print(f"Runtime: {result3.runtime:.2f}s")
    print("\nMetrics:")
    for name, metric in single_metrics.items():
        print(f"  {name}: {metric}")
    print("\n" + "="*50 + "\n")
    
    # Test Case 4: Suboptimal but valid result
    print("4. Testing SUBOPTIMAL BUT VALID RESULT:")
    result4 = PlannerResult(
        planner_name="Z3",
        query="Plan a 4-day ski trip to Cortina for 2 people with a budget of 2000 euros, need equipment",
        result_type="optimal",
        plan_text="""### Z3 SKI TRIP PLAN
SELECTED RESORT: Cortina d'Ampezzo
TOTAL COST: €1800.00

ACCOMMODATION:
- Resort: Cortina d'Ampezzo for 4 days

EQUIPMENT RENTAL:
- Rented Equipment for 2 people for 4 days

CAR RENTAL:
- No car rental selected
""",
        runtime=8.5
    )
    
    metrics.add_result(result4)
    single_metrics = metrics.evaluate_single_result(result4)
    
    print(f"Query: {result4.query}")
    print(f"Result: {result4.result_type}")
    if result4.cost:
        print(f"Cost: €{result4.cost:.2f}")
    else:
        print("Cost: Not extracted from plan text")
    print(f"Runtime: {result4.runtime:.2f}s")
    print("\nMetrics:")
    for name, metric in single_metrics.items():
        print(f"  {name}: {metric}")
    print("\n" + "="*50 + "\n")
    
    # Generate aggregated report
    print("5. AGGREGATED REPORT:")
    print("="*50)
    
    # Get aggregated metrics for each planner
    planners = ["Gurobi", "Z3", "Pure LLM"]
    for planner in planners:
        agg_metrics = metrics.get_aggregated_metrics(planner)
        if agg_metrics:
            print(f"\n{planner} SUMMARY:")
            print(f"  Final Pass Rate: {agg_metrics['final_pass_rate']['pass_rate']:.1%}")
            print(f"  Delivery Rate: {agg_metrics['delivery_rate']['pass_rate']:.1%}")
            print(f"  Hard Constraints: {agg_metrics['hard_constraint_pass_rate']['pass_rate']:.1%}")
            print(f"  Repair Success: {agg_metrics['repair_success']['pass_rate']:.1%}")
            print(f"  Runtime Efficiency: {agg_metrics['runtime_efficiency']['pass_rate']:.1%}")
    
    print("\n" + "="*50)
    print("TEST COMPLETE! This demonstrates:")
    print("✓ Perfect constraint satisfaction (Gurobi)")
    print("✓ Infeasible query handling (Z3)")
    print("✓ System failure detection (Pure LLM)")
    print("✓ Suboptimal but valid results (Z3)")
    print("✓ Aggregated metrics across planners")
    print("="*50)

if __name__ == "__main__":
    test_simple_metrics()
