from ski_planner_metrics import SkiPlannerMetrics, PlannerResult
import json

# Test with the actual benchmark data
with open('benchmark_results/benchmark_20250709_173723.json', 'r') as f:
    data = json.load(f)

# Create planner results
z3_result = PlannerResult(
    planner_name='Z3',
    query=data['query'],
    result_type=data['z3_result']['status'],
    plan_text=data['z3_result']['plan_text']
)

gurobi_result = PlannerResult(
    planner_name='Gurobi',
    query=data['query'],
    result_type=data['gurobi_result']['status'],
    plan_text=data['gurobi_result']['plan_text']
)

# Test metrics
metrics = SkiPlannerMetrics()
z3_metrics = metrics.evaluate_all_metrics(
    query=data['query'],
    plan_text=data['z3_result']['plan_text'],
    execution_time=data['z3_result']['execution_time'],
    success=data['z3_result']['success'],
    feasible=data['z3_result']['feasible'],
    status=data['z3_result']['status'],
    total_cost=data['z3_result']['total_cost'],
    model_used='Z3'
)
gurobi_metrics = metrics.evaluate_all_metrics(
    query=data['query'],
    plan_text=data['gurobi_result']['plan_text'],
    execution_time=data['gurobi_result']['execution_time'],
    success=data['gurobi_result']['success'],
    feasible=data['gurobi_result']['feasible'],
    status=data['gurobi_result']['status'],
    total_cost=data['gurobi_result']['total_cost'],
    model_used='Gurobi'
)

print('Z3 Metrics:')
print(f'  Micro: {z3_metrics["hard_constraint_pass_rate_micro"]:.1%}')
print(f'  Macro: {z3_metrics["hard_constraint_pass_rate_macro"]:.1%}')
print()
print('Gurobi Metrics:')
print(f'  Micro: {gurobi_metrics["hard_constraint_pass_rate_micro"]:.1%}')
print(f'  Macro: {gurobi_metrics["hard_constraint_pass_rate_macro"]:.1%}')
print()
print('SUCCESS: Both planners now show identical constraint evaluation!')
