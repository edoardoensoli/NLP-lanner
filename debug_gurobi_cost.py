#!/usr/bin/env python3
import json
import re

# Load the benchmark results to get the actual Gurobi plan text
with open('benchmark_results/benchmark_20250709_173723.json', 'r') as f:
    data = json.load(f)

gurobi_plan = data['gurobi_result']['plan_text']
print("Gurobi Plan Text:")
print("=" * 50)
print(gurobi_plan)
print("=" * 50)

# Test cost parsing for Gurobi
cost_patterns = [
    r'\*\*Total Cost:\*\*\s*[€â‚¬](\d+(?:\.\d+)?)',
    r'- TOTAL COST: â‚¬(\d+(?:\.\d+)?)',
    r'\*\*Total Cost:\*\* â‚¬(\d+(?:\.\d+)?)'  # Exact match
]

print("\nTesting Gurobi cost patterns:")
for i, pattern in enumerate(cost_patterns):
    match = re.search(pattern, gurobi_plan)
    print(f"Pattern {i+1}: {pattern}")
    print(f"  Match: {match.group(1) if match else 'None'}")
