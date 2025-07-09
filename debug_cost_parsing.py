#!/usr/bin/env python3
import json
import re

# Load the benchmark results to get the actual Z3 plan text
with open('benchmark_results/benchmark_20250709_173723.json', 'r') as f:
    data = json.load(f)

z3_plan = data['z3_result']['plan_text']
print("Z3 Plan Text:")
print("=" * 50)
print(z3_plan)
print("=" * 50)

# Test cost parsing
cost_patterns = [
    r'\*\*Total Cost:\*\*\s*[€â‚¬](\d+(?:\.\d+)?)',
    r'-?\s*TOTAL COST:\s*[€â‚¬](\d+(?:\.\d+)?)',
    r'TOTAL COST:\s*[€â‚¬](\d+(?:\.\d+)?)',
    r'- TOTAL COST:\s*[€â‚¬](\d+(?:\.\d+)?)',
    r'- TOTAL COST: â‚¬(\d+(?:\.\d+)?)'  # Exact match
]

print("\nTesting cost patterns:")
for i, pattern in enumerate(cost_patterns):
    match = re.search(pattern, z3_plan)
    print(f"Pattern {i+1}: {pattern}")
    print(f"  Match: {match.group(1) if match else 'None'}")
