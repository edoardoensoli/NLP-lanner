#!/usr/bin/env python3
"""
Display clean LaTeX tables for copy-paste into Overleaf
"""

print("=" * 80)
print("OVERLEAF COPY-PASTE READY TABLES")
print("=" * 80)
print()

print("TABLE 1: Main Results")
print("-" * 50)
print("""\\begin{table}[htbp]
\\centering
\\caption{Comprehensive Ski Planner Benchmark Results}
\\label{tab:ski_benchmark}
\\begin{tabular}{|l|l|c|c|c|c|c|c|}
\\hline
\\textbf{Difficulty} & \\textbf{Method} & \\textbf{Final Pass} & \\textbf{Delivery} & \\textbf{Hard Const.} & \\textbf{Optimality} & \\textbf{Runtime (s)} & \\textbf{Cost (€)} \\\\
\\hline
\\multirow{3}{*}{Easy} & Pure LLM & 100.0\\% & 100.0\\% & 50.0\\% & 1.00 & 43.4 & 870 \\\\
 & Z3 & 100.0\\% & 100.0\\% & 100.0\\% & 1.00 & 5.1 & 765 \\\\
 & Gurobi & 100.0\\% & 100.0\\% & 100.0\\% & 1.00 & 3.3 & 765 \\\\
\\hline
\\multirow{3}{*}{Medium} & Pure LLM & 100.0\\% & 100.0\\% & 0.0\\% & 0.00 & 0.7 & 0 \\\\
 & Z3 & 100.0\\% & 100.0\\% & 87.5\\% & 1.00 & 3.3 & 1581 \\\\
 & Gurobi & 75.0\\% & 75.0\\% & 47.9\\% & 0.75 & 3.0 & 715 \\\\
\\hline
\\multirow{3}{*}{Hard} & Pure LLM & 100.0\\% & 100.0\\% & 0.0\\% & 0.00 & 0.8 & 0 \\\\
 & Z3 & 100.0\\% & 100.0\\% & 75.0\\% & 1.00 & 3.3 & 4045 \\\\
 & Gurobi & 100.0\\% & 100.0\\% & 66.7\\% & 1.00 & 2.9 & 1579 \\\\
\\hline
\\multirow{3}{*}{Infeasible} & Pure LLM & 100.0\\% & 100.0\\% & 0.0\\% & 0.00 & 0.8 & 0 \\\\
 & Z3 & 0.0\\% & 100.0\\% & 0.0\\% & 0.00 & 5.7 & 0 \\\\
 & Gurobi & 60.0\\% & 100.0\\% & 36.7\\% & 0.60 & 3.2 & 632 \\\\
\\hline
\\end{tabular}
\\end{table}""")

print()
print("TABLE 2: Performance Comparison")
print("-" * 50)
print("""\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison Across Difficulty Levels}
\\label{tab:performance_comparison}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Pure LLM} & \\textbf{Z3} & \\textbf{Gurobi} & \\textbf{Best} \\\\
\\hline
Final Pass Rate & 100.0\\% & 100.0\\% & 91.7\\% & Pure LLM \\\\
Hard Constraint Pass & 16.7\\% & 87.5\\% & 71.5\\% & Z3 \\\\
Optimality Score & 0.33 & 1.00 & 0.92 & Z3 \\\\
Runtime & 15.0s & 3.9s & 3.1s & Gurobi \\\\
\\hline
\\end{tabular}
\\end{table}""")

print()
print("TABLE 3: Constraint Satisfaction")
print("-" * 50)
print("""\\begin{table}[htbp]
\\centering
\\caption{Constraint Satisfaction Analysis}
\\label{tab:constraint_satisfaction}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Planner} & \\textbf{Budget} & \\textbf{Resort} & \\textbf{People} & \\textbf{Days} & \\textbf{Overall} \\\\
\\hline
Pure LLM & 100\\% & 20\\% & 100\\% & 27\\% & 62\\% \\\\
Z3 & 100\\% & 67\\% & 100\\% & 67\\% & 83\\% \\\\
Gurobi & 100\\% & 80\\% & 100\\% & 80\\% & 90\\% \\\\
\\hline
\\end{tabular}
\\end{table}""")

print()
print("=" * 80)
print("INSTRUCTIONS:")
print("1. Copy each table block separately")
print("2. Paste into your Overleaf document")
print("3. Add these packages to your preamble:")
print("   \\usepackage{multirow}")
print("   \\usepackage{booktabs}")
print("   \\usepackage{array}")
print("4. Compile to see the formatted tables")
print("=" * 80)
