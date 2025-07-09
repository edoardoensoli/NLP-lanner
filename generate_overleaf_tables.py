#!/usr/bin/env python3
"""
Generate Overleaf-ready LaTeX tables from comprehensive benchmark results
"""

import json
import pandas as pd
from datetime import datetime

def load_benchmark_results(filename):
    """Load the comprehensive benchmark results"""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_metrics_by_difficulty(results):
    """Extract metrics organized by difficulty level"""
    data = {
        'Easy': [],
        'Medium': [],
        'Hard': [],
        'Infeasible': []
    }
    
    for result in results['results']:
        difficulty = result['difficulty']
        
        # Extract metrics for each planner - they're under 'metrics' key
        if 'metrics' in result:
            for planner in ['pure_llm', 'z3', 'gurobi']:
                metrics_key = f'{planner}_metrics'
                if metrics_key in result['metrics']:
                    metrics = result['metrics'][metrics_key]
                    row = {
                        'query_id': result['id'],
                        'planner': planner.replace('_', ' ').title(),
                        'final_pass_rate': metrics.get('final_pass_rate', 0),
                        'delivery_rate': metrics.get('delivery_rate', 0),
                        'hard_constraint_micro': metrics.get('hard_constraint_pass_rate_micro', 0),
                        'hard_constraint_macro': metrics.get('hard_constraint_pass_rate_macro', 0),
                        'commonsense_pass_rate': metrics.get('commonsense_pass_rate', 0),
                        'optimality': metrics.get('optimality', 0),
                        'runtime': metrics.get('runtime', 0),
                        'cost': metrics.get('cost', 0)
                    }
                    data[difficulty].append(row)
    
    return data

def calculate_average_metrics(data):
    """Calculate average metrics for each difficulty level and planner"""
    averages = {}
    
    for difficulty, rows in data.items():
        if not rows:
            continue
            
        df = pd.DataFrame(rows)
        avg_by_planner = df.groupby('planner').agg({
            'final_pass_rate': 'mean',
            'delivery_rate': 'mean',
            'hard_constraint_micro': 'mean',
            'hard_constraint_macro': 'mean',
            'commonsense_pass_rate': 'mean',
            'optimality': 'mean',
            'runtime': 'mean',
            'cost': 'mean'
        }).round(3)
        
        averages[difficulty] = avg_by_planner
    
    return averages

def generate_main_results_table(averages):
    """Generate the main results table in LaTeX format"""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Comprehensive Ski Planner Benchmark Results}")
    latex.append("\\label{tab:ski_benchmark}")
    latex.append("\\begin{tabular}{|l|l|c|c|c|c|c|c|}")
    latex.append("\\hline")
    latex.append("\\textbf{Difficulty} & \\textbf{Method} & \\textbf{Final Pass} & \\textbf{Delivery} & \\textbf{Hard Const.} & \\textbf{Optimality} & \\textbf{Runtime (s)} & \\textbf{Cost (€)} \\\\")
    latex.append("\\hline")
    
    for difficulty in ['Easy', 'Medium', 'Hard', 'Infeasible']:
        if difficulty in averages:
            first_row = True
            for planner in ['Pure Llm', 'Z3', 'Gurobi']:
                if planner in averages[difficulty].index:
                    row = averages[difficulty].loc[planner]
                    
                    if first_row:
                        latex.append(f"\\multirow{{3}}{{*}}{{{difficulty}}} & {planner} & {row['final_pass_rate']:.1%} & {row['delivery_rate']:.1%} & {row['hard_constraint_micro']:.1%} & {row['optimality']:.2f} & {row['runtime']:.1f} & {row['cost']:.0f} \\\\")
                        first_row = False
                    else:
                        latex.append(f" & {planner} & {row['final_pass_rate']:.1%} & {row['delivery_rate']:.1%} & {row['hard_constraint_micro']:.1%} & {row['optimality']:.2f} & {row['runtime']:.1f} & {row['cost']:.0f} \\\\")
            latex.append("\\hline")
    
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)

def generate_performance_comparison_table(averages):
    """Generate performance comparison table"""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison Across Difficulty Levels}")
    latex.append("\\label{tab:performance_comparison}")
    latex.append("\\begin{tabular}{|l|c|c|c|c|}")
    latex.append("\\hline")
    latex.append("\\textbf{Metric} & \\textbf{Pure LLM} & \\textbf{Z3} & \\textbf{Gurobi} & \\textbf{Best} \\\\")
    latex.append("\\hline")
    
    # Calculate overall averages
    overall_avg = {}
    for planner in ['Pure Llm', 'Z3', 'Gurobi']:
        overall_avg[planner] = {}
        for metric in ['final_pass_rate', 'hard_constraint_micro', 'optimality', 'runtime']:
            values = []
            for difficulty in ['Easy', 'Medium', 'Hard']:
                if difficulty in averages and planner in averages[difficulty].index:
                    values.append(averages[difficulty].loc[planner][metric])
            overall_avg[planner][metric] = sum(values) / len(values) if values else 0
    
    # Create comparison rows
    metrics = [
        ('Final Pass Rate', 'final_pass_rate', '%', 'max'),
        ('Hard Constraint Pass', 'hard_constraint_micro', '%', 'max'),
        ('Optimality Score', 'optimality', '', 'max'),
        ('Runtime', 'runtime', 's', 'min')
    ]
    
    for metric_name, metric_key, unit, best_type in metrics:
        pure_llm_val = overall_avg['Pure Llm'][metric_key]
        z3_val = overall_avg['Z3'][metric_key]
        gurobi_val = overall_avg['Gurobi'][metric_key]
        
        if unit == '%':
            pure_llm_str = f"{pure_llm_val:.1%}"
            z3_str = f"{z3_val:.1%}"
            gurobi_str = f"{gurobi_val:.1%}"
        elif unit == 's':
            pure_llm_str = f"{pure_llm_val:.1f}"
            z3_str = f"{z3_val:.1f}"
            gurobi_str = f"{gurobi_val:.1f}"
        else:
            pure_llm_str = f"{pure_llm_val:.2f}"
            z3_str = f"{z3_val:.2f}"
            gurobi_str = f"{gurobi_val:.2f}"
        
        # Determine best performer
        if best_type == 'max':
            best_val = max(pure_llm_val, z3_val, gurobi_val)
        else:
            best_val = min(pure_llm_val, z3_val, gurobi_val)
        
        if pure_llm_val == best_val:
            best_performer = "Pure LLM"
        elif z3_val == best_val:
            best_performer = "Z3"
        else:
            best_performer = "Gurobi"
        
        latex.append(f"{metric_name} & {pure_llm_str} & {z3_str} & {gurobi_str} & {best_performer} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)

def generate_constraint_satisfaction_table(results):
    """Generate constraint satisfaction analysis table"""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Constraint Satisfaction Analysis}")
    latex.append("\\label{tab:constraint_satisfaction}")
    latex.append("\\begin{tabular}{|l|c|c|c|c|c|}")
    latex.append("\\hline")
    latex.append("\\textbf{Planner} & \\textbf{Budget} & \\textbf{Resort} & \\textbf{People} & \\textbf{Days} & \\textbf{Overall} \\\\")
    latex.append("\\hline")
    
    # Calculate constraint satisfaction rates
    constraint_stats = {}
    for planner in ['Pure Llm', 'Z3', 'Gurobi']:
        constraint_stats[planner] = {
            'budget': 0,
            'resort': 0,
            'people': 0,
            'days': 0,
            'total': 0
        }
    
    total_queries = len(results['results'])
    
    for result in results['results']:
        if 'metrics' in result:
            for planner in ['pure_llm', 'z3', 'gurobi']:
                metrics_key = f'{planner}_metrics'
                if metrics_key in result['metrics']:
                    planner_name = planner.replace('_', ' ').title()
                    constraints = result['metrics'][metrics_key].get('constraint_details', {})
                    
                    for constraint in ['budget_constraint', 'resort_constraint', 'people_constraint', 'days_constraint']:
                        if constraints.get(constraint, False):
                            constraint_stats[planner_name][constraint.replace('_constraint', '')] += 1
    
    # Convert to percentages and generate table
    for planner in ['Pure Llm', 'Z3', 'Gurobi']:
        budget_pct = constraint_stats[planner]['budget'] / total_queries * 100
        resort_pct = constraint_stats[planner]['resort'] / total_queries * 100
        people_pct = constraint_stats[planner]['people'] / total_queries * 100
        days_pct = constraint_stats[planner]['days'] / total_queries * 100
        overall_pct = (budget_pct + resort_pct + people_pct + days_pct) / 4
        
        latex.append(f"{planner} & {budget_pct:.0f}\\% & {resort_pct:.0f}\\% & {people_pct:.0f}\\% & {days_pct:.0f}\\% & {overall_pct:.0f}\\% \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return '\n'.join(latex)

def generate_full_overleaf_document(results):
    """Generate complete Overleaf document with all tables"""
    data = extract_metrics_by_difficulty(results)
    averages = calculate_average_metrics(data)
    
    latex = []
    latex.append("% Ski Planner Benchmark Results for Overleaf")
    latex.append("% Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    latex.append("")
    latex.append("\\documentclass{article}")
    latex.append("\\usepackage{multirow}")
    latex.append("\\usepackage{booktabs}")
    latex.append("\\usepackage{array}")
    latex.append("")
    latex.append("\\begin{document}")
    latex.append("")
    latex.append("\\section{Ski Trip Planning Benchmark Results}")
    latex.append("")
    latex.append("\\subsection{Overview}")
    latex.append(f"This benchmark evaluated {results['benchmark_info']['total_queries']} queries across different difficulty levels:")
    latex.append("\\begin{itemize}")
    latex.append(f"\\item Easy queries: {results['benchmark_info']['query_distribution']['easy']}")
    latex.append(f"\\item Medium queries: {results['benchmark_info']['query_distribution']['medium']}")
    latex.append(f"\\item Hard queries: {results['benchmark_info']['query_distribution']['hard']}")
    latex.append(f"\\item Infeasible queries: {results['benchmark_info']['query_distribution']['infeasible']}")
    latex.append("\\end{itemize}")
    latex.append("")
    latex.append("\\subsection{Main Results}")
    latex.append(generate_main_results_table(averages))
    latex.append("")
    latex.append("\\subsection{Performance Comparison}")
    latex.append(generate_performance_comparison_table(averages))
    latex.append("")
    latex.append("\\subsection{Constraint Satisfaction}")
    latex.append(generate_constraint_satisfaction_table(results))
    latex.append("")
    latex.append("\\end{document}")
    
    return '\n'.join(latex)

def main():
    """Main function to generate all Overleaf tables"""
    
    # Load the benchmark results
    results_file = "benchmark_results/comprehensive_benchmark_20250709_225940.json"
    results = load_benchmark_results(results_file)
    
    print("🎿 GENERATING OVERLEAF TABLES")
    print("=" * 60)
    
    # Generate individual tables
    data = extract_metrics_by_difficulty(results)
    averages = calculate_average_metrics(data)
    
    print("\n📊 Main Results Table:")
    print(generate_main_results_table(averages))
    
    print("\n📈 Performance Comparison Table:")
    print(generate_performance_comparison_table(averages))
    
    print("\n🔍 Constraint Satisfaction Table:")
    print(generate_constraint_satisfaction_table(results))
    
    # Generate full document
    full_doc = generate_full_overleaf_document(results)
    
    # Save to file
    output_file = "overleaf_ski_benchmark_tables.tex"
    with open(output_file, 'w') as f:
        f.write(full_doc)
    
    print(f"\n💾 Full Overleaf document saved to: {output_file}")
    print("\n✅ Ready to copy-paste into Overleaf!")
    print("\n📝 Instructions:")
    print("1. Copy the LaTeX code from the generated .tex file")
    print("2. Paste into your Overleaf document")
    print("3. Compile to see the formatted tables")
    print("4. Adjust table positioning and captions as needed")

if __name__ == "__main__":
    main()
