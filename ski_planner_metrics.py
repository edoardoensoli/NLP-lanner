"""
Comprehensive metrics evaluation system for ski trip planners.
Implements metrics inspired by the TravelPlanner paper for evaluating
Z3, Gurobi, and LLM-based ski trip planning systems.
"""

import re
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os


@dataclass
class MetricResult:
    """Container for a single metric result"""
    value: float
    passed: bool
    details: str = ""
    
    def __str__(self):
        status = "✓" if self.passed else "✗"
        return f"{status} {self.value:.2f} - {self.details}"


@dataclass
class PlannerResult:
    """Container for a planner's result on a single query"""
    planner_name: str
    query: str
    result_type: str  # 'optimal', 'infeasible', 'failed'
    plan_text: str
    cost: Optional[float] = None
    runtime: Optional[float] = None
    resort_name: Optional[str] = None
    suggestions: Optional[str] = None
    
    # Extracted plan components
    accommodation: Optional[str] = None
    car_rental: Optional[str] = None
    equipment_rental: Optional[str] = None
    
    def __post_init__(self):
        """Extract plan components after initialization"""
        if self.result_type == 'optimal' and self.plan_text:
            self._extract_plan_components()
    
    def _extract_plan_components(self):
        """Extract key components from the plan text"""
        if not self.plan_text:
            return
            
        # Extract resort name - handle both Z3 and Gurobi formats
        resort_match = re.search(r'\*\*Selected Resort:\*\*\s*(.+?)(?:\n|$)', self.plan_text)
        if not resort_match:
            # Try Z3 format
            resort_match = re.search(r'SELECTED RESORT:\s*(.+?)(?:\n|$)', self.plan_text)
        if resort_match:
            self.resort_name = resort_match.group(1).strip()
        
        # Extract cost - handle both formats
        cost_match = re.search(r'\*\*Total Cost:\*\* â‚¬(\d+(?:\.\d+)?)', self.plan_text)
        if not cost_match:
            # Try Z3 format (with exact encoding)
            cost_match = re.search(r'- TOTAL COST: â‚¬(\d+(?:\.\d+)?)', self.plan_text)
        if cost_match:
            self.cost = float(cost_match.group(1))
        
        # Extract accommodation - handle both formats
        acc_section = re.search(r'#### Accommodation\n(.*?)(?=####|$)', self.plan_text, re.DOTALL)
        if not acc_section:
            # For Z3 format, if we have a resort selected, consider that as accommodation
            if self.resort_name:
                self.accommodation = f"Stay at {self.resort_name}"
            # Also try to find accommodation cost in breakdown
            acc_cost_match = re.search(r'Accommodation:\s*€(\d+(?:\.\d+)?)', self.plan_text)
            if acc_cost_match:
                cost_value = acc_cost_match.group(1)
                if self.accommodation:
                    self.accommodation += f" (€{cost_value})"
                else:
                    self.accommodation = f"Accommodation cost: €{cost_value}"
        if acc_section:
            self.accommodation = acc_section.group(1).strip()
        
        # Extract car rental - handle both formats
        car_section = re.search(r'#### Car Rental\n(.*?)(?=####|$)', self.plan_text, re.DOTALL)
        if not car_section:
            # Try Z3 format
            car_match = re.search(r'CAR RENTAL:\s*\n(.+?)(?:\n\n|\nEQUIPMENT|\nCOST|\n$)', self.plan_text, re.DOTALL)
            if car_match:
                self.car_rental = car_match.group(1).strip()
        if car_section:
            self.car_rental = car_section.group(1).strip()
        
        # Extract equipment rental - handle both formats
        equip_section = re.search(r'#### Equipment Rental\n(.*?)(?=####|$)', self.plan_text, re.DOTALL)
        if not equip_section:
            # Try Z3 format
            equip_match = re.search(r'EQUIPMENT RENTAL:\s*\n(.+?)(?:\n\n|\nCAR|\nCOST|\n$)', self.plan_text, re.DOTALL)
            if equip_match:
                self.equipment_rental = equip_match.group(1).strip()
        if equip_section:
            self.equipment_rental = equip_section.group(1).strip()


class QueryParameterExtractor:
    """Extracts structured parameters from natural language queries"""
    
    @staticmethod
    def extract_parameters(query: str) -> Dict[str, Any]:
        """Extract key parameters from a query string"""
        params = {
            'destination': None,
            'days': None,
            'people': None,
            'budget': None,
            'car_required': False,
            'equipment_required': False,
            'slope_difficulty': None,
            'fuel_type': None,
            'car_type': None,
            'special_requirements': []
        }
        
        query_lower = query.lower()
        
        # Extract destination
        ski_destinations = {
            "livigno": "Livigno",
            "cortina": "Cortina d'Ampezzo",
            "val gardena": "Val Gardena",
            "madonna di campiglio": "Madonna di Campiglio",
            "sestriere": "Sestriere",
            "la thuile": "La Thuile",
            "val senales": "Val Senales Glacier",
            "courmayeur": "Courmayeur",
            "zermatt": "Zermatt",
            "st. moritz": "St. Moritz",
            "verbier": "Verbier",
            "davos": "Davos",
            "switzerland": "Switzerland"
        }
        
        for key, value in ski_destinations.items():
            if key in query_lower:
                params['destination'] = value
                break
        
        # Extract days
        days_match = re.search(r'(\d+)\s*[-\s]*days?', query_lower)
        if days_match:
            params['days'] = int(days_match.group(1))
        
        # Extract people
        people_match = re.search(r'(\d+)\s*people', query_lower)
        if people_match:
            params['people'] = int(people_match.group(1))
        
        # Extract budget
        budget_match = re.search(r'(\d+)\s*euro', query_lower)
        if budget_match:
            params['budget'] = int(budget_match.group(1))
        
        # Check for car requirements
        car_patterns = [
            r'car rental', r'rent.*car', r'suv', r'sedan', r'pick up',
            r'cabriolet', r'electric', r'hybrid', r'diesel', r'petrol'
        ]
        params['car_required'] = any(re.search(pattern, query_lower) for pattern in car_patterns)
        
        # Check for equipment requirements
        equipment_patterns = [
            r'equipment', r'ski.*rental', r'rent.*ski', r'ski.*equipment',
            r'rental.*equipment', r'rent.*equipment', r'rent.*skis'
        ]
        params['equipment_required'] = any(re.search(pattern, query_lower) for pattern in equipment_patterns)
        
        # Extract slope difficulty
        if any(word in query_lower for word in ['black', 'advanced', 'expert']):
            params['slope_difficulty'] = 'advanced'
        elif any(word in query_lower for word in ['red', 'intermediate']):
            params['slope_difficulty'] = 'intermediate'
        elif any(word in query_lower for word in ['blue', 'beginner', 'easy', 'not expert']):
            params['slope_difficulty'] = 'beginner'
        
        # Extract fuel type
        if 'electric' in query_lower:
            params['fuel_type'] = 'electric'
        elif 'hybrid' in query_lower:
            params['fuel_type'] = 'hybrid'
        elif 'diesel' in query_lower:
            params['fuel_type'] = 'diesel'
        elif 'petrol' in query_lower:
            params['fuel_type'] = 'petrol'
        
        # Extract car type
        if 'suv' in query_lower:
            params['car_type'] = 'suv'
        elif 'sedan' in query_lower:
            params['car_type'] = 'sedan'
        elif 'pick up' in query_lower or 'pickup' in query_lower:
            params['car_type'] = 'pickup'
        elif 'cabriolet' in query_lower:
            params['car_type'] = 'cabriolet'
        
        return params


class SkiPlannerMetrics:
    """Comprehensive metrics evaluation for ski trip planners"""
    
    def __init__(self):
        self.results = []
        self.query_extractor = QueryParameterExtractor()
    
    def evaluate_single_result(self, result: PlannerResult) -> Dict[str, MetricResult]:
        """Evaluate all metrics for a single planner result"""
        metrics = {}
        
        # Extract query parameters
        query_params = self.query_extractor.extract_parameters(result.query)
        
        # Final Pass Rate (overall success)
        metrics['final_pass_rate'] = self._evaluate_final_pass_rate(result)
        
        # Delivery Rate (successful plan generation)
        metrics['delivery_rate'] = self._evaluate_delivery_rate(result)
        
        # Hard Constraint Pass Rate
        hard_constraints_result = self._evaluate_hard_constraints(result, query_params)
        metrics['hard_constraint_pass_rate'] = hard_constraints_result
        
        # Calculate micro and macro separately
        metrics['hard_constraint_micro'] = self._evaluate_hard_constraints_micro(result, query_params)
        metrics['hard_constraint_macro'] = hard_constraints_result  # Macro is same as overall
        
        # Commonsense Constraint Pass Rate
        metrics['commonsense_constraint_pass_rate'] = self._evaluate_commonsense_constraints(result, query_params)
        
        # Repair Success (for infeasible queries)
        metrics['repair_success'] = self._evaluate_repair_success(result)
        
        # Optimality (cost efficiency)
        metrics['optimality'] = self._evaluate_optimality(result, query_params)
        
        # Runtime efficiency
        metrics['runtime_efficiency'] = self._evaluate_runtime_efficiency(result)
        
        # Cost reasonableness
        metrics['cost_reasonableness'] = self._evaluate_cost_reasonableness(result, query_params)
        
        return metrics
    
    def _evaluate_final_pass_rate(self, result: PlannerResult) -> MetricResult:
        """Overall success rate - 1.0 if optimal solution found, 0.0 otherwise"""
        passed = result.result_type == 'optimal'
        value = 1.0 if passed else 0.0
        details = f"Result: {result.result_type}"
        return MetricResult(value, passed, details)
    
    def _evaluate_delivery_rate(self, result: PlannerResult) -> MetricResult:
        """Rate of successful plan generation (optimal or infeasible with suggestions)"""
        passed = result.result_type in ['optimal', 'infeasible']
        value = 1.0 if passed else 0.0
        details = f"Delivered: {result.result_type}"
        return MetricResult(value, passed, details)
    
    def _evaluate_hard_constraints(self, result: PlannerResult, query_params: Dict[str, Any]) -> MetricResult:
        """Check satisfaction of hard constraints (budget, required services)"""
        if result.result_type != 'optimal':
            return MetricResult(0.0, False, "No optimal solution to evaluate")
        
        violations = []
        
        # Budget constraint
        if query_params.get('budget') and result.cost:
            if result.cost > query_params['budget']:
                violations.append(f"Budget exceeded: €{result.cost:.2f} > €{query_params['budget']}")
        
        # Required car rental
        if query_params.get('car_required'):
            if not result.car_rental or 'No car rental' in result.car_rental:
                violations.append("Car rental required but not provided")
        
        # Required equipment rental
        if query_params.get('equipment_required'):
            if not result.equipment_rental or 'No equipment rental' in result.equipment_rental:
                violations.append("Equipment rental required but not provided")
        
        # Required accommodation
        if not result.accommodation or 'No accommodation' in result.accommodation:
            violations.append("Accommodation required but not provided")
        
        passed = len(violations) == 0
        value = 1.0 if passed else 0.0
        details = "All hard constraints satisfied" if passed else "; ".join(violations)
        
        return MetricResult(value, passed, details)
    
    def _evaluate_hard_constraints_micro(self, result: PlannerResult, query_params: Dict[str, Any]) -> MetricResult:
        """Check individual constraint satisfaction rates (micro-level)"""
        if result.result_type != 'optimal':
            return MetricResult(0.0, False, "No optimal solution to evaluate")
        
        constraints_checked = []
        constraints_passed = []
        
        # Budget constraint
        if query_params.get('budget') and result.cost:
            constraints_checked.append('budget')
            if result.cost <= query_params['budget']:
                constraints_passed.append('budget')
        
        # Required car rental
        if query_params.get('car_required'):
            constraints_checked.append('car_rental')
            if result.car_rental and 'No car rental' not in result.car_rental:
                constraints_passed.append('car_rental')
        
        # Required equipment rental
        if query_params.get('equipment_required'):
            constraints_checked.append('equipment_rental')
            if result.equipment_rental and 'No equipment rental' not in result.equipment_rental:
                constraints_passed.append('equipment_rental')
        
        # Required accommodation (always checked)
        constraints_checked.append('accommodation')
        if result.accommodation and 'No accommodation' not in result.accommodation:
            constraints_passed.append('accommodation')
        
        # Calculate micro pass rate
        if not constraints_checked:
            return MetricResult(1.0, True, "No hard constraints to check")
        
        pass_rate = len(constraints_passed) / len(constraints_checked)
        passed = pass_rate >= 0.8  # 80% threshold for passing
        details = f"Passed {len(constraints_passed)}/{len(constraints_checked)} constraints: {', '.join(constraints_passed)}"
        
        return MetricResult(pass_rate, passed, details)
    
    def _evaluate_commonsense_constraints(self, result: PlannerResult, query_params: Dict[str, Any]) -> MetricResult:
        """Check satisfaction of commonsense constraints (logical consistency)"""
        if result.result_type != 'optimal':
            return MetricResult(0.0, False, "No optimal solution to evaluate")
        
        violations = []
        
        # Destination matching
        if query_params.get('destination') and result.resort_name:
            # Check if the selected resort is in the requested destination area
            destination = query_params['destination'].lower()
            resort = result.resort_name.lower()
            
            # Simple matching logic - can be enhanced
            if destination in ['switzerland', 'zermatt', 'st. moritz', 'verbier', 'davos']:
                if not any(swiss_indicator in resort for swiss_indicator in ['zermatt', 'st. moritz', 'verbier', 'davos']):
                    violations.append(f"Resort '{result.resort_name}' may not match destination '{query_params['destination']}'")
        
        # Equipment and car consistency
        if query_params.get('equipment_required') and not query_params.get('car_required'):
            if result.equipment_rental and 'No equipment rental' not in result.equipment_rental:
                if not result.car_rental or 'No car rental' in result.car_rental:
                    violations.append("Equipment rental without car rental may be impractical")
        
        # Cost reasonableness for group size
        people_count = query_params.get('people') or 1
        if people_count and people_count > 1 and result.cost:
            try:
                cost_per_person = result.cost / people_count
                if cost_per_person < 200:  # Very low cost per person might indicate missing services
                    violations.append(f"Cost per person (€{cost_per_person:.2f}) seems unusually low")
            except (TypeError, ZeroDivisionError):
                # Skip this check if we can't calculate cost per person
                pass
        
        passed = len(violations) == 0
        value = 1.0 if passed else max(0.0, 1.0 - len(violations) * 0.5)  # Partial credit
        details = "All commonsense constraints satisfied" if passed else "; ".join(violations)
        
        return MetricResult(value, passed, details)
    
    def _evaluate_repair_success(self, result: PlannerResult) -> MetricResult:
        """Evaluate quality of suggestions for infeasible queries"""
        if result.result_type != 'infeasible':
            return MetricResult(1.0, True, "Not applicable (feasible query)")
        
        if not result.suggestions:
            return MetricResult(0.0, False, "No suggestions provided for infeasible query")
        
        # Check quality of suggestions
        suggestion_quality = 0.0
        suggestions_lower = result.suggestions.lower()
        
        # Budget-related suggestions
        if 'budget' in suggestions_lower:
            suggestion_quality += 0.3
        
        # Specific constraint suggestions
        if any(word in suggestions_lower for word in ['try', 'consider', 'increase', 'decrease', 'relax']):
            suggestion_quality += 0.3
        
        # Specific values or alternatives
        if any(word in suggestions_lower for word in ['€', 'euro', 'resort', 'destination']):
            suggestion_quality += 0.4
        
        passed = suggestion_quality >= 0.5
        value = min(1.0, suggestion_quality)
        details = f"Suggestion quality: {suggestion_quality:.1f}"
        
        return MetricResult(value, passed, details)
    
    def _get_cost_baseline(self, query_params: Dict[str, Any]) -> float:
        """Calculate baseline cost based on destination and trip characteristics"""
        days = query_params.get('days') or 3
        people = query_params.get('people') or 1
        destination = query_params.get('destination', '') or ''
        destination = destination.lower()
        
        # Use the detailed destination cost baseline
        base_cost = self._get_destination_cost_baseline(destination, days, people)
        
        # Add estimated costs for required services
        if query_params.get('car_required'):
            base_cost += days * 50  # €50 per day for car
        
        if query_params.get('equipment_required'):
            base_cost += days * people * 30  # €30 per person per day for equipment
        
        return base_cost
    
    def _get_destination_cost_baseline(self, destination: str, days: int, people: int) -> float:
        """Get cost baseline based on destination and trip type"""
        # Define baseline cost per person per day by destination type
        cost_baselines = {
            # Swiss resorts (premium)
            'switzerland': 250,
            'zermatt': 280,
            'st. moritz': 300,
            'verbier': 270,
            'davos': 250,
            'saas-fee': 240,
            
            # Italian resorts (mid-range)
            'italy': 200,
            'livigno': 180,
            'cortina': 220,
            'val gardena': 200,
            'madonna di campiglio': 210,
            'courmayeur': 200,
            
            # Austrian resorts (mid-range)
            'austria': 190,
            'innsbruck': 200,
            'salzburg': 180,
            
            # French resorts (premium)
            'france': 230,
            'chamonix': 250,
            'val d\'isère': 270,
            'courchevel': 280,
            
            # Norwegian resorts (premium)
            'norway': 240,
            'hemsedal': 220,
            'geilo': 230,
            
            # Default for unknown destinations
            'default': 200
        }
        
        # Find matching baseline
        baseline_per_person_per_day = cost_baselines.get('default', 200)
        
        if destination:
            destination_lower = destination.lower()
            # Check for exact matches first
            for key in cost_baselines:
                if key in destination_lower:
                    baseline_per_person_per_day = cost_baselines[key]
                    break
        
        # Calculate base cost
        base_cost = days * people * baseline_per_person_per_day
        return base_cost

    def _evaluate_optimality(self, result: PlannerResult, query_params: Dict[str, Any]) -> MetricResult:
        """Evaluate cost optimality (lower cost is better)"""
        if result.result_type != 'optimal' or not result.cost:
            return MetricResult(0.0, False, "No optimal solution with cost to evaluate")
        
        # Use configurable baseline instead of hardcoded value
        base_cost = self._get_cost_baseline(query_params)
        
        # Optimality score: how close to reasonable minimum cost
        if result.cost <= base_cost:
            value = 1.0
            passed = True
            details = f"Excellent cost optimization: €{result.cost:.2f} ≤ €{base_cost:.2f}"
        elif result.cost <= base_cost * 1.3:
            value = 0.8
            passed = True
            details = f"Good cost optimization: €{result.cost:.2f} vs €{base_cost:.2f} baseline"
        elif result.cost <= base_cost * 1.6:
            value = 0.6
            passed = True
            details = f"Reasonable cost: €{result.cost:.2f} vs €{base_cost:.2f} baseline"
        else:
            value = 0.4
            passed = False
            details = f"High cost: €{result.cost:.2f} vs €{base_cost:.2f} baseline"
        
        return MetricResult(value, passed, details)
    
    def _evaluate_runtime_efficiency(self, result: PlannerResult) -> MetricResult:
        """Evaluate runtime efficiency"""
        if result.runtime is None:
            return MetricResult(0.5, True, "Runtime not measured")
        
        # Define efficiency thresholds
        if result.runtime <= 5.0:
            value = 1.0
            passed = True
            details = f"Excellent runtime: {result.runtime:.2f}s"
        elif result.runtime <= 15.0:
            value = 0.8
            passed = True
            details = f"Good runtime: {result.runtime:.2f}s"
        elif result.runtime <= 30.0:
            value = 0.6
            passed = True
            details = f"Acceptable runtime: {result.runtime:.2f}s"
        else:
            value = 0.4
            passed = False
            details = f"Slow runtime: {result.runtime:.2f}s"
        
        return MetricResult(value, passed, details)
    
    def _evaluate_cost_reasonableness(self, result: PlannerResult, query_params: Dict[str, Any]) -> MetricResult:
        """Evaluate if cost is reasonable for the requested trip"""
        if result.result_type != 'optimal' or not result.cost:
            return MetricResult(0.0, False, "No optimal solution with cost to evaluate")
        
        days = query_params.get('days') or 3
        people = query_params.get('people') or 1
        budget = query_params.get('budget')
        
        # Cost per person per day
        cost_per_person_per_day = result.cost / (days * people)
        
        # Define reasonable ranges
        if cost_per_person_per_day < 100:
            value = 0.6
            passed = True
            details = f"Very economical: €{cost_per_person_per_day:.2f}/person/day"
        elif cost_per_person_per_day <= 300:
            value = 1.0
            passed = True
            details = f"Reasonable cost: €{cost_per_person_per_day:.2f}/person/day"
        elif cost_per_person_per_day <= 500:
            value = 0.8
            passed = True
            details = f"Premium cost: €{cost_per_person_per_day:.2f}/person/day"
        else:
            value = 0.4
            passed = False
            details = f"Very expensive: €{cost_per_person_per_day:.2f}/person/day"
        
        # Check against budget if provided
        if budget and result.cost > budget:
            value *= 0.5  # Penalize exceeding budget
            passed = False
            details += f" (exceeds budget of €{budget})"
        
        return MetricResult(value, passed, details)
    
    def add_result(self, result: PlannerResult):
        """Add a result and evaluate its metrics"""
        metrics = self.evaluate_single_result(result)
        self.results.append({
            'result': result,
            'metrics': metrics
        })
    
    def get_aggregated_metrics(self, planner_name: str = None) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics across all results or for a specific planner"""
        filtered_results = self.results
        if planner_name:
            filtered_results = [r for r in self.results if r['result'].planner_name == planner_name]
        
        if not filtered_results:
            return {}
        
        # Initialize aggregation
        metric_names = list(filtered_results[0]['metrics'].keys())
        aggregated = {}
        
        for metric_name in metric_names:
            values = [r['metrics'][metric_name].value for r in filtered_results]
            passed_count = sum(1 for r in filtered_results if r['metrics'][metric_name].passed)
            
            aggregated[metric_name] = {
                'mean': sum(values) / len(values),
                'pass_rate': passed_count / len(filtered_results),
                'count': len(values)
            }
        
        return aggregated
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive metrics report"""
        if not self.results:
            return "No results to report."
        
        report = []
        report.append("# SKI PLANNER METRICS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total queries evaluated: {len(self.results)}")
        report.append("")
        
        # Get unique planners
        planners = list(set(r['result'].planner_name for r in self.results))
        
        # Overall summary
        report.append("## OVERALL SUMMARY")
        report.append("")
        
        for planner in planners:
            planner_results = [r for r in self.results if r['result'].planner_name == planner]
            metrics = self.get_aggregated_metrics(planner)
            
            report.append(f"### {planner.upper()}")
            report.append(f"- Queries processed: {len(planner_results)}")
            
            if metrics:
                report.append(f"- Final Pass Rate: {metrics['final_pass_rate']['pass_rate']:.1%} (avg: {metrics['final_pass_rate']['mean']:.3f})")
                report.append(f"- Delivery Rate: {metrics['delivery_rate']['pass_rate']:.1%} (avg: {metrics['delivery_rate']['mean']:.3f})")
                report.append(f"- Hard Constraints: {metrics['hard_constraint_pass_rate']['pass_rate']:.1%} (avg: {metrics['hard_constraint_pass_rate']['mean']:.3f})")
                report.append(f"- Commonsense Constraints: {metrics['commonsense_constraint_pass_rate']['pass_rate']:.1%} (avg: {metrics['commonsense_constraint_pass_rate']['mean']:.3f})")
                report.append(f"- Repair Success: {metrics['repair_success']['pass_rate']:.1%} (avg: {metrics['repair_success']['mean']:.3f})")
                report.append(f"- Optimality: {metrics['optimality']['pass_rate']:.1%} (avg: {metrics['optimality']['mean']:.3f})")
                report.append(f"- Runtime Efficiency: {metrics['runtime_efficiency']['pass_rate']:.1%} (avg: {metrics['runtime_efficiency']['mean']:.3f})")
                report.append(f"- Cost Reasonableness: {metrics['cost_reasonableness']['pass_rate']:.1%} (avg: {metrics['cost_reasonableness']['mean']:.3f})")
            
            report.append("")
        
        # Detailed results
        report.append("## DETAILED RESULTS")
        report.append("")
        
        for i, result_data in enumerate(self.results, 1):
            result = result_data['result']
            metrics = result_data['metrics']
            
            report.append(f"### Query {i}: {result.planner_name}")
            report.append(f"**Query:** {result.query}")
            report.append(f"**Result:** {result.result_type}")
            if result.cost:
                report.append(f"**Cost:** €{result.cost:.2f}")
            if result.runtime:
                report.append(f"**Runtime:** {result.runtime:.2f}s")
            if result.resort_name:
                report.append(f"**Resort:** {result.resort_name}")
            report.append("")
            
            report.append("**Metrics:**")
            for metric_name, metric_result in metrics.items():
                report.append(f"- {metric_name.replace('_', ' ').title()}: {metric_result}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Metrics report saved to: {output_file}")
        
        return report_text
    
    def export_to_csv(self, output_file: str):
        """Export metrics to CSV for further analysis"""
        if not self.results:
            print("No results to export.")
            return
        
        rows = []
        for result_data in self.results:
            result = result_data['result']
            metrics = result_data['metrics']
            
            row = {
                'planner': result.planner_name,
                'query': result.query,
                'result_type': result.result_type,
                'cost': result.cost,
                'runtime': result.runtime,
                'resort_name': result.resort_name,
            }
            
            # Add metric values
            for metric_name, metric_result in metrics.items():
                row[f'{metric_name}_value'] = metric_result.value
                row[f'{metric_name}_passed'] = metric_result.passed
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Metrics data exported to: {output_file}")
    
    def evaluate_all_metrics(self, query: str, plan_text: str, execution_time: float, 
                          success: bool, feasible: bool, status: str, 
                          error_message: str = "", total_cost: float = 0.0, 
                          llm_calls: int = 0, model_used: str = "") -> Dict[str, float]:
        """
        Wrapper method to evaluate all metrics for a single query result.
        Returns a dictionary with float values for compatibility with the benchmarking system.
        """
        
        # Convert to PlannerResult format
        result = PlannerResult(
            planner_name=model_used or "Unknown",
            query=query,
            result_type=status,
            plan_text=plan_text,
            suggestions=error_message if not success else "",
            runtime=execution_time,
            cost=total_cost
        )
        
        # Evaluate metrics
        metrics = self.evaluate_single_result(result)
        
        # Extract query parameters for detailed constraint evaluation
        query_params = self.query_extractor.extract_parameters(query)
        
        # Evaluate individual constraints properly
        budget_satisfied = True
        if query_params.get('budget') and total_cost > 0:
            budget_satisfied = total_cost <= query_params['budget']
        
        # Check if required services are provided based on plan text
        car_required = query_params.get('car_required', False)
        equipment_required = query_params.get('equipment_required', False)
        
        # Only evaluate constraints that are actually required
        car_satisfied = None  # Not applicable if not required
        if car_required:
            car_satisfied = plan_text and ('car rental' in plan_text.lower() or 'rented car' in plan_text.lower()) and 'no car rental' not in plan_text.lower()
        
        equipment_satisfied = None  # Not applicable if not required
        if equipment_required:
            equipment_satisfied = plan_text and ('equipment' in plan_text.lower() or 'skis' in plan_text.lower() or 'boots' in plan_text.lower()) and 'no equipment' not in plan_text.lower()
        
        # Resort constraint - check if a resort is mentioned
        resort_satisfied = plan_text and any(keyword in plan_text.lower() for keyword in ['resort', 'hotel', 'accommodation', 'stay'])
        
        # Days constraint - assume satisfied if plan mentions duration
        days_satisfied = plan_text and (str(query_params.get('days', 3)) in plan_text or 'day' in plan_text.lower())
        
        # People constraint - assume satisfied if cost is reasonable for group size
        people_satisfied = True
        people_count = query_params.get('people') or 1
        if people_count > 1 and total_cost > 0:
            try:
                cost_per_person = total_cost / people_count
                people_satisfied = cost_per_person >= 50  # Minimum reasonable cost per person
            except (TypeError, ZeroDivisionError):
                people_satisfied = True  # Default to satisfied if we can't calculate
        
        # Build constraint details dictionary - only include applicable constraints
        constraint_details = {
            'budget_constraint': budget_satisfied,
            'dates_constraint': days_satisfied,
            'people_constraint': people_satisfied,
            'resort_constraint': resort_satisfied,
            'days_constraint': days_satisfied,  # Same as dates for now
        }
        
        # Only add car/equipment constraints if they were actually required
        if car_required:
            constraint_details['car_constraint'] = car_satisfied
        if equipment_required:
            constraint_details['equipment_constraint'] = equipment_satisfied
        
        # Convert to flat dictionary with float values
        return {
            'final_pass_rate': metrics['final_pass_rate'].value,
            'delivery_rate': metrics['delivery_rate'].value,
            'hard_constraint_pass_rate_micro': metrics['hard_constraint_micro'].value,
            'hard_constraint_pass_rate_macro': metrics['hard_constraint_macro'].value,
            'commonsense_pass_rate': metrics['commonsense_constraint_pass_rate'].value,
            'interactive_plan_repair_success': metrics['repair_success'].value,
            'optimality': metrics['optimality'].value,
            'runtime': execution_time,
            'cost': total_cost,
            'constraint_details': constraint_details
        }


# Example usage and testing
if __name__ == "__main__":
    # Example test
    metrics = SkiPlannerMetrics()
    
    # Test result 1: Optimal solution
    result1 = PlannerResult(
        planner_name="Gurobi",
        query="Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment",
        result_type="optimal",
        plan_text="""### GUROBI SKI TRIP PLAN
**Query:** Plan a 3-day ski trip to Livigno for 2 people with a budget of 1500 euros, need car rental and equipment
**Result:** Optimal solution found!
**Selected Resort:** Livigno Resort
**Total Cost:** €1450.00
**Cost Breakdown:**
- Accommodation: €900.00
- Car Rental: €300.00
- Equipment Rental: €250.00

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
    
    # Test result 2: Infeasible solution
    result2 = PlannerResult(
        planner_name="Z3",
        query="Plan a 2-day ski trip to Zermatt for 4 people with a budget of 500 euros",
        result_type="infeasible",
        plan_text="The query is infeasible.",
        suggestions="Suggestion: try relaxing some constraints, for example, the budget of €500 is too low for the requested trip.",
        runtime=1.8
    )
    
    metrics.add_result(result1)
    metrics.add_result(result2)
    
    # Generate report
    report = metrics.generate_report()
    print(report)
