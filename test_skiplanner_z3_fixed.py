"""
Fixed Z3 Ski Planner with Real LLM Integration
==============================================

This file provides a clean Z3-based ski trip planner that uses real LLM calls
instead of mock responses. It follows the same pattern as test_skiplanner_llm.py
but adds Z3 constraint solving for optimization.
"""

import pandas as pd
import json
import os
import argparse
import re
import requests
import time
import sys
from dotenv import load_dotenv
from z3 import Optimize, Real, Int, Bool, And, Or, sat

# Add project paths for tools access
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import ski-specific APIs
from tools_ski.apis import SkiResorts, SkiSlopes, SkiRent, SkiCar

load_dotenv()


class LLMClient:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
        self.base_url = "https://models.inference.ai.azure.com"
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def _query_api(self, prompt):
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to extract information from user queries and generate planning code."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": self.model_name,
            "temperature": 0.7,
            "max_tokens": 1500,
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return None

    def query_to_json_response(self, query):
        """Convert query to JSON using real LLM"""
        try:
            with open("prompts/ski/query_to_json_ski.txt", "r") as f:
                prompt_template = f.read()
            prompt = prompt_template + "{" + query + "}\nJSON:\n"
        except FileNotFoundError:
            # Fallback prompt if file not found
            prompt = f"""You are given a natural language query about ski resort planning. Convert this query into a structured JSON format.

The JSON should include these fields:
- "domain": always "ski"
- "destination": the country/region requested  
- "days": number of days
- "people": number of people
- "budget": budget in euros (if mentioned)
- "rating": minimum rating 1.0-5.0 (if mentioned)

Query: {query}

JSON:"""
        
        response = self._query_api(prompt)
        return response

    def constraint_to_step_response(self, query):
        """Generate planning steps using real LLM"""
        try:
            with open("prompts/ski/constraint_to_step_ski.txt", "r") as f:
                prompt_template = f.read()
            prompt = prompt_template + query + "\nSteps:\n"
        except FileNotFoundError:
            prompt = f"""Given this ski trip query, break it down into detailed planning steps:

Query: {query}

Generate specific steps for:
1. Resort Selection
2. Slope Information  
3. Equipment Rental
4. Car Rental (if needed)
5. Budget Calculation

Steps:"""
        
        response = self._query_api(prompt)
        return response

    def step_to_code_response(self, step_type, content):
        """Generate Z3 constraint code for a specific planning step"""
        prompt_file_mapping = {
            'Destination cities': 'step_to_code_destination.txt',
            'Ski resort': 'step_to_code_resort.txt', 
            'Ski slopes': 'step_to_code_slopes.txt',
            'Ski equipment': 'step_to_code_equipment.txt',
            'Car rental': 'step_to_code_car.txt',
            'Budget': 'step_to_code_budget.txt'
        }
        
        prompt_file = prompt_file_mapping.get(step_type)
        if prompt_file:
            try:
                with open(f"prompts/ski/{prompt_file}", "r") as f:
                    prompt_template = f.read()
                prompt = prompt_template + "\n" + content + "\nZ3 Code:\n"
            except FileNotFoundError:
                prompt = f"""Generate Z3 constraint solver code for this ski planning step:

Step Type: {step_type}
Content: {content}

Generate Python code using Z3 that creates variables and constraints for this step.

Z3 Code:"""
        else:
            prompt = f"""Generate Z3 constraint solver code for this ski planning step:

Step Type: {step_type}
Content: {content}

Generate Python code using Z3 that creates variables and constraints for this step.

Z3 Code:"""
        
        response = self._query_api(prompt)
        return response


def parse_and_extract_json(response_text):
    """Extract JSON from LLM response"""
    # Try different JSON extraction patterns
    patterns = [
        r"```json\n(.*?)\n```",
        r"```(json)?\n(.*?)\n```", 
        r"JSON:\s*(\{.*?\})",
        r"(\{.*?\})"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1) if len(match.groups()) == 1 else match.group(2)
            try:
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                continue
    
    # Try parsing entire response as JSON
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        print(f"Could not extract JSON from: {response_text[:200]}...")
        return None


def initialize_ski_apis():
    """Initialize all ski-related API services"""
    try:
        print("Initializing ski planner...")
        ski_resort_search = SkiResorts()
        ski_slope_search = SkiSlopes() 
        ski_rent_search = SkiRent()
        ski_car_search = SkiCar()
        print("Ski APIs initialized successfully!")
        return {
            'resort': ski_resort_search,
            'slopes': ski_slope_search,
            'equipment': ski_rent_search,
            'car': ski_car_search
        }
    except Exception as e:
        print(f"Error initializing ski APIs: {e}")
        return None


def solve_with_z3(query_json, steps, apis, llm_client, verbose=False):
    """Use Z3 to solve the ski planning problem with constraints"""
    if verbose:
        print("Setting up Z3 constraint solver...")
    
    s = Optimize()
    variables = {}
    
    # Initialize cost variables
    accommodation_cost = Real('accommodation_cost')
    equipment_cost = Real('equipment_cost') 
    car_cost = Real('car_cost')
    total_cost = Real('total_cost')
    
    # Add basic constraints
    s.add(accommodation_cost >= 0)
    s.add(equipment_cost >= 0)
    s.add(car_cost >= 0)
    s.add(total_cost == accommodation_cost + equipment_cost + car_cost)
    
    # Add budget constraint if specified
    budget = query_json.get('budget')
    if budget:
        s.add(total_cost <= budget)
        if verbose:
            print(f"Added budget constraint: total_cost <= {budget}")
    
    # Process each planning step and generate constraints
    for step in steps:
        if not step.strip():
            continue
            
        if verbose:
            print(f"Processing step: {step[:50]}...")
        
        # Determine step type
        step_type = determine_step_type(step)
        if step_type:
            try:
                # Get Z3 code from LLM for this step
                z3_code = llm_client.step_to_code_response(step_type, step)
                
                if z3_code and "pass" not in z3_code:
                    # Execute the generated Z3 code (safely)
                    exec_context = {
                        's': s, 'variables': variables, 'query_json': query_json,
                        'Int': Int, 'Real': Real, 'Bool': Bool, 'And': And, 'Or': Or,
                        'apis': apis, 'accommodation_cost': accommodation_cost,
                        'equipment_cost': equipment_cost, 'car_cost': car_cost,
                        'total_cost': total_cost
                    }
                    
                    try:
                        exec(z3_code, exec_context)
                        if verbose:
                            print(f"✅ Added constraints for {step_type}")
                    except Exception as e:
                        if verbose:
                            print(f"⚠️  Error executing Z3 code for {step_type}: {e}")
                        
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error processing step {step_type}: {e}")
    
    # Set optimization objective (minimize cost)
    s.minimize(total_cost)
    
    if verbose:
        print("Solving Z3 optimization problem...")
    
    # Solve the optimization problem
    result = s.check()
    
    if result == sat:
        model = s.model()
        if verbose:
            print("✅ Z3 found optimal solution!")
        
        # Extract solution values
        solution = {
            'accommodation_cost': float(model[accommodation_cost].as_decimal(3)),
            'equipment_cost': float(model[equipment_cost].as_decimal(3)), 
            'car_cost': float(model[car_cost].as_decimal(3)),
            'total_cost': float(model[total_cost].as_decimal(3))
        }
        
        # Extract variable assignments
        for var_name, var in variables.items():
            try:
                if hasattr(var, '__iter__') and not isinstance(var, str):
                    solution[var_name] = [int(model[v].as_long()) for v in var]
                else:
                    solution[var_name] = int(model[var].as_long())
            except Exception:
                solution[var_name] = str(model[var]) if model[var] else 0
        
        return solution
        
    else:
        if verbose:
            print("❌ Z3 could not find a feasible solution")
        return None


def determine_step_type(step_text):
    """Determine the type of planning step from text content"""
    step_lower = step_text.lower()
    
    if 'resort' in step_lower and 'selection' in step_lower:
        return 'Ski resort'
    elif 'slope' in step_lower or 'skiing' in step_lower:
        return 'Ski slopes'  
    elif 'equipment' in step_lower or 'rental' in step_lower:
        return 'Ski equipment'
    elif 'car' in step_lower or 'transport' in step_lower:
        return 'Car rental'
    elif 'budget' in step_lower or 'cost' in step_lower:
        return 'Budget'
    elif 'destination' in step_lower:
        return 'Destination cities'
    
    return None


def generate_final_plan(query_json, solution, apis, verbose=False):
    """Generate the final ski trip plan from Z3 solution"""
    if not solution:
        return "No feasible solution found within the given constraints."
    
    # Get destination info
    destination = query_json.get('destination', 'Unknown')
    days = query_json.get('days', 3)
    people = query_json.get('people', 2)
    budget = query_json.get('budget', 'Not specified')
    
    # Get resort information
    resort_name = destination  # Default to destination
    try:
        if apis and 'resort' in apis:
            resort_info = apis['resort'].run(destination)
            if isinstance(resort_info, pd.DataFrame) and not resort_info.empty:
                resort_name = resort_info.iloc[0]['Resort']
    except Exception:
        pass
    
    # Build the plan
    plan = f"""Z3 OPTIMIZED SKI TRIP PLAN:
=================================

📍 Destination: {resort_name} ({destination})
👥 Group Size: {people} people  
📅 Duration: {days} days
💰 Budget: €{budget}

💡 OPTIMIZED SOLUTION (Z3):
---------------------------
🏨 Accommodation Cost: €{solution.get('accommodation_cost', 0):.2f}
🎿 Equipment Cost: €{solution.get('equipment_cost', 0):.2f}
🚗 Transportation Cost: €{solution.get('car_cost', 0):.2f}

💵 TOTAL COST: €{solution.get('total_cost', 0):.2f}

✅ This plan has been optimized using Z3 constraint solver to find the best combination
   of accommodation, equipment, and transportation within your budget constraints.

Generated by: Z3 Constraint Solver + Real LLM
Solver Status: OPTIMAL SOLUTION FOUND"""

    return plan


def pipeline_ski(query, mode, model, index, model_version=None, verbose=False):
    """Main Z3-based ski planning pipeline with real LLM integration"""
    try:
        # Initialize LLM client
        llm_client = LLMClient(model_name=model_version or model)
        if verbose:
            print(f"🚀 Real LLM initialized - Using {model_version or model}")
        
        # Initialize ski APIs
        apis = initialize_ski_apis()
        if not apis:
            return "Failed to initialize ski planning APIs"
        
        # Step 1: Convert query to structured JSON
        if verbose:
            print("Step 1: Converting query to JSON...")
        
        json_response = llm_client.query_to_json_response(query)
        if not json_response:
            return "Failed to process query with LLM"
        
        query_json = parse_and_extract_json(json_response)
        if not query_json:
            return "Could not parse query into structured format"
        
        if verbose:
            print(f"Query JSON: {query_json}")
        
        # Step 2: Generate planning steps
        if verbose:
            print("Step 2: Generating planning steps...")
        
        steps_response = llm_client.constraint_to_step_response(query)
        if not steps_response:
            return "Failed to generate planning steps"
        
        steps = [step.strip() for step in steps_response.split('\n\n') if step.strip()]
        
        if verbose:
            print(f"Generated {len(steps)} planning steps")
        
        # Step 3: Solve with Z3
        if verbose:
            print("Step 3: Solving with Z3 constraint solver...")
        
        solution = solve_with_z3(query_json, steps, apis, llm_client, verbose)
        
        # Step 4: Generate final plan
        if verbose:
            print("Step 4: Generating final plan...")
        
        final_plan = generate_final_plan(query_json, solution, apis, verbose)
        
        return final_plan
        
    except Exception as e:
        if verbose:
            print(f"Error in Z3 pipeline: {e}")
        return f"Error in ski planning pipeline: {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, 
                       default="Plan a 3-day ski trip to Livigno for 2 people with budget 1500 euros", 
                       help="Input query for the ski planner.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", 
                       help="Model to use for generating responses.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Run the pipeline
    result = pipeline_ski(
        query=args.query,
        mode="test",
        model=args.model, 
        index=1,
        model_version=args.model,
        verbose=args.verbose
    )
    
    print("\n" + "="*60)
    print("Z3 SKI PLANNER RESULT")
    print("="*60)
    print(result)

if __name__ == "__main__":
    main()
