import re
import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from z3 import *
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Use tools_ski for ski-specific APIs
from tools_ski.apis import *

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
                    "content": "You are a helpful assistant designed to extract information from user queries and generate planning information."
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
        """Convert query to structured JSON format"""
        prompt = f"""
Extract the following information from this ski trip query and return as JSON:

Query: {query}

Extract:
- domain: "ski"
- destination: the ski resort or city mentioned
- days: number of days for the trip
- people: number of people
- budget: budget amount in euros (number only)
- special_requirements: list of special requirements (car rental, equipment, etc.)

Return only valid JSON in this format:
{{"domain": "ski", "destination": "resort_name", "days": 3, "people": 2, "budget": 1500, "special_requirements": []}}
"""
        return self._query_api(prompt)

def parse_and_extract_json(response_text):
    """Extract JSON from LLM response"""
    try:
        # Try to find JSON in code blocks first
        import re
        match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Try to find JSON object directly
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        
        # Try to parse the entire response as JSON
        return json.loads(response_text)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None

def load_resort_data():
    """Load resort data from CSV"""
    try:
        df = pd.read_csv('dataset_ski/resorts/resorts.csv')
        return df
    except Exception as e:
        print(f"Error loading resort data: {e}")
        return pd.DataFrame()

def pipeline_ski(query, mode, model, index, model_version=None, verbose=False):
    """Z3-based ski trip planning pipeline"""
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(model_name=model_version or model)
        if verbose:
            print(f"🚀 Real LLM initialized - Using {model_version or model}")
    except Exception as e:
        if verbose:
            print(f"❌ Failed to initialize LLM: {e}")
        return None
    
    # Initialize ski APIs
    if verbose:
        print("Initializing ski planner...")
    try:
        ski_resorts = SkiResorts()
        ski_slopes = SkiSlopes()
        ski_rent = SkiRent()
        ski_car = SkiCar()
        if verbose:
            print("Ski Resorts loaded.")
            print("Ski Slopes loaded.")
            print("Ski Equipment Rental loaded.")
            print("Ski Car Rental loaded.")
            print("Ski APIs initialized successfully!")
    except Exception as e:
        if verbose:
            print(f"❌ Failed to initialize Ski APIs: {e}")
        return None
    
    try:
        # Step 1: Convert query to JSON
        if verbose:
            print("Step 1: Converting query to JSON...")
        
        json_response = llm_client.query_to_json_response(query)
        if not json_response:
            if verbose:
                print("❌ Failed to get JSON response from LLM")
            return None
        
        query_json = parse_and_extract_json(json_response)
        if not query_json:
            if verbose:
                print("❌ Failed to parse JSON from LLM response")
            return None
        
        if verbose:
            print(f"Query JSON: {query_json}")
        
        # Extract parameters
        destination = query_json.get('destination', 'Livigno')
        days = query_json.get('days', 3)
        people = query_json.get('people', 2)
        budget = query_json.get('budget', 1500)
        special_requirements = query_json.get('special_requirements', [])
        
        # Step 2: Set up Z3 solver
        if verbose:
            print("Step 2: Setting up Z3 constraint solver...")
        
        solver = Solver()
        
        # Step 3: Get available options from APIs
        if verbose:
            print("Step 3: Querying ski APIs for available options...")
        
        # Get resort options
        resort_data = ski_resorts.run(destination)
        if isinstance(resort_data, pd.DataFrame) and not resort_data.empty:
            resorts = resort_data.to_dict('records')
        else:
            # Fallback data
            resorts = [{'Resort': destination, 'Price_day': 150, 'Beds': 2, 'Rating': 4}]
        
        # Get slope options
        slope_data = ski_slopes.run(destination)
        if isinstance(slope_data, pd.DataFrame) and not slope_data.empty:
            slopes = slope_data.to_dict('records')
        else:
            slopes = [{'Difficulty': 'Blue', 'Length': 2.5, 'Price': 50}]
        
        # Get equipment options
        equipment_data = ski_rent.run(destination)
        if isinstance(equipment_data, pd.DataFrame) and not equipment_data.empty:
            equipment = equipment_data.to_dict('records')
        else:
            equipment = [{'Equipment': 'Skis', 'Price_day': 25}, {'Equipment': 'Boots', 'Price_day': 15}]
        
        # Get car options
        car_data = ski_car.run(destination)
        if isinstance(car_data, pd.DataFrame) and not car_data.empty:
            cars = car_data.to_dict('records')
        else:
            cars = [{'Car': 'SUV', 'Price_day': 80, 'Fuel': 'Petrol'}]
        
        # Step 4: Create Z3 variables and constraints
        if verbose:
            print("Step 4: Creating Z3 variables and constraints...")
        
        # Resort selection variables
        resort_vars = []
        for i, resort in enumerate(resorts):
            var = Bool(f"resort_{i}")
            resort_vars.append(var)
        
        # Exactly one resort must be selected
        solver.add(Sum([If(var, 1, 0) for var in resort_vars]) == 1)
        
        # Equipment selection variables
        equipment_vars = []
        for i, eq in enumerate(equipment):
            var = Bool(f"equipment_{i}")
            equipment_vars.append(var)
        
        # Car selection variables
        car_vars = []
        for i, car in enumerate(cars):
            var = Bool(f"car_{i}")
            car_vars.append(var)
        
        # At most one car (car rental is optional)
        solver.add(Sum([If(var, 1, 0) for var in car_vars]) <= 1)
        
        # Cost calculation
        total_cost = Int('total_cost')
        
        # Resort cost (accommodation)
        resort_cost = Sum([
            If(resort_vars[i], 
               resorts[i].get('Price_day', 150) * days * people, 
               0) 
            for i in range(len(resorts))
        ])
        
        # Equipment cost
        equipment_cost = Sum([
            If(equipment_vars[i], 
               equipment[i].get('Price_day', 25) * days * people, 
               0) 
            for i in range(len(equipment))
        ])
        
        # Car cost
        car_cost = Sum([
            If(car_vars[i], 
               cars[i].get('Price_day', 80) * days, 
               0) 
            for i in range(len(cars))
        ])
        
        # Total cost constraint
        solver.add(total_cost == resort_cost + equipment_cost + car_cost)
        
        # Budget constraint
        solver.add(total_cost <= budget)
        
        if verbose:
            print(f"Added budget constraint: total_cost <= {budget}")
        
        # Step 5: Solve with Z3
        if verbose:
            print("Step 5: Solving with Z3...")
        
        # Try to minimize cost
        opt = Optimize()
        for constraint in solver.assertions():
            opt.add(constraint)
        opt.minimize(total_cost)
        
        result = opt.check()
        
        if result == sat:
            model = opt.model()
            if verbose:
                print("✅ Z3 found a solution!")
            
            # Extract solution
            selected_resort = None
            selected_equipment = []
            selected_car = None
            
            for i, var in enumerate(resort_vars):
                if model[var]:
                    selected_resort = resorts[i]
                    break
            
            for i, var in enumerate(equipment_vars):
                if model[var]:
                    selected_equipment.append(equipment[i])
            
            for i, var in enumerate(car_vars):
                if model[var]:
                    selected_car = cars[i]
                    break
            
            # Calculate actual costs
            resort_cost_val = selected_resort.get('Price_day', 150) * days * people if selected_resort else 0
            equipment_cost_val = sum(eq.get('Price_day', 25) * days * people for eq in selected_equipment)
            car_cost_val = selected_car.get('Price_day', 80) * days if selected_car else 0
            total_cost_val = resort_cost_val + equipment_cost_val + car_cost_val
            
            # Generate plan
            plan = f"""Z3 SKI TRIP PLAN:

DESTINATION: {destination}
DURATION: {days} days
PEOPLE: {people}
BUDGET: €{budget}

SELECTED RESORT:
- Resort: {selected_resort.get('Resort', destination) if selected_resort else destination}
- Price per day: €{selected_resort.get('Price_day', 150) if selected_resort else 150}
- Beds available: {selected_resort.get('Beds', people) if selected_resort else people}
- Rating: {selected_resort.get('Rating', 4) if selected_resort else 4}/5

EQUIPMENT RENTAL:
"""
            
            if selected_equipment:
                for eq in selected_equipment:
                    plan += f"- {eq.get('Equipment', 'Equipment')}: €{eq.get('Price_day', 25)}/day\n"
            else:
                plan += "- No equipment rental selected\n"
            
            plan += f"\nCAR RENTAL:\n"
            if selected_car:
                plan += f"- {selected_car.get('Car', 'Car')}: €{selected_car.get('Price_day', 80)}/day\n"
                plan += f"- Fuel type: {selected_car.get('Fuel', 'Petrol')}\n"
            else:
                plan += "- No car rental selected\n"
            
            plan += f"""
COST BREAKDOWN:
- Accommodation: €{resort_cost_val:.2f}
- Equipment: €{equipment_cost_val:.2f}
- Car rental: €{car_cost_val:.2f}
- TOTAL COST: €{total_cost_val:.2f}

BUDGET STATUS: {'✅ Within budget' if total_cost_val <= budget else '❌ Over budget'}

Generated by: Z3 Constraint Solver with Real LLM Integration
Solver: Z3 SMT Solver
Model: {model_version or model}
"""
            
            return plan
            
        else:
            if verbose:
                print("❌ Z3 could not find a solution within the budget constraints")
            
            # Return a basic plan even if no optimal solution found
            return f"""Z3 SKI TRIP PLAN:

DESTINATION: {destination}
DURATION: {days} days
PEOPLE: {people}
BUDGET: €{budget}

❌ No optimal solution found within budget constraints.

The Z3 solver could not find a combination of resort, equipment, and car rental
that fits within the specified budget of €{budget}.

Suggestions:
- Increase budget
- Reduce number of days
- Skip optional services (car rental, equipment)

Generated by: Z3 Constraint Solver
Status: No feasible solution
"""
    
    except Exception as e:
        if verbose:
            print(f"❌ Error in Z3 pipeline: {e}")
            import traceback
            traceback.print_exc()
        return None

def main():
    """Test the Z3 ski planner"""
    test_query = "Plan a 3-day ski trip to Livigno for 2 people with budget 1500 euros"
    print("Testing Z3 Ski Planner...")
    print(f"Query: {test_query}")
    
    result = pipeline_ski(
        query=test_query,
        mode="test",
        model="gpt-4o-mini",
        index=1,
        verbose=True
    )
    
    if result:
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(result)
    else:
        print("❌ Failed to generate plan")

if __name__ == "__main__":
    main()
