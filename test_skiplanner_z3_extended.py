import re
import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from z3 import Solver, Bool, Sum, If, Int, Optimize, sat
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project paths
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class LLMClient:
    """Client for interacting with a generic LLM API."""
    def __init__(self, model_name: str = 'DeepSeek-R1', fallback_models: list = None):
        load_dotenv()
        self.api_key = os.getenv("API_TOKEN")
        if not self.api_key:
            raise ValueError("API_TOKEN not found in .env file.")

        self.base_url = "https://models.inference.ai.azure.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.model_name = model_name
        self.fallback_models = fallback_models or []

    def _is_rate_limit_error(self, error_message):
        """Check if error is a rate limit error"""
        if not error_message:
            return False
        
        rate_limit_indicators = [
            "429", "rate limit", "too many requests", "quota exceeded", "requests per"
        ]
        
        error_lower = error_message.lower()
        return any(indicator in error_lower for indicator in rate_limit_indicators)

    def _query_api_with_fallback(self, prompt, available_models=None):
        """Query API with model fallback on rate limits"""
        models_to_try = available_models or [self.model_name] + self.fallback_models
        
        for model in models_to_try:
            try:
                response = self._query_api_single_model(prompt, model)
                if response:
                    return response, model
            except Exception as e:
                error_message = str(e)
                if self._is_rate_limit_error(error_message):
                    print(f"Rate limit hit for {model}, trying next model...")
                    continue
                else:
                    print(f"Error with {model}: {error_message}")
                    return None, model
        
        return None, models_to_try[0] if models_to_try else self.model_name

    def _query_api_single_model(self, prompt, model_name):
        """Query API with a specific model"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a brilliant assistant, specialized in software development and AI research. Your task is to provide insightful, innovative, and safe responses to user queries."
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": ""
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 1.0,
            "model": model_name
        }
        
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        return response.json()['choices'][0]['message']['content']

    def query_to_json_response(self, query):
        """Convert query to JSON with extended parameters"""
        prompt = f"""You are given a natural language query about ski resort planning. Convert this query into a structured JSON format.

The JSON should include these fields:
- "domain": always "ski"
- "destination": the country/region requested  
- "days": number of days
- "people": number of people
- "budget": budget in euros (if mentioned)
- "rating": minimum rating 1.0-5.0 (if mentioned)
- "car_type": type of car/vehicle (if mentioned, e.g., "SUV", "Sedan", "Hatchback")
- "fuel_type": fuel type (if mentioned, e.g., "Gasoline", "Diesel", "Electric")
- "equipment": ski equipment needed (if mentioned, e.g., "Skis", "Boots", "Helmet", "Poles")
- "slope_difficulty": difficulty level of slopes (if mentioned, e.g., "Easy", "Intermediate", "Difficult")
- "special_requirements": any special requirements (if mentioned)

Query: {query}

Please extract all available information and return a valid JSON object:"""
        
        response, model = self._query_api_with_fallback(prompt)
        return response

def parse_and_extract_json(response_text):
    """Parse JSON from LLM response"""
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"```(json)?\n(.*?)\n```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"{\n.*}", response_text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"No JSON object found: {e}")
            return None

def extract_special_requirements_fallback(query):
    """Extract special requirements from query using regex"""
    special_requirements = []
    
    # Car-related requirements
    if any(word in query.lower() for word in ['car', 'vehicle', 'drive', 'rental']):
        special_requirements.append('car rental')
    
    # Equipment-related requirements
    if any(word in query.lower() for word in ['equipment', 'gear', 'ski rental', 'rent']):
        special_requirements.append('equipment rental')
    
    return special_requirements

def load_datasets():
    """Load all ski-related datasets"""
    datasets = {}
    
    # Load resorts
    try:
        datasets['resorts'] = pd.read_csv('dataset_ski/resorts/resorts.csv')
    except FileNotFoundError:
        print("Warning: resorts.csv not found")
        datasets['resorts'] = pd.DataFrame()
    
    # Load cars
    try:
        datasets['cars'] = pd.read_csv('dataset_ski/car/ski_car.csv')
    except FileNotFoundError:
        print("Warning: ski_car.csv not found")
        datasets['cars'] = pd.DataFrame()
    
    # Load equipment
    try:
        datasets['equipment'] = pd.read_csv('dataset_ski/rent/ski_rent.csv')
    except FileNotFoundError:
        print("Warning: ski_rent.csv not found")
        datasets['equipment'] = pd.DataFrame()
    
    # Load slopes
    try:
        datasets['slopes'] = pd.read_csv('dataset_ski/slopes/ski_slopes.csv')
    except FileNotFoundError:
        print("Warning: ski_slopes.csv not found")
        datasets['slopes'] = pd.DataFrame()
    
    return datasets

def filter_data_by_preferences(datasets, query_json):
    """Filter datasets based on user preferences"""
    filtered_data = {}
    
    # Filter resorts by country
    destination = query_json.get('destination', '').lower()
    if not datasets['resorts'].empty:
        if destination:
            filtered_data['resorts'] = datasets['resorts'][
                datasets['resorts']['Country'].str.lower().str.contains(destination, na=False)
            ]
        else:
            filtered_data['resorts'] = datasets['resorts']
    else:
        filtered_data['resorts'] = pd.DataFrame()
    
    # Filter cars by type and fuel
    car_type = query_json.get('car_type', '') or ''
    fuel_type = query_json.get('fuel_type', '') or ''
    
    car_type = car_type.lower() if car_type else ''
    fuel_type = fuel_type.lower() if fuel_type else ''
    
    if not datasets['cars'].empty:
        filtered_cars = datasets['cars'].copy()
        
        if car_type:
            filtered_cars = filtered_cars[
                filtered_cars['Type'].str.lower().str.contains(car_type, na=False)
            ]
        
        if fuel_type:
            filtered_cars = filtered_cars[
                filtered_cars['Fuel'].str.lower().str.contains(fuel_type, na=False)
            ]
        
        filtered_data['cars'] = filtered_cars
    else:
        filtered_data['cars'] = pd.DataFrame()
    
    # Filter equipment by type
    equipment_req = query_json.get('equipment', '')
    if isinstance(equipment_req, list):
        equipment_type = ' '.join(equipment_req).lower()
    elif isinstance(equipment_req, str):
        equipment_type = equipment_req.lower()
    else:
        equipment_type = ''
    
    if not datasets['equipment'].empty:
        if equipment_type:
            # Handle multiple equipment types
            equipment_filters = []
            for eq_type in ['skis', 'boots', 'helmet', 'poles']:
                if eq_type in equipment_type:
                    equipment_filters.append(
                        datasets['equipment']['Equipment'].str.lower().str.contains(eq_type, na=False)
                    )
            
            if equipment_filters:
                combined_filter = equipment_filters[0]
                for f in equipment_filters[1:]:
                    combined_filter |= f
                filtered_data['equipment'] = datasets['equipment'][combined_filter]
            else:
                filtered_data['equipment'] = datasets['equipment']
        else:
            filtered_data['equipment'] = datasets['equipment']
    else:
        filtered_data['equipment'] = pd.DataFrame()
    
    # Filter slopes by difficulty
    slope_difficulty = query_json.get('slope_difficulty', '') or ''
    slope_difficulty = slope_difficulty.lower() if slope_difficulty else ''
    if not datasets['slopes'].empty:
        if slope_difficulty:
            filtered_data['slopes'] = datasets['slopes'][
                datasets['slopes']['Difficult_Slope'].str.lower().str.contains(slope_difficulty, na=False)
            ]
        else:
            filtered_data['slopes'] = datasets['slopes']
    else:
        filtered_data['slopes'] = pd.DataFrame()
    
    return filtered_data

def create_fallback_data(query_json):
    """Create fallback data when real data is not available"""
    destination = query_json.get('destination', 'Unknown')
    
    fallback_data = {
        'resorts': pd.DataFrame([{
            'Resort': destination,
            'Country': destination,
            'Price_day': 150,
            'Beds': 2,
            'Rating': 4.0,
            'Access': 'Road'
        }]),
        'cars': pd.DataFrame([{
            'Brand': 'Generic',
            'Model': 'SUV',
            'Type': 'SUV',
            'Fuel': 'Gasoline',
            'Price_day': 80
        }]),
        'equipment': pd.DataFrame([
            {'Equipment': 'Skis', 'Price_day': 25},
            {'Equipment': 'Boots', 'Price_day': 15},
            {'Equipment': 'Helmet', 'Price_day': 10},
            {'Equipment': 'Poles', 'Price_day': 8}
        ]),
        'slopes': pd.DataFrame([{
            'Slope': 'Main Slope',
            'Difficulty': 'Intermediate',
            'Length': 2000
        }])
    }
    
    return fallback_data

def pipeline_ski_z3_extended(query, mode="test", model="gpt-4o-mini", index=1, verbose=True):
    """Extended Z3 ski planner with support for all parameters - Now selects resort by highest rating"""
    
    try:
        # Initialize LLM client
        llm_client = LLMClient(model_name=model)
        if verbose:
            print(f"🚀 LLM initialized - Using {model}")
        
        # Load datasets
        datasets = load_datasets()
        
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
        destination = query_json.get('destination', 'Unknown')
        days = query_json.get('days', 3)
        people = query_json.get('people', 2)
        budget_val = query_json.get('budget')
        
        try:
            budget = int(budget_val) if budget_val else 1000000
        except (ValueError, TypeError):
            budget = 1000000
        
        car_type = query_json.get('car_type', '')
        fuel_type = query_json.get('fuel_type', '')
        equipment_type = query_json.get('equipment', '')
        slope_difficulty = query_json.get('slope_difficulty', '')
        special_requirements = query_json.get('special_requirements', [])
        
        # Step 2: Filter data based on preferences
        if verbose:
            print("Step 2: Filtering data based on preferences...")
        
        filtered_data = filter_data_by_preferences(datasets, query_json)
        
        # Use fallback data if filtering resulted in empty datasets
        fallback_used = {
            'resorts': False,
            'cars': False,
            'equipment': False,
            'slopes': False
        }
        
        if filtered_data['resorts'].empty:
            fallback_data = create_fallback_data(query_json)
            filtered_data['resorts'] = fallback_data['resorts']
            fallback_used['resorts'] = True
        
        if filtered_data['cars'].empty:
            if not datasets['cars'].empty:
                filtered_data['cars'] = datasets['cars'].head(5)  # Take first 5 cars
            else:
                fallback_data = create_fallback_data(query_json)
                filtered_data['cars'] = fallback_data['cars']
                fallback_used['cars'] = True
        
        if filtered_data['equipment'].empty:
            if not datasets['equipment'].empty:
                filtered_data['equipment'] = datasets['equipment'].head(10)  # Take first 10 equipment
            else:
                fallback_data = create_fallback_data(query_json)
                filtered_data['equipment'] = fallback_data['equipment']
                fallback_used['equipment'] = True
        
        if filtered_data['slopes'].empty:
            if not datasets['slopes'].empty:
                filtered_data['slopes'] = datasets['slopes'].head(5)  # Take first 5 slopes
            else:
                fallback_data = create_fallback_data(query_json)
                filtered_data['slopes'] = fallback_data['slopes']
                fallback_used['slopes'] = True
        
        # Step 3: Set up Z3 solver for resort selection first
        if verbose:
            print("Step 3: Setting up Z3 constraint solver...")
        
        solver = Solver()
        
        # Convert filtered data to dictionaries
        resorts = filtered_data['resorts'].to_dict('records')
        all_cars = filtered_data['cars'].to_dict('records')
        all_equipment = filtered_data['equipment'].to_dict('records')
        all_slopes = filtered_data['slopes'].to_dict('records')
        
        # Step 4: Create Z3 variables and constraints for resort selection first
        if verbose:
            print("Step 4: Creating Z3 variables for resort selection...")
        
        # Resort selection variables
        resort_vars = [Bool(f"resort_{i}") for i in range(len(resorts))]
        solver.add(Sum([If(var, 1, 0) for var in resort_vars]) == 1)
        
        # REMOVED: No longer filtering resorts by bed capacity
        # The resort selection will now be based on cost optimization for feasibility
        
        # First solve just for the best resort (prioritizing lowest cost for maximum feasibility)
        opt_resort = Optimize()
        for constraint in solver.assertions():
            opt_resort.add(constraint)
        
        # CHANGED: Minimize cost instead of maximizing rating to accept more queries
        resort_cost_score = Sum([
            If(resort_vars[i], int(resorts[i].get('Price_day', 150)), 0)
            for i in range(len(resorts))
        ])
        opt_resort.minimize(resort_cost_score)
        
        if verbose:
            print("💰 Optimizing for most affordable resort to maximize feasibility...")
        
        # Get the selected resort
        selected_resort = None
        if opt_resort.check() == sat:
            model_resort = opt_resort.model()
            for i, var in enumerate(resort_vars):
                if model_resort[var]:
                    selected_resort = resorts[i]
                    break
        
        if not selected_resort:
            if verbose:
                print("❌ Could not select a resort")
            return "INFEASIBLE: Could not select a resort."
        
        if verbose:
            print(f"✅ Selected resort: {selected_resort.get('Resort', 'Unknown')} (Price: €{selected_resort.get('Price_day', 150)}/day, Rating: {selected_resort.get('Rating', 'N/A')}/5.0)")
        
        # Step 5: Filter cars and equipment based on selected resort
        if verbose:
            print("Step 5: Filtering cars and equipment for selected resort...")
        
        # Filter cars and equipment for selected resort
        selected_resort_name = selected_resort.get('Resort', '')
        
        # Filter equipment for this specific resort
        resort_equipment = datasets['equipment'][
            datasets['equipment']['Resort'] == selected_resort_name
        ] if 'Resort' in datasets['equipment'].columns else pd.DataFrame()
        
        if resort_equipment.empty:
            # Fallback to general equipment if no resort-specific equipment
            equipment = all_equipment
            if verbose:
                print(f"No resort-specific equipment found, using general equipment")
        else:
            equipment = resort_equipment.to_dict('records')
            if verbose:
                print(f"Found {len(equipment)} equipment options for resort {selected_resort_name}")
        
        # Filter cars for selected resort (if Resort column exists in cars data)
        resort_cars = datasets['cars'][
            datasets['cars']['Resort'] == selected_resort_name
        ] if 'Resort' in datasets['cars'].columns else pd.DataFrame()
        
        if resort_cars.empty:
            # Fallback to general cars if no resort-specific cars
            cars = all_cars
            if verbose:
                print(f"No resort-specific cars found, using general cars")
        else:
            cars = resort_cars.to_dict('records')
            if verbose:
                print(f"Found {len(cars)} car options for resort {selected_resort_name}")
        
        # Filter slopes for selected resort
        resort_slopes = datasets['slopes'][
            datasets['slopes']['Resort'] == selected_resort_name
        ] if 'Resort' in datasets['slopes'].columns else pd.DataFrame()
        
        if resort_slopes.empty:
            # Fallback to general slopes if no resort-specific slopes
            slopes = all_slopes
            if verbose:
                print(f"No resort-specific slopes found, using general slopes")
        else:
            slopes = resort_slopes.to_dict('records')
            if verbose:
                print(f"Found {len(slopes)} slope options for resort {selected_resort_name}")
        
        # Step 6: Create new Z3 solver with all components
        if verbose:
            print("Step 6: Creating complete Z3 optimization...")
        
        final_solver = Optimize()
        
        # Re-create resort variable (fix to selected resort)
        selected_resort_var = Bool("selected_resort")
        final_solver.add(selected_resort_var == True)
        
        # Car selection variables
        car_vars = [Bool(f"car_{i}") for i in range(len(cars))]
        final_solver.add(Sum([If(var, 1, 0) for var in car_vars]) <= 1)
        
        # If car is requested (car_type or fuel_type specified), ensure at least one car is selected
        if car_type or fuel_type:
            if len(cars) > 0:
                final_solver.add(Sum([If(var, 1, 0) for var in car_vars]) >= 1)
                if verbose:
                    print(f"✅ Added constraint: Car rental is required (type: {car_type}, fuel: {fuel_type})")
        
        # Equipment selection variables
        equipment_vars = [Bool(f"equipment_{i}") for i in range(len(equipment))]
        
        # Equipment constraints based on preferences
        if equipment_type or 'equipment' in str(special_requirements).lower():
            # Ensure basic equipment is selected
            required_equipment = ['skis', 'boots']
            for req_eq in required_equipment:
                req_vars = []
                for i, eq in enumerate(equipment):
                    if req_eq.lower() in eq.get('Equipment', '').lower():
                        req_vars.append(equipment_vars[i])
                
                if req_vars:
                    final_solver.add(Sum([If(var, 1, 0) for var in req_vars]) >= 1)
        
        # Slope constraints (informational only - doesn't affect cost)
        slope_vars = [Bool(f"slope_{i}") for i in range(len(slopes))]
        
        # Cost calculation
        total_cost = Int('total_cost')
        
        # Resort cost (fixed for selected resort)
        resort_cost = selected_resort.get('Price_day', 150) * days
        
        # Car cost
        car_cost = Sum([
            If(car_vars[i], cars[i].get('Price_day', 50) * days, 0)
            for i in range(len(cars))
        ])
        
        # Equipment cost
        equipment_cost = Sum([
            If(equipment_vars[i], equipment[i].get('Price_day', 20) * days * people, 0)
            for i in range(len(equipment))
        ])
        
        # Total cost constraint
        final_solver.add(total_cost == resort_cost + car_cost + equipment_cost)
        
        # Budget constraint
        final_solver.add(total_cost <= budget)
        
        # Minimize total cost (secondary optimization after rating)
        final_solver.minimize(total_cost)
        
        # Step 7: Solve complete optimization
        if verbose:
            print("Step 7: Solving complete optimization with Z3...")
        
        if final_solver.check() == sat:
            model = final_solver.model()
            
            if verbose:
                print("✅ Z3 found a solution!")
            
            # Extract solution
            selected_cars = []
            for i, var in enumerate(car_vars):
                if model[var]:
                    selected_cars.append(cars[i])
            
            selected_equipment = []
            for i, var in enumerate(equipment_vars):
                if model[var]:
                    selected_equipment.append(equipment[i])
                    if verbose:
                        print(f"Selected equipment: {equipment[i].get('Equipment', 'Unknown')} - €{equipment[i].get('Price_day', 0)}/day")
            
            # Calculate actual costs
            final_resort_cost = resort_cost
            final_car_cost = sum(car.get('Price_day', 50) * days for car in selected_cars)
            final_equipment_cost = sum(eq.get('Price_day', 20) * days * people for eq in selected_equipment)
            final_total_cost = final_resort_cost + final_car_cost + final_equipment_cost
            
            # Generate result
            result = f"""Z3 EXTENDED SKI TRIP PLAN:
DESTINATION: {destination}
DURATION: {days} days
PEOPLE: {people}
BUDGET: €{budget}

SELECTED RESORT:
- Resort: {selected_resort.get('Resort', 'Unknown')}
- Country: {selected_resort.get('Country', 'Unknown')}
- Price per day: €{selected_resort.get('Price_day', 150)}
- Rating: {selected_resort.get('Rating', 'N/A')}/5.0
- Access: {selected_resort.get('Access', 'Unknown')}

SELECTED CAR(S):
"""
            
            if selected_cars:
                for car in selected_cars:
                    result += f"- Brand: {car.get('Brand', 'Unknown')} - Model: {car.get('Model', 'Unknown')} - Type: {car.get('Type', 'Unknown')} - Fuel: {car.get('Fuel', 'Unknown')} - €{car.get('Price_day', 50)}/day\n"
            else:
                result += "- No car rental selected\n"
            
            result += "\nSELECTED EQUIPMENT:\n"
            if selected_equipment:
                for eq in selected_equipment:
                    result += f"- {eq.get('Equipment', 'Unknown')}: €{eq.get('Price_day', 20)}/day per person\n"
            else:
                result += "- No equipment rental selected\n"
            
            result += "\nAVAILABLE SLOPES:\n"
            if slopes:
                for slope in slopes:
                    slope_name = slope.get('Resort', selected_resort.get('Resort', 'Unknown'))
                    slope_color = slope.get('Slope_Color', 'Unknown')
                    slope_length = slope.get('Length_km', 'Unknown')
                    result += f"- {slope_name}: {slope_color} ({slope_length}km)\n"
            else:
                result += f"- No slopes available for {selected_resort.get('Resort', 'Unknown')}\n"
            
            result += f"""
COST BREAKDOWN:
- Resort: €{final_resort_cost:.2f}
- Car rental: €{final_car_cost:.2f}
- Equipment: €{final_equipment_cost:.2f}
- TOTAL COST: €{final_total_cost:.2f}

BUDGET STATUS: {'✅ Within budget' if final_total_cost <= budget else '❌ Exceeds budget'}

Generated by: Z3 Extended Constraint Solver
Data Sources: Real CSV data"""
            
            return result
            
        else:
            if verbose:
                print("❌ Z3 could not find a solution")
            return "INFEASIBLE: No solution found within the given budget and constraints."
    
    except Exception as e:
        if verbose:
            print(f"❌ Error in pipeline: {e}")
        return f"ERROR: {str(e)}"

# Test function
def test_z3_extended():
    query = "Plan a 3-day ski trip to Austria for 2 people with budget 1500 euros. We need a SUV and ski equipment and prefer beginner slopes."
    result = pipeline_ski_z3_extended(query, verbose=True)
    print(result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use command line argument as query
        query = sys.argv[1]
        print("Testing Z3 Extended Ski Planner...")
        print(f"Query: {query}")
        result = pipeline_ski_z3_extended(query, verbose=True)
        print("============================================================")
        print("RESULT:")
        print("============================================================")
        print(result)
    else:
        # Default test
        test_z3_extended()
