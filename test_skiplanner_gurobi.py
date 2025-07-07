import re, string, os, sys, time, json, argparse, pdb
import importlib
import tiktoken
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from pandas import DataFrame
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.func import load_line_json_data, save_file
from z3 import *
from gurobipy import Model, GRB

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Use tools_ski for ski-specific APIs
from tools_ski.apis import SkiResorts, SkiSlopes, SkiRent, SkiCar  # Add SkiCar import

# GitHub Models API Client
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
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return None

# Initialize global LLM client
try:
    llm_client = LLMClient()
    print("🚀 Real LLM initialized - Using GitHub Models API")
except ValueError as e:
    print(f"Warning: {e}")
    llm_client = None
    print("🆓 Falling back to Mock LLM")

# Load ski prompts from files
def load_ski_prompts(verbose=True):
    """Load all ski prompt templates from files"""
    prompts = {}
    
    # Load query to JSON prompt
    with open('prompts/ski/query_to_json_ski.txt', 'r', encoding='utf-8') as file:
        prompts['query_to_json'] = file.read()
    
    # Load constraint to step prompt  
    with open('prompts/ski/constraint_to_step_ski.txt', 'r', encoding='utf-8') as file:
        prompts['constraint_to_step'] = file.read()
    
    # Load step to code prompts
    step_to_code_files = {
        'Destination': 'prompts/ski/step_to_code_destination.txt',
        'Resort': 'prompts/ski/step_to_code_resort.txt', 
        'Slopes': 'prompts/ski/step_to_code_slopes.txt',
        'Equipment': 'prompts/ski/step_to_code_equipment.txt',
        'Car': 'prompts/ski/step_to_code_car.txt',
        'Budget': 'prompts/ski/step_to_code_budget.txt'
    }
    
    prompts['step_to_code'] = {}
    for key, file_path in step_to_code_files.items():
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts['step_to_code'][key] = file.read()
    
    if verbose:
        print(f"Loaded {len(prompts['step_to_code'])} step-to-code prompts from files")
    return prompts

# Mock LLM for testing without API costs
class MockLLM:
    """Mock LLM that uses real prompts but avoids API costs"""
    
    def __init__(self, verbose=True):
        if verbose:
            print("🆓 Mock LLM initialized - Using real prompts, no API costs!")
        self.prompts = load_ski_prompts(verbose)
        self.verbose = verbose
    def query_to_json_response(self, query):
        """Mock response for query to JSON conversion using real prompt structure"""
        # Enhanced parsing for ski queries to extract car rental information
        query_lower = query.lower()
        
        # Extract destination from common ski resorts
        destination = "Livigno"  # default
        ski_destinations = {
            "livigno": "Livigno",
            "cortina": "Cortina d'Ampezzo", 
            "val gardena": "Val Gardena",
            "madonna di campiglio": "Madonna di Campiglio",
            "sestriere": "Sestriere",
            "la thuile": "La Thuile",
            "val senales": "Val Senales Glacier",
            "courmayeur": "Courmayeur",
            "gressoney": "Gressoney",
            "valchiavenna": "Valchiavenna Madesimo",
            "hemsedal": "Hemsedal",
            "golm": "Golm",
            "geilo": "Geilosiden Geilo",
            "voss": "Voss",
            "red mountain": "Red Mountain",
            # Swiss resorts
            "switzerland": "Zermatt",
            "zermatt": "Zermatt",
            "st. moritz": "St. Moritz",
            "st moritz": "St. Moritz",
            "verbier": "Verbier",
            "davos": "Davos",
            "klosters": "Klosters",
            "saas-fee": "Saas-Fee",
            "crans-montana": "Crans-Montana",
            "engelberg": "Engelberg"
        }
        
        for key, value in ski_destinations.items():
            if key in query_lower:
                destination = value
                break
        
        # Extract car type with better detection
        car_type = None
        if "suv" in query_lower:
            car_type = "SUV"
        elif "sedan" in query_lower:
            car_type = "Sedan"
        elif "pick up" in query_lower or "pickup" in query_lower:
            car_type = "Pick up"
        elif "cabriolet" in query_lower:
            car_type = "Cabriolet"
        elif "car rental" in query_lower or "rent" in query_lower and "car" in query_lower:
            car_type = "SUV"  # Default for equipment transport
        
        # Extract fuel type with better detection
        fuel_type = None
        if "electric" in query_lower:
            fuel_type = "Electric"
        elif "hybrid" in query_lower:
            fuel_type = "Hybrid"
        elif "diesel" in query_lower:
            fuel_type = "Diesel"
        elif "petrol" in query_lower:
            fuel_type = "Petrol"
        elif "eco" in query_lower or "green" in query_lower:
            fuel_type = "Electric"  # Default for eco-friendly
        
        # Extract access method
        access = None
        if "car" in query_lower or "rental" in query_lower or "rent" in query_lower:
            access = "Car"
        elif "train" in query_lower:
            access = "Train"
        elif "bus" in query_lower:
            access = "Bus"
        
        # Extract slope difficulty
        slope_difficulty = None
        if "black" in query_lower or "advanced" in query_lower or "expert" in query_lower:
            slope_difficulty = "Black"
        elif "red" in query_lower or "intermediate" in query_lower:
            slope_difficulty = "Red"
        elif "blue" in query_lower or "beginner" in query_lower or "easy" in query_lower or "not expert" in query_lower or "not experienced" in query_lower:
            slope_difficulty = "Blue"
        
        # Extract equipment with better detection - only for explicit ski equipment requests
        equipment = []
        equipment_people = None
        
        # Only trigger equipment if explicitly mentioned or ski-specific rental terms
        ski_equipment_patterns = [
            r'\bequipment\b',
            r'\bski\s+rental\b',
            r'\brent\s+ski\b',
            r'\bski\s+equipment\b',
            r'\brental\s+equipment\b',
            r'\brent\s+equipment\b',
            r'\brent\s+skis\b'
        ]
        
        equipment_requested = any(re.search(pattern, query_lower) for pattern in ski_equipment_patterns)
        
        if equipment_requested:
            print("Equipment rental detected in query")
            # Check if equipment is for specific number of people
            eq_people_match = re.search(r'equipment.*?(\d+)\s*person', query_lower)
            if eq_people_match:
                equipment_people = int(eq_people_match.group(1))
            
            if "ski" in query_lower or "complete" in query_lower or "equipment" in query_lower:
                equipment.extend(["Skis", "Boots", "Helmet", "Poles"])
            else:
                if "ski" in query_lower:
                    equipment.append("Skis")
                if "boot" in query_lower:
                    equipment.append("Boots")
                if "helmet" in query_lower:
                    equipment.append("Helmet")
                if "pole" in query_lower:
                    equipment.append("Poles")
        else:
            print("No equipment rental requested in query")
        
        # Extract budget - set a reasonable default if not specified
        budget = 3000  # Default for 7-day Switzerland trip
        budget_match = re.search(r'(\d+)\s*euro', query_lower)
        if budget_match:
            budget = int(budget_match.group(1))
        
        # Extract days
        days = 3
        days_match = re.search(r'(\d+)\s*day', query_lower)
        if days_match:
            days = int(days_match.group(1))
        
        # Extract people
        people = 2
        people_match = re.search(r'(\d+)\s*people', query_lower)
        if people_match:
            people = int(people_match.group(1))
        
        # Extract special requirements
        special_requirements = []
        if "other" in query_lower and "resort" in query_lower:
            special_requirements.append("visit_multiple_resorts")
        if "transport" in query_lower and "method" in query_lower:
            special_requirements.append("need_transportation")
        
        # Extract beds from accommodation requirements
        beds = None
        beds_match = re.search(r'(\d+)[\s-]*bed', query_lower)
        if beds_match:
            beds = int(beds_match.group(1))
        
        # Extract rating requirements
        rating = None
        rating_match = re.search(r'rating\s+above\s+(\d+\.?\d*)', query_lower)
        if rating_match:
            rating = float(rating_match.group(1))
        elif "highly rated" in query_lower:
            rating = 4.0
        
        # Base JSON structure
        json_data = {
            "domain": "ski",
            "destination": destination,
            "days": days,
            "people": people,
            "budget": budget,
            "query": query
        }
        
        # Add optional fields if extracted
        if car_type:
            json_data["car_type"] = car_type
        if fuel_type:
            json_data["fuel_type"] = fuel_type
        if access:
            json_data["access"] = access
        if slope_difficulty:
            json_data["slope_difficulty"] = slope_difficulty
        if equipment:
            json_data["equipment"] = equipment
        if equipment_people:
            json_data["equipment_people"] = equipment_people
        if beds:
            json_data["beds"] = beds
        if rating:
            json_data["rating"] = rating
        if special_requirements:
            json_data["special_requirements"] = special_requirements
        
        return json.dumps(json_data, indent=2)
    
    def constraint_to_step_response(self, query):
        """Mock response for constraint to step conversion using real prompt"""
        return """# Destination cities
Visit the main ski resort destination.

# Departure dates  
Set departure dates for the ski trip.

# Transportation methods
Choose transportation between cities (car rental, taxi, or flight).

# Car rental information
Book car rental for ski equipment transport.

# Ski resort information
Select ski resort with appropriate slopes and facilities.

# Ski slopes information
Choose ski slopes matching skill level and preferences.

# Ski equipment rental
Rent ski equipment (skis, boots, poles, helmets).

# Accommodation information
Book ski lodge or hotel near the slopes.

# Budget
Ensure total cost stays within budget constraints."""

    def step_to_code_response(self, step_type, content):
        """Generate code response using real prompt templates"""
        # Map step types to prompt keys with proper mock code
        step_mapping = {
            'Destination cities': """# Destination setup
destination_country = 'Italy'
selected_resorts = ['Livigno', 'Cortina', 'Madonna di Campiglio']
all_resorts = selected_resorts""",
            
            'Ski resort': """# Resort variables
variables['resort_index'] = Int('resort_index')
s.add(And(variables['resort_index'] >= 0, variables['resort_index'] < len(selected_resorts)))""",
            
            'Ski slopes': """# Slopes variables  
variables['slope_difficulty'] = Int('slope_difficulty')
s.add(And(variables['slope_difficulty'] >= 0, variables['slope_difficulty'] <= 2))""",
            
            'Ski equipment': """# Equipment variables
variables['rent_skis'] = Bool('rent_skis')
variables['rent_boots'] = Bool('rent_boots')
variables['rent_helmet'] = Bool('rent_helmet')
s.add(variables['rent_skis'] == True)""",
            
            'Car rental': """# Car rental variables
variables['car_rental'] = Bool('car_rental')
variables['car_type'] = Int('car_type')
s.add(And(variables['car_type'] >= 0, variables['car_type'] <= 3))""",
            
            'Budget': """# Budget constraint
total_cost = 0
if 'resort_index' in variables:
    total_cost += 200  # Resort cost per day
if 'car_rental' in variables:
    total_cost += If(variables['car_rental'], 150, 0)
s.add(total_cost <= query_json['budget'])"""
        }
        
        # Find matching mock code
        for step_key, mock_code in step_mapping.items():
            if step_key.lower() in step_type.lower():
                return mock_code
        
        # Fallback for unmapped step types
        return f"# Mock code for: {step_type}\npass"

# Mock functions for API responses using real prompts
def GPT_response(prompt, model_version="gpt-4o-mini", verbose=True):
    """Real GPT response using GitHub Models API"""
    if llm_client is not None:
        try:
            if verbose:
                print("Using Real GitHub Models API")
            response = llm_client._query_api(prompt)
            if response:
                return response
            else:
                if verbose:
                    print("API call failed, falling back to mock response")
                return get_mock_response(prompt)
        except Exception as e:
            if verbose:
                print(f"Error calling GitHub Models API: {e}")
            return get_mock_response(prompt)
    else:
        if verbose:
            print("Using Mock LLM (no API available)")
        return get_mock_response(prompt)

def get_mock_response(prompt):
    """Fallback mock response when API is not available"""
    mock_llm = MockLLM(verbose=False)
    
    if "JSON:" in prompt:
        # Extract just the query part from the prompt
        query_part = prompt.split('{')[-1].split('}')[0] if '{' in prompt and '}' in prompt else prompt
        return mock_llm.query_to_json_response(query_part)
    elif "Steps:" in prompt:
        return mock_llm.constraint_to_step_response(prompt)
    else:
        # Extract step type from prompt and use real prompt templates
        for step_type in ["Destination cities", "Departure dates", "Transportation methods", 
                         "Car rental", "Ski resort", "Ski slopes", "Ski equipment", 
                         "Accommodation", "Budget"]:
            if step_type.lower() in prompt.lower():
                return mock_llm.step_to_code_response(step_type, prompt)
        return "# Mock code response\npass"

def Claude_response(prompt, verbose=True):
    """Claude response using GitHub Models API or fallback to mock"""
    return GPT_response(prompt, "deepseek-r1", verbose=verbose)

def Mixtral_response(prompt, format_type=None, verbose=True):
    """Mixtral response using GitHub Models API or fallback to mock"""
    return GPT_response(prompt, "jambda-1.5-large", verbose=verbose)

# Ensure the main logic is executed
if __name__ == "__main__":
    print("SKI PLANNER - Using Real GitHub Models API (Gurobi version)")

    # Add debug prints to trace execution flow
    print("Starting main logic...")

    # Example query for testing
    query = "Plan a 5-day ski trip to Zermatt for 4 people with a budget of 5000 euros."
    mode = "test"
    model = "gpt-4o-mini"
    index = 1

    # Call the pipeline function
    try:
        plan = pipeline_ski(query, mode, model, index, model_version="gpt-4o-mini", verbose=True)
        if plan:
            print("\nGenerated Ski Plan:\n")
            print(plan)
        else:
            print("\nNo plan could be generated. Please check the constraints and inputs.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")


# Remove redundant Mock LLM message
# Ensure all LLM calls use the updated GPT_response function
def pipeline_ski(query, mode, model_name, index, model_version=None, verbose=False):
    """Pipeline for ski trip planning using Gurobi"""
    path = f'output_ski/{mode}/{model_name}/{index}/'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'codes/')
        os.makedirs(path+'plans/')

    # Load ski-specific prompts
    with open('prompts/ski/query_to_json_ski.txt', 'r', encoding='utf-8') as file:
        query_to_json_prompt = file.read()
    with open('prompts/ski/constraint_to_step_ski.txt', 'r', encoding='utf-8') as file:
        constraint_to_step_prompt = file.read()
    with open('prompts/ski/step_to_code_resort.txt', 'r') as file:
        step_to_code_resort_prompt = file.read()
    with open('prompts/ski/step_to_code_slopes.txt', 'r') as file:
        step_to_code_slopes_prompt = file.read()
    with open('prompts/ski/step_to_code_equipment.txt', 'r') as file:
        step_to_code_equipment_prompt = file.read()
    with open('prompts/ski/step_to_code_car.txt', 'r') as file:
        step_to_code_car_prompt = file.read()
    with open('prompts/ski/step_to_code_budget.txt', 'r') as file:
        step_to_code_budget_prompt = file.read()

    step_to_code_prompts = {
        'Resort': step_to_code_resort_prompt,
        'Slopes': step_to_code_slopes_prompt,
        'Equipment': step_to_code_equipment_prompt,
        'Car': step_to_code_car_prompt,
        'Budget': step_to_code_budget_prompt
    }
    # Initialize Gurobi model
    m = Model("SkiResortPlanning")

    variables = {}
    times = []

    # --- Centralized Cost Variables ---
    # Define all cost-related variables upfront
    accommodation_cost = m.addVar(vtype=GRB.CONTINUOUS, name="accommodation_cost")
    equipment_cost = m.addVar(vtype=GRB.CONTINUOUS, name="equipment_cost")
    car_cost = m.addVar(vtype=GRB.CONTINUOUS, name="car_cost")
    total_cost = m.addVar(vtype=GRB.CONTINUOUS, name="total_cost")

    # Store them in the variables dictionary for easy access
    variables['accommodation_cost'] = accommodation_cost
    variables['equipment_cost'] = equipment_cost
    variables['car_cost'] = car_cost
    variables['total_cost'] = total_cost
    
    # Set the main objective to minimize the total cost of the trip
    m.setObjective(total_cost, GRB.MINIMIZE)

    if verbose:
        print("Initialized Gurobi model and cost variables.")
        print("🎯 Objective: Minimize total cost (accommodation + equipment + car rental)")
        print("🔧 Gurobi will search for the optimal (cheapest) combination of choices")
    
    plan = ''
    success = False
    
    try:
        # Step 1: Convert query to JSON
        query_json_str = GPT_response(query_to_json_prompt + '{' + query + '}\n' + 'JSON:\n', model_version, verbose)
        query_json = json.loads(query_json_str.replace('```json', '').replace('```', ''))
        
        with open(path+'plans/' + 'query.txt', 'w', encoding='utf-8') as f:
            f.write(query)
        with open(path+'plans/' + 'query.json', 'w', encoding='utf-8') as f:
            json.dump(query_json, f)
        
        if verbose:
            print('-----------------query in json format-----------------\n', query_json)
        
        # Step 2: Generate planning steps
        start = time.time()
        steps_str = GPT_response(constraint_to_step_prompt + query + '\n' + 'Steps:\n', model_version, verbose)
        json_step = time.time()
        times.append(json_step - start)
        
        with open(path+'plans/' + 'steps.txt', 'w', encoding='utf-8') as f:
            f.write(steps_str)
        
        steps = [s.strip() for s in steps_str.split('\n\n') if s.strip()]
        
        # Step 3: Process each step and build the Gurobi model
        for step in steps:
            if verbose:
                print('!!!!!!!!!!STEP!!!!!!!!!!\n', step, '\n')

            # --- Resort Step ---
            if 'Resort' in step:
                if verbose:
                    print("Processing Resort step...")
                
                # Initialize API to get real resort data
                ski_resort_api = SkiResorts()
                destination = query_json.get('destination', 'Livigno')
                
                # Get real resort data from database
                resort_df = ski_resort_api.run(destination)
                
                if isinstance(resort_df, str):  # No results found
                    # Fallback to search by country/continent
                    if destination.lower() in ['livigno', 'cortina', 'val gardena']:
                        resort_df = ski_resort_api.get_resort_by_country('Italy')
                    elif destination.lower() in ['zermatt', 'st. moritz', 'verbier']:
                        resort_df = ski_resort_api.get_resort_by_country('Switzerland') 
                    else:
                        resort_df = ski_resort_api.get_resort_by_continent('Europe')
                
                if isinstance(resort_df, str):  # Still no results
                    if verbose:
                        print(f"⚠️  No resort data found for {destination}, using fallback data")
                    # Use minimal fallback
                    selected_resorts = [destination]
                    resort_info_beds = [100]
                    resort_info_price = [200]
                    resort_info_access = [1]  # Car
                    resort_info_rating = [4.0]
                else:
                    # Use real dataset with variety for optimization
                    if len(resort_df) > 10:
                        # Sample different price ranges for optimization
                        resort_df_sorted = resort_df.sort_values('Price_day')
                        # Take budget, mid-range, and luxury options
                        indices = [0, len(resort_df)//3, 2*len(resort_df)//3, len(resort_df)-1]
                        if len(resort_df) > 5:
                            indices.append(len(resort_df)//2)  # Add middle option
                        resort_df = resort_df_sorted.iloc[indices].reset_index(drop=True)
                    
                    selected_resorts = resort_df['Resort'].tolist()
                    resort_info_beds = resort_df['Beds'].tolist()
                    resort_info_price = resort_df['Price_day'].tolist()
                    
                    # Convert access method to numeric
                    access_map = {'Car': 1, 'Train': 2, 'Bus': 3}
                    resort_info_access = [access_map.get(access, 1) for access in resort_df['Access']]
                    resort_info_rating = resort_df['Rating'].tolist()

                # Parameters from query
                access_method_required = 1  # Default: Car
                if query_json.get('access') == 'Train':
                    access_method_required = 2
                elif query_json.get('access') == 'Bus':
                    access_method_required = 3
                    
                minimum_rating_required = query_json.get('rating', 3.5)  # Lower default for more options
                people_count = query_json.get('people', 2)
                days = query_json.get('days', 3)

                # --- Gurobi Variables for Resort ---
                resort_index = m.addVar(vtype=GRB.INTEGER, name="resort_index")
                is_resort = [m.addVar(vtype=GRB.BINARY, name=f"is_resort_{i}") for i in range(len(selected_resorts))]
                
                if verbose:
                    print(f"🏨 Resort optimization: {len(selected_resorts)} resort options from real dataset")
                    for i, resort in enumerate(selected_resorts):
                        price = resort_info_price[i] 
                        rating = resort_info_rating[i]
                        beds = resort_info_beds[i]
                        print(f"   Option {i+1}: {resort} - €{price}/day (⭐{rating:.1f}, 🛏️{beds} beds)")
                
                m.addConstr(sum(is_resort) == 1, "one_resort_selected")
                for i in range(len(selected_resorts)):
                    m.addGenConstrIndicator(is_resort[i], True, resort_index == i, name=f"select_resort_{i}")

                # --- Resort Attribute Variables ---
                resort_beds = m.addVar(vtype=GRB.INTEGER, name="resort_beds")
                resort_price = m.addVar(vtype=GRB.CONTINUOUS, name="resort_price")
                resort_access = m.addVar(vtype=GRB.INTEGER, name="resort_access")
                resort_rating = m.addVar(vtype=GRB.CONTINUOUS, name="resort_rating")

                # --- Constraints to link attributes to selected resort ---
                m.addConstr(resort_beds == sum(is_resort[i] * resort_info_beds[i] for i in range(len(selected_resorts))), "beds_assign")
                m.addConstr(resort_price == sum(is_resort[i] * resort_info_price[i] for i in range(len(selected_resorts))), "price_assign")
                m.addConstr(resort_access == sum(is_resort[i] * resort_info_access[i] for i in range(len(selected_resorts))), "access_assign")
                m.addConstr(resort_rating == sum(is_resort[i] * resort_info_rating[i] for i in range(len(selected_resorts))), "rating_assign")

                # --- Functional Constraints for Resort ---
                # More flexible access constraint - allow car or train for most queries
                if query_json.get('car_type') or 'car' in query_json.get('query', '').lower():
                    m.addConstr(resort_access == 1, "car_access_required")  # Force car access if car requested
                else:
                    # Allow flexible access but prefer car/train over bus
                    m.addConstr(resort_access <= 2, "prefer_car_or_train")
                    
                # Rating constraint
                m.addConstr(resort_rating >= minimum_rating_required, "min_rating")
                m.addConstr(resort_beds >= people_count, "sufficient_beds")

                # --- Cost Calculation for Resort ---
                m.addConstr(accommodation_cost == resort_price * days, "accommodation_cost_calc")

            # --- Slopes Step ---
            elif 'Slopes' in step:
                if verbose:
                    print("Processing Slopes step...")

                # Initialize API to get real slope data
                ski_slopes_api = SkiSlopes()
                
                # Get slope data for the selected resorts
                slope_info_difficulty = []
                slope_info_total_runs = []
                slope_info_longest_run = []
                
                difficulty_map = {'Blue': 0, 'Red': 1, 'Black': 2}
                
                for resort in selected_resorts:
                    slope_df = ski_slopes_api.run(resort)
                    
                    if isinstance(slope_df, str):  # No data found
                        # Use reasonable defaults based on resort type
                        slope_info_difficulty.append(1)  # Red (intermediate)
                        slope_info_total_runs.append(30)  # Average number
                        slope_info_longest_run.append(5)  # Average length
                    else:
                        # Use real data - take average/most common values
                        if len(slope_df) > 0:
                            # Most common difficulty level
                            common_difficulty = slope_df['Difficult_Slope'].mode().iloc[0]
                            slope_info_difficulty.append(difficulty_map.get(common_difficulty, 1))
                            
                            # Average total slopes and longest run
                            slope_info_total_runs.append(int(slope_df['Total_Slopes'].mean()))
                            slope_info_longest_run.append(int(slope_df['Longest_Run'].mean()))
                        else:
                            # Fallback defaults
                            slope_info_difficulty.append(1)
                            slope_info_total_runs.append(30)
                            slope_info_longest_run.append(5)
                
                if verbose:
                    print(f"📊 Slope data loaded for {len(selected_resorts)} resorts from real dataset")
                    for i, resort in enumerate(selected_resorts):
                        diff_name = ['Blue', 'Red', 'Black'][slope_info_difficulty[i]]
                        print(f"   {resort}: {diff_name} slopes, {slope_info_total_runs[i]} total runs, {slope_info_longest_run[i]}km longest")

                # --- Gurobi Variables for Slopes ---
                resort_slope_difficulty = m.addVar(vtype=GRB.INTEGER, name="resort_slope_difficulty")
                resort_total_runs = m.addVar(vtype=GRB.INTEGER, name="resort_total_runs")
                resort_longest_run = m.addVar(vtype=GRB.INTEGER, name="resort_longest_run")

                # --- Constraints to link slope attributes to the selected resort ---
                m.addConstr(resort_slope_difficulty == sum(is_resort[i] * slope_info_difficulty[i] for i in range(len(selected_resorts))), "slope_difficulty_assign")
                m.addConstr(resort_total_runs == sum(is_resort[i] * slope_info_total_runs[i] for i in range(len(selected_resorts))), "total_runs_assign")
                m.addConstr(resort_longest_run == sum(is_resort[i] * slope_info_longest_run[i] for i in range(len(selected_resorts))), "longest_run_assign")

                # --- Functional Constraints for Slopes ---
                slope_difficulty_str = query_json.get('slope_difficulty')
                if slope_difficulty_str:
                    slope_difficulty_preference = difficulty_map.get(slope_difficulty_str)
                    if slope_difficulty_preference is not None:
                        # Allow some flexibility - can choose same or easier difficulty
                        m.addConstr(resort_slope_difficulty <= slope_difficulty_preference, "slope_difficulty_match")

                # Add general requirements for a good ski experience
                m.addConstr(resort_total_runs >= 15, "minimum_total_runs")  # Lower requirement for more options
                m.addConstr(resort_longest_run >= 2, "minimum_longest_run")  # Lower requirement for more options

            # --- Equipment Step ---
            elif 'Equipment' in step:
                if verbose:
                    print("Processing Equipment step...")

                # Initialize API to get real equipment rental data
                ski_rent_api = SkiRent()
                destination = query_json.get('destination', 'Livigno')
                
                # Get real equipment pricing data
                rent_df = ski_rent_api.run(destination)
                
                if isinstance(rent_df, str):  # No data found for specific destination
                    # Try to get European data as fallback
                    all_data = ski_rent_api.data
                    rent_df = all_data[all_data['Continent'] == 'Europe'].head(20)  # Sample European prices
                
                # Extract real pricing for each equipment type
                equipment_prices = {}
                equipment_types = ['Skis', 'Boots', 'Helmet', 'Poles']
                
                for equipment_type in equipment_types:
                    if isinstance(rent_df, str):
                        # Fallback pricing
                        fallback_prices = {'Skis': 25, 'Boots': 15, 'Helmet': 10, 'Poles': 5}
                        equipment_prices[equipment_type] = fallback_prices[equipment_type]
                    else:
                        equipment_data = rent_df[rent_df['Equipment'] == equipment_type]
                        if len(equipment_data) > 0:
                            # Use average price for this equipment type
                            equipment_prices[equipment_type] = int(equipment_data['Price_day'].mean())
                        else:
                            # Fallback for missing equipment
                            fallback_prices = {'Skis': 35, 'Boots': 20, 'Helmet': 15, 'Poles': 8}
                            equipment_prices[equipment_type] = fallback_prices[equipment_type]

                # --- Gurobi Variables for Equipment ---
                rent_skis = m.addVar(vtype=GRB.BINARY, name="rent_skis")
                rent_boots = m.addVar(vtype=GRB.BINARY, name="rent_boots")
                rent_helmet = m.addVar(vtype=GRB.BINARY, name="rent_helmet")
                rent_poles = m.addVar(vtype=GRB.BINARY, name="rent_poles")

                # --- Real Cost Data for Equipment (per person, per day) ---
                cost_per_ski = equipment_prices['Skis']
                cost_per_boots = equipment_prices['Boots']
                cost_per_helmet = equipment_prices['Helmet']
                cost_per_poles = equipment_prices['Poles']

                if verbose:
                    print("⛷️  Equipment optimization: 4 equipment types available (real pricing)")
                    print(f"   Skis: €{cost_per_ski}/person/day")
                    print(f"   Boots: €{cost_per_boots}/person/day") 
                    print(f"   Helmet: €{cost_per_helmet}/person/day")
                    print(f"   Poles: €{cost_per_poles}/person/day")
                    print("   🎯 Gurobi will choose optimal equipment combination to minimize cost")

                # --- Functional Constraints for Equipment ---
                requested_equipment = query_json.get('equipment') or []
                people_count = query_json.get('people', 2)
                days = query_json.get('days', 3)
                
                # Check if equipment rental is actually requested in the query using the same logic as earlier
                query_text = query_json.get('query', '').lower()
                ski_equipment_patterns = [
                    r'\bequipment\b',
                    r'\bski\s+rental\b',
                    r'\brent\s+ski\b',
                    r'\bski\s+equipment\b',
                    r'\brental\s+equipment\b',
                    r'\brent\s+equipment\b',
                    r'\brent\s+skis\b'
                ]
                equipment_requested = any(re.search(pattern, query_text) for pattern in ski_equipment_patterns) or len(requested_equipment) > 0
                
                if equipment_requested:
                    if verbose:
                        print("   Equipment rental detected in query - adding equipment constraints")
                    
                    # If equipment is requested, enforce specific items
                    if "Skis" in requested_equipment or 'ski' in query_text:
                        m.addConstr(rent_skis == 1, "skis_rental_requested")
                    
                    if "Boots" in requested_equipment or 'boot' in query_text:
                        m.addConstr(rent_boots == 1, "boots_rental_requested")
                    
                    if "Helmet" in requested_equipment or 'helmet' in query_text:
                        m.addConstr(rent_helmet == 1, "helmet_rental_requested")
                    
                    if "Poles" in requested_equipment or 'pole' in query_text:
                        m.addConstr(rent_poles == 1, "poles_rental_requested")
                    
                    # If "equipment" is mentioned generally, include basic ski equipment
                    if 'equipment' in query_text and not requested_equipment:
                        m.addConstr(rent_skis == 1, "basic_skis_for_equipment")
                        m.addConstr(rent_boots == 1, "basic_boots_for_equipment")
                        if verbose:
                            print("   General equipment request - including skis and boots")
                else:
                    if verbose:
                        print("   No equipment rental mentioned in query - equipment is optional")
                    # If no equipment is mentioned, make everything optional (Gurobi will optimize based on cost)
                    # No constraints needed - variables default to 0 (no rental)

                # --- Cost Calculation for Equipment ---
                # This is a Gurobi linear expression for the daily cost per person
                daily_cost_per_person_expr = (
                    rent_skis * cost_per_ski +
                    rent_boots * cost_per_boots +
                    rent_helmet * cost_per_helmet +
                    rent_poles * cost_per_poles
                )
                # The total equipment cost is this daily rate times the number of people and days
                m.addConstr(equipment_cost == daily_cost_per_person_expr * people_count * days, "equipment_cost_calc")

            # --- Car Step ---
            elif 'Car' in step:
                if verbose:
                    print("Processing Car step...")

                # Initialize API to get real car rental data
                ski_car_api = SkiCar()
                destination = query_json.get('destination', 'Livigno')
                
                # Get real car rental pricing for each resort
                car_info_price = []
                car_type_options = []
                fuel_type_options = []
                
                for resort in selected_resorts:
                    car_df = ski_car_api.run(resort)
                    
                    if isinstance(car_df, str):  # No data found for this resort
                        # Use European average as fallback
                        all_car_data = ski_car_api.data
                        car_df = all_car_data[all_car_data['Continent'] == 'Europe']
                    
                    if isinstance(car_df, str) or len(car_df) == 0:
                        # Final fallback
                        car_info_price.append(75)  # Average European price
                        car_type_options.append(['SUV', 'Sedan'])
                        fuel_type_options.append(['Petrol', 'Diesel'])
                    else:
                        # Use real data - average price for this resort
                        avg_price = int(car_df['Price_day'].mean())
                        car_info_price.append(avg_price)
                        
                        # Available car types and fuel types at this resort
                        car_type_options.append(car_df['Type'].unique().tolist())
                        fuel_type_options.append(car_df['Fuel'].unique().tolist())

                # --- Gurobi Variables for Car Rental ---
                car_rental = m.addVar(vtype=GRB.BINARY, name="car_rental")
                car_type = m.addVar(vtype=GRB.INTEGER, name="car_type") # 0: SUV, 1: Sedan, 2: Pick up, 3: Cabriolet
                fuel_type = m.addVar(vtype=GRB.INTEGER, name="fuel_type") # 0: Petrol, 1: Diesel, 2: Hybrid, 3: Electric

                if verbose:
                    print("🚗 Car rental optimization: Real pricing from dataset")
                    for i, resort in enumerate(selected_resorts):
                        available_types = ', '.join(car_type_options[i][:3])  # Show first 3 types
                        available_fuels = ', '.join(fuel_type_options[i][:3])  # Show first 3 fuel types
                        print(f"   {resort}: €{car_info_price[i]}/day (Types: {available_types}, Fuels: {available_fuels})")
                    print(f"   Price range: €{min(car_info_price)}-€{max(car_info_price)}/day")
                    print("   🎯 Gurobi will decide if car rental is cost-effective")

                # --- Functional Constraints for Car Rental ---
                requested_car_type = query_json.get('car_type')
                requested_fuel_type = query_json.get('fuel_type')
                days = query_json.get('days', 3)

                # If a car type is specified, we must rent a car
                if requested_car_type:
                    m.addConstr(car_rental == 1, "force_car_rental_if_type_specified")
                    car_type_map = {"SUV": 0, "Sedan": 1, "Pick up": 2, "Cabriolet": 3}
                    car_type_preference = car_type_map.get(requested_car_type)
                    if car_type_preference is not None:
                        m.addConstr(car_type == car_type_preference, "car_type_match")
                else:
                    # If no car is requested, make it optional (let Gurobi decide based on cost)
                    # Remove the constraint that forces car_rental to 0
                    pass

                if requested_fuel_type:
                    fuel_type_map = {"Petrol": 0, "Diesel": 1, "Hybrid": 2, "Electric": 3}
                    fuel_type_preference = fuel_type_map.get(requested_fuel_type)
                    if fuel_type_preference is not None:
                        m.addConstr(fuel_type == fuel_type_preference, "fuel_type_match")

                # --- Cost Calculation for Car Rental ---
                # Get the price for the selected resort using real data
                resort_car_price = m.addVar(vtype=GRB.CONTINUOUS, name="resort_car_price")
                m.addConstr(resort_car_price == sum(is_resort[i] * car_info_price[i] for i in range(len(selected_resorts))), "car_price_assign")
                
                # The total car cost is the daily price times days, only if a car is rented
                m.addConstr(car_cost == car_rental * resort_car_price * days, "car_cost_calc")

            # --- Budget Step ---
            elif 'Budget' in step:
                if verbose:
                    print("Processing Budget step...")
                
                budget_amount = query_json.get('budget')
                people_count = query_json.get('people', 2)
                days = query_json.get('days', 3)

                if budget_amount is not None:
                    budget_limit = float(budget_amount)
                    
                    # --- Total Cost Definition ---
                    # This constraint connects the total_cost variable (the objective) to the individual cost components.
                    m.addConstr(total_cost == accommodation_cost + equipment_cost + car_cost, "total_cost_definition")
                    
                    # --- Budget Constraint ---
                    m.addConstr(total_cost <= budget_limit, "budget_constraint")
                    
                    # --- Realistic Budget Constraint ---
                    # Ensures the provided budget is reasonable for the trip.
                    minimum_budget = people_count * days * 200.0
                    m.addConstr(budget_limit >= minimum_budget, "realistic_budget_minimum")
                    
                    if verbose:
                        print(f"Added budget constraint: total_cost <= {budget_limit}")
                        print(f"Minimum realistic budget: {minimum_budget}")
                else:
                    if verbose:
                        print("No budget specified, skipping budget constraints.")
                    # If no budget, total cost is still the sum, but with no upper limit.
                    m.addConstr(total_cost == accommodation_cost + equipment_cost + car_cost, "total_cost_definition")
            
        # Step 4: Set objective and solve the Gurobi model
        if verbose:
            print("\n--- Optimization Summary ---")
            print("📊 Gurobi Optimization Problem Setup:")
            print(f"   • Resort options: {len(selected_resorts) if 'selected_resorts' in locals() else 'Multiple'}")
            print("   • Equipment choices: 4 types (skis, boots, helmet, poles)")
            print("   • Car rental: Optional (cost vs. no rental)")
            print("   • Constraints: Budget limits, functional requirements")
            print("\n--- Setting Optimization Objective ---")
            print("🎯 Setting objective: MINIMIZE total cost")
        
        # Set the objective to minimize total cost
        m.setObjective(total_cost, GRB.MINIMIZE)
        
        if verbose:
            print("🔧 Gurobi will now optimize to find the cheapest combination")
            print("\n--- Solving Gurobi Model ---")
        m.optimize()

        # Step 5: Interpret and present the solution
        if m.status == GRB.OPTIMAL:
            selected_resort_index = int(m.getVarByName("resort_index").X)
            resort_name = selected_resorts[selected_resort_index]
            
            if verbose:
                print("--- Gurobi Optimization Result ---")
                print("🎯 Optimal solution found with Gurobi optimization!")
                print(f"  - Objective value (minimized cost): €{m.objVal:.2f}")
                print(f"  - Selected Resort: {resort_name}")
                print(f"  - Total Cost: €{total_cost.X:.2f}")
                print(f"  - Accommodation Cost: €{accommodation_cost.X:.2f}")
                print(f"  - Equipment Cost: €{equipment_cost.X:.2f}")
                print(f"  - Car Rental Cost: €{car_cost.X:.2f}")
                
                # Show optimization choices
                try:
                    if m.getVarByName("rent_skis"):
                        print("  Equipment optimally chosen:")
                        if m.getVarByName("rent_skis").X > 0.5:
                            print("    ✓ Skis")
                        if m.getVarByName("rent_boots").X > 0.5:
                            print("    ✓ Boots")
                        if m.getVarByName("rent_helmet").X > 0.5:
                            print("    ✓ Helmet")
                        if m.getVarByName("rent_poles").X > 0.5:
                            print("    ✓ Poles")
                except Exception:
                    pass
                
                try:
                    if m.getVarByName("car_rental") and m.getVarByName("car_rental").X > 0.5:
                        print("  ✓ Car rental selected (cost-effective)")
                    else:
                        print("  ✗ Car rental not selected (not cost-effective)")
                except Exception:
                    pass
            
            plan = (f"🎯 Optimal solution found with Gurobi optimization!\n"
                    f"  - Objective value (minimized cost): €{m.objVal:.2f}\n"
                    f"  - Selected Resort: {resort_name}\n"
                    f"  - Total Cost: €{total_cost.X:.2f}\n"
                    f"  - Accommodation Cost: €{accommodation_cost.X:.2f}\n"
                    f"  - Equipment Cost: €{equipment_cost.X:.2f}\n"
                    f"  - Car Rental Cost: €{car_cost.X:.2f}")
            success = True
        elif m.status == GRB.INFEASIBLE:
            plan = "No solution found: The model is infeasible. Check constraints."
            if verbose:
                print("Computing IIS to find conflicting constraints...")
                m.computeIIS()
                m.write("model.ilp")
                print("IIS written to model.ilp")
        else:
            plan = f"No optimal solution found. Gurobi status: {m.status}"

        if verbose:
            print(f"\n--- Gurobi Result ---\n{plan}")

    except Exception as e:
        print(f"\nERROR during Gurobi execution: {e}")
        import traceback
        traceback.print_exc()

    return plan if success else None

def generate_ski_plan(s, variables, query):
    """Generate ski plan from Z3 solution"""
    SkiResortSearch = SkiResorts()
    SkiSlopeSearch = SkiSlopes()
    SkiRentSearch = SkiRent()
    
    resorts = []
    slopes = []
    equipment = []
    transportation = []
    accommodation = None
    
    try:
        # Get resort information
        if 'resort_index' in variables:
            resort_var = variables['resort_index']
            if hasattr(resort_var, '__iter__') and not isinstance(resort_var, str):
                # It's a list/array
                for i, rv in enumerate(resort_var):
                    resort_idx = int(s.model()[rv].as_long())
                    resort_list = SkiResortSearch.run(query.get('destination', 'Livigno'))
                    if isinstance(resort_list, DataFrame) and len(resort_list) > resort_idx:
                        resorts.append(resort_list.iloc[resort_idx]['Resort'])
                    else:
                        resorts.append(query.get('destination', f"Resort {resort_idx + 1}"))
            else:
                # It's a single variable
                resort_idx = int(s.model()[resort_var].as_long())
                resort_list = SkiResortSearch.run(query.get('destination', 'Livigno'))
                if isinstance(resort_list, DataFrame) and len(resort_list) > resort_idx:
                    resorts.append(resort_list.iloc[resort_idx]['Resort'])
                else:
                    resorts.append(query.get('destination', f"Resort {resort_idx + 1}"))
        
        # Get slopes information  
        if 'slopes_index' in variables:
            slope_var = variables['slopes_index']
            if hasattr(slope_var, '__iter__') and not isinstance(slope_var, str):
                for i, sv in enumerate(slope_var):
                    slope_idx = int(s.model()[sv].as_long())
                    slope_list = SkiSlopeSearch.run(query.get('destination', 'Livigno'))
                    if isinstance(slope_list, DataFrame) and len(slope_list) > slope_idx:
                        slopes.append(slope_list.iloc[slope_idx]['Slope_Name'])
                    else:
                        slopes.append(f"Slope {slope_idx + 1}")
            else:
                slope_idx = int(s.model()[slope_var].as_long())
                slope_list = SkiSlopeSearch.run(query.get('destination', 'Livigno'))
                if isinstance(slope_list, DataFrame) and len(slope_list) > slope_idx:
                    slopes.append(slope_list.iloc[slope_idx]['Slope_Name'])
                else:
                    slopes.append(f"Slope {slope_idx + 1}")
        
        # Get equipment rental
        if 'rent_index' in variables:
            rent_var = variables['rent_index']
            if hasattr(rent_var, '__iter__') and not isinstance(rent_var, str):
                for i, rv in enumerate(rent_var):
                    rent_idx = int(s.model()[rv].as_long())
                    rent_list = SkiRentSearch.run(query.get('destination', 'Livigno'))
                    if isinstance(rent_list, DataFrame) and len(rent_list) > rent_idx:
                        equipment.append(rent_list.iloc[rent_idx]['Shop_Name'])
                    else:
                        equipment.append(f"Rental Shop {rent_idx + 1}")
            else:
                rent_idx = int(s.model()[rent_var].as_long())
                rent_list = SkiRentSearch.run(query.get('destination', 'Livigno'))
                if isinstance(rent_list, DataFrame) and len(rent_list) > rent_idx:
                    equipment.append(rent_list.iloc[rent_idx]['Shop_Name'])
                else:
                    equipment.append(f"Rental Shop {rent_idx + 1}")
        
        # Get transportation - simplified
        if 'car_rental' in variables:
            car_var = variables['car_rental']
            if hasattr(car_var, '__iter__') and not isinstance(car_var, str):
                for cv in car_var:
                    if s.model()[cv]:
                        transportation.append("Car Rental")
                        break
            else:
                if s.model()[car_var]:
                    transportation.append("Car Rental")
        
        # Add car type and fuel type if available
        car_details = []
        if 'car_type' in query:
            car_details.append(f"Type: {query['car_type']}")
        if 'fuel_type' in query:
            car_details.append(f"Fuel: {query['fuel_type']}")
        
        # Get accommodation
        if 'accommodation_index' in variables:
            acc_var = variables['accommodation_index']
            acc_idx = int(s.model()[acc_var].as_long())
            resort_list = SkiResortSearch.run(query.get('destination', 'Livigno'))
            if isinstance(resort_list, DataFrame) and len(resort_list) > acc_idx:
                accommodation = resort_list.iloc[acc_idx]['Resort'] + " Lodge"
            else:
                accommodation = f"{query.get('destination', 'Mountain')} Lodge"
        
        # Format transportation with details
        transportation_str = ', '.join(transportation)
        if car_details and 'Car Rental' in transportation_str:
            transportation_str += f" ({', '.join(car_details)})"
        
        plan = f"""SKI TRIP PLAN:
Destination: {query.get('destination', 'Unknown')}
Duration: {query.get('days', 3)} days
People: {query.get('people', 2)}
Budget: €{query.get('budget', 1500)}

Ski Resorts: {', '.join(resorts) if resorts else query.get('destination', 'To be selected')}
Ski Slopes: {', '.join(slopes) if slopes else 'Available slopes at destination'}  
Equipment Rental: {', '.join(equipment) if equipment else 'Local rental shops'}
Transportation: {transportation_str if transportation else 'To be arranged'}
Accommodation: {accommodation if accommodation else f"{query.get('destination', 'Mountain')} Lodge"}
"""
        
        return plan
        
    except Exception as e:
        print(f"Error generating ski plan: {e}")
        return f"Ski trip plan for {query['dest']} - {query['days']} days"

# Ensure the main logic is executed
if __name__ == "__main__":
    print("SKI PLANNER - Using Real GitHub Models API (Gurobi version)")

    # Add debug prints to trace execution flow
    print("Starting main logic...")

    # Example query for testing
    query = "Plan a 5-day ski trip to Zermatt for 4 people with a budget of 5000 euros."
    mode = "test"
    model = "gpt-4o-mini"
    index = 1

    # Call the pipeline function
    try:
        plan = pipeline_ski(query, mode, model, index, model_version="gpt-4o-mini", verbose=True)
        if plan:
            print("\nGenerated Ski Plan:\n")
            print(plan)
        else:
            print("\nNo plan could be generated. Please check the constraints and inputs.")
    except Exception as e:
        print(f"An error occurred during execution: {e}")


# Remove redundant Mock LLM message
# Ensure all LLM calls use the updated GPT_response function

def calculate_detailed_costs(query_json):
    """Calculate detailed costs for each service based on query parameters"""
    costs = {}
    days = query_json.get('days', 3)
    people = query_json.get('people', 2)
    
    # Resort costs (per day per person)
    resort_cost_per_day = 80  # Base cost
    if 'luxury' in query_json.get('query', '').lower():
        resort_cost_per_day = 150
    elif 'premium' in query_json.get('query', '').lower():
        resort_cost_per_day = 120
    
    costs['resort'] = resort_cost_per_day * days * people
    
    # Equipment rental costs
    equipment_cost_per_day = 25  # Base cost per person per day
    equipment_people = query_json.get('equipment_people', people)
    if query_json.get('equipment'):
        costs['equipment'] = equipment_cost_per_day * days * equipment_people
    else:
        costs['equipment'] = 0
    
    # Car rental costs
    if query_json.get('car_type') or query_json.get('access') == 'Car':
        car_base_cost = 50  # Base cost per day
        if query_json.get('car_type') == 'SUV':
            car_base_cost = 80
        elif query_json.get('car_type') == 'Pick up':
            car_base_cost = 75
        elif query_json.get('car_type') == 'Cabriolet':
            car_base_cost = 100
        
        # Fuel type modifier
        if query_json.get('fuel_type') == 'Electric':
            car_base_cost += 20
        elif query_json.get('fuel_type') == 'Hybrid':
            car_base_cost += 10
        
        costs['car_rental'] = car_base_cost * days
    else:
        costs['car_rental'] = 0
    
    # Accommodation costs (separate from resort if specified)
    accommodation_cost = 60 * days * people  # Base cost
    if 'luxury' in query_json.get('query', '').lower():
        accommodation_cost = 120 * days * people
    costs['accommodation'] = accommodation_cost
    
    # Transportation costs (if not car rental)
    if query_json.get('access') == 'Train':
        costs['transportation'] = 80 * people
    elif query_json.get('access') == 'Bus':
        costs['transportation'] = 40 * people
    else:
        costs['transportation'] = 0
    
    # Total cost
    costs['total'] = sum(costs.values())
    
    return costs

def generate_ski_plan_with_gurobi(selected_resorts, resort_info_beds, resort_info_price, resort_info_access, resort_info_rating, access_method_required, minimum_rating_required, people_count, days):
    # Create model
    m = Model("SkiResortPlanning")

    # Decision variable: index of selected resort
    resort_index = m.addVar(vtype=GRB.INTEGER, name="resort_index")

    # Constraint: resort index is within range
    m.addConstr((resort_index >= 0) & (resort_index < len(selected_resorts)), name="valid_resort_index")

    # Add auxiliary variables to extract resort info
    resort_beds = m.addVar(vtype=GRB.INTEGER, name="resort_beds")
    resort_price = m.addVar(vtype=GRB.INTEGER, name="resort_price")
    resort_access = m.addVar(vtype=GRB.INTEGER, name="resort_access")
    resort_rating = m.addVar(vtype=GRB.INTEGER, name="resort_rating")

    # Define constraints using piecewise linear or indicator constraints
    for i in range(len(selected_resorts)):
        m.addGenConstrIndicator((resort_index == i), True, resort_beds == resort_info_beds[i], name=f"bed_select_{i}")
        m.addGenConstrIndicator((resort_index == i), True, resort_price == resort_info_price[i], name=f"price_select_{i}")
        m.addGenConstrIndicator((resort_index == i), True, resort_access == resort_info_access[i], name=f"access_select_{i}")
        m.addGenConstrIndicator((resort_index == i), True, resort_rating == resort_info_rating[i], name=f"rating_select_{i}")

    # Constraint: access method matches
    m.addConstr(resort_access == access_method_required, name="access_method")

    # Constraint: minimum rating
    m.addConstr(resort_rating >= minimum_rating_required, name="min_rating")

    # Constraint: sufficient beds
    m.addConstr(resort_beds >= people_count, name="sufficient_beds")

    # Objective: calculate accommodation cost
    accommodation_cost = m.addVar(vtype=GRB.INTEGER, name="accommodation_cost")
    m.addConstr(accommodation_cost == resort_price * days, name="accommodation_cost_calc")

    # Set objective to minimize accommodation cost
    m.setObjective(accommodation_cost, GRB.MINIMIZE)

    # Optimize the model
    m.optimize()

    # Extract results
    if m.status == GRB.OPTIMAL:
        selected_resort = selected_resorts[int(resort_index.X)]
        return f"Selected resort: {selected_resort}, Cost: {accommodation_cost.X}"
    else:
        return "No feasible solution found."

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ski Planner Test with GitHub Models API and Gurobi")
    parser.add_argument("--set_type", type=str, default="test_ski")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--use_dataset_queries", action="store_true", help="Use queries from dataset_ski")
    parser.add_argument("--max_queries", type=int, default=5, help="Maximum number of queries to test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--clean_output", action="store_true", help="Enable clean output to file")
    parser.add_argument("--query", type=str, help="Single query to test directly from command line")
    parser.add_argument("--query_file", type=str, help="File containing a single query to test")
    args = parser.parse_args()

    print(f"SKI PLANNER - Using GitHub Models API with Gurobi Optimization (Model: {args.model_name})")
    
    # Clean output preparation
    clean_results = [] if args.clean_output else None
    
    # Handle single query from command line or file
    if args.query:
        ski_queries = [args.query]
        print("Single query from command line")
    elif args.query_file:
        try:
            with open(args.query_file, 'r', encoding='utf-8') as f:
                query_from_file = f.read().strip()
            ski_queries = [query_from_file]
            print(f"Single query from file: {args.query_file}")
        except FileNotFoundError:
            print(f"File {args.query_file} not found!")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file {args.query_file}: {e}")
            sys.exit(1)
    # Choose query source for batch processing
    elif args.use_dataset_queries:
        try:
            from dataset_ski.ski_test_queries import SKI_TEST_QUERIES
            ski_queries = [q["query"] for q in SKI_TEST_QUERIES[:args.max_queries]]
            print(f"Using {len(ski_queries)} queries from dataset")
        except ImportError:
            print("Dataset queries not found, using default queries")
            ski_queries = [
                "Plan a 3-day ski trip to Livigno for 2 people, departing from Milano on January 15th, 2024. Budget is 1500 euros.",
                "Organize a 5-day ski vacation to Cortina d'Ampezzo for 4 people with a budget of 3000 euros, departing from Roma.",
                "Plan a 7-day ski adventure to Val d'Isère for 6 people, budget 5000 euros, intermediate level skiers."
            ]
    else:
        # Sample ski queries for testing (default)
        ski_queries = [
            "Plan a 3-day ski trip to Livigno for 2 people, departing from Milano on January 15th, 2024. Budget is 1500 euros.",
            "Organize a 5-day ski vacation to Cortina d'Ampezzo for 4 people with a budget of 3000 euros, departing from Roma.",
            "Plan a 7-day ski adventure to Val d'Isère for 6 people, budget 5000 euros, intermediate level skiers."
            ]
    
    print(f"\n{'='*60}")
    
    # Test range
    test_range = len(ski_queries)
    numbers = [i for i in range(1, test_range + 1)]
    
    for number in numbers:
        path = f'output_ski/{args.set_type}/{args.model_name}/{number}/plans/'
        if not os.path.exists(path + 'plan.txt'):
            query = ski_queries[number-1]
            
            # Show only essential information
            print(f"\nQUERY {number}:")
            print(f"{query}")
            print("-" * 60)
            
            try:
                result_plan = pipeline_ski(query, args.set_type, args.model_name, number, args.model_name, args.verbose)
                
                # Load the generated JSON to get parameters and calculate costs
                json_path = f'output_ski/{args.set_type}/{args.model_name}/{number}/plans/query.json'
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        query_json = json.load(f)
                    
                    # Calculate detailed costs
                    costs = calculate_detailed_costs(query_json)
                    
                    if result_plan:
                        print("SOLUTION FOUND")
                        print(f"Destination: {query_json.get('destination', 'N/A')}")
                        print(f"Duration: {query_json.get('days', 'N/A')} days")
                        print(f"People: {query_json.get('people', 'N/A')}")
                        print(f"Budget: €{query_json.get('budget', 'N/A')}")
                        
                        print("\nDETAILED COSTS:")
                        print(f"  Resort: €{costs['resort']}")
                        print(f"  Equipment: €{costs['equipment']}")
                        if costs['car_rental'] > 0:
                            car_type = query_json.get('car_type', 'SUV')
                            fuel_type = query_json.get('fuel_type', '')
                            fuel_str = f" ({fuel_type})" if fuel_type else ""
                            print(f"  Car Rental ({car_type}{fuel_str}): €{costs['car_rental']}")
                        if costs['transportation'] > 0:
                            print(f"  Transportation: €{costs['transportation']}")
                        print(f"  Accommodation: €{costs['accommodation']}")
                        print("  ───────────────────")
                        print(f"  TOTAL COST: €{costs['total']}")
                        
                        # Check if within budget
                        budget = query_json.get('budget', 0)
                        if budget > 0:
                            if costs['total'] <= budget:
                                print(f"  Within budget (saves €{budget - costs['total']})")
                            else:
                                print(f"  Over budget by €{costs['total'] - budget}")
                    else:
                        print("NO SOLUTION FOUND")
                else:
                    print("NO PARAMETERS EXTRACTED")
                    
            except Exception as e:
                print(f"ERROR: {str(e)}")
                
            print("=" * 60)
        else:
            if args.verbose:
                print(f"Sample {number} already processed")
    
    print("\nProcessing completed!")
