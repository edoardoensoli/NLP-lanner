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
from utils.func import load_line_json_data
from z3 import *
import gurobipy as gp
from gurobipy import GRB

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import ski-specific tools
try:
    from tools_ski.apis import SkiResorts, SkiSlopes, SkiRent, SkiCar
except ImportError:
    # Fallback to simplified mock classes if tools are not available
    class SkiResorts:
        def __init__(self):
            self.data = pd.DataFrame({'Resort': ['Mock Resort'], 'Price_day': [100]})
        def get_all_resorts_df(self):
            return pd.DataFrame({'name': ['Mock Resort'], 'price': [100]})
        def get_all_accommodations_df(self):
            return pd.DataFrame({'name': ['Mock Hotel'], 'price': [150]})
    
    class SkiSlopes:
        def __init__(self):
            self.data = pd.DataFrame({'Slope': ['Mock Slope'], 'Difficulty': ['easy']})
        def get_all_slopes_df(self):
            return pd.DataFrame({'name': ['Mock Slope'], 'difficulty': ['easy']})
    
    class SkiCar:
        def __init__(self):
            self.data = pd.DataFrame({'Car': ['Mock Car'], 'Price_day': [50]})
        def get_all_cars_df(self):
            return self.data
    
    class SkiRent:
        def __init__(self):
            self.data = pd.DataFrame({'Equipment': ['Mock Equipment'], 'Price_day': [30]})
        def get_all_equipment_df(self):
            return self.data

# GitHub Models API Client with model switching
class LLMClient:
    """Client for interacting with a generic LLM API."""
    def __init__(self, model_name: str = 'DeepSeek-R1', fallback_models: list = None):
        load_dotenv() # Load environment variables from .env file
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
            "429",
            "rate limit",
            "too many requests",
            "quota exceeded",
            "requests per"
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
                    # Fallback to mock response if a non-rate-limit error occurs
                    return get_mock_response(prompt), model
        
        # If all models fail, fall back to mock response
        print("All models failed, falling back to mock response.")
        return get_mock_response(prompt), models_to_try[0] if models_to_try else self.model_name

    def _query_api_single_model(self, prompt, model_name):
        """Query API with a specific model"""
        url = f"{self.base_url}/chat/completions"
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 1
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        
        if response.status_code == 200:
            response_json = response.json()
            return response_json['choices'][0]['message']['content']
        else:
            error_message = f"Error {response.status_code}: {response.text}"
            if response.status_code == 429:
                raise Exception(f"Rate limit exceeded: {error_message}")
            else:
                raise Exception(error_message)

    def _query_api(self, prompt):
        """Legacy method for backward compatibility"""
        response, _ = self._query_api_with_fallback(prompt)
        return response

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
- special_requirements: list of special requirements. Look for phrases like:
  * "need car rental", "car rental", "rent a car", "rental car" → add "car rental"
  * "need equipment", "equipment rental", "rent equipment", "ski equipment" → add "equipment"
  * "multiple resorts", "visit multiple" → add "multiple_resorts"

Return only valid JSON in this format:
{{"domain": "ski", "destination": "resort_name", "days": 3, "people": 2, "budget": 1500, "special_requirements": []}}
"""
        return self._query_api(prompt)

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
        'Equipment': 'prompts/ski/step_to_code_equipment_gurobi.txt',
        'Car': 'prompts/ski/step_to_code_car_gurobi.txt',
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

class SkiPlannerGurobi:
    def __init__(self, use_mock=False, verbose=False):
        print(f"SKI PLANNER - Using {'Mock' if use_mock else 'Real GitHub Models API'} (Gurobi version)")
        self.verbose = verbose
        self.llm_client = MockLLM(verbose=verbose) if use_mock else LLMClient()
        self.ski_resorts = SkiResorts()
        self.ski_car = SkiCar()
        self.ski_rent = SkiRent()
        self.all_resorts = self.ski_resorts.data
        self.car_options = self.ski_car.data
        self.equipment_options = self.ski_rent.data
        self.model = None
        self.variables = None
        self.filtered_resorts = None
        if self.verbose:
            print("Ski Resorts, Car, and Equipment data loaded.")

    def query_to_json(self, query: str) -> Dict[str, Any]:
        if self.verbose:
            print("Converting natural language query to JSON...")
        response = self.llm_client.query_to_json_response(query)
        if not response:
            print("Error: LLM response is None.")
            return None
        try:
            json_str = re.search(r'\{.*\}', response, re.DOTALL).group(0)
            query_json = json.loads(json_str)
            query_json['query'] = query
            if self.verbose:
                print(f"Successfully converted query to JSON: {json.dumps(query_json, indent=2)}")
            return query_json
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing JSON from LLM response: {e}\nResponse: {response}")
            return None

    def build_gurobi_model(self, query_json, filtered_resorts, car_options, equipment_options):
        model = gp.Model("ski_trip_planner")
        variables = {}
        
        if self.verbose:
            print("--- Building Gurobi Model ---")
            print(f"Query JSON: {json.dumps(query_json, indent=2)}")

        # Decision variables
        variables['resort'] = model.addVars(len(filtered_resorts), vtype=GRB.BINARY, name="resort")
        variables['car'] = model.addVars(len(car_options), vtype=GRB.BINARY, name="car")
        variables['equipment'] = model.addVars(len(equipment_options), vtype=GRB.BINARY, name="equipment")

        # Constraints
        model.addConstr(gp.quicksum(variables['resort']) == 1, "one_resort")
        model.addConstr(gp.quicksum(variables['car']) <= 1, "one_car")
        
        people = query_json.get('people', 1)
        model.addConstr(gp.quicksum(variables['equipment']) <= people * len(self.ski_rent.get_equipment_types()), "equipment_per_person")

        days = query_json.get('days', 1)
        
        if query_json.get("car_type") or "car rental" in query_json.get("special_requirements", []):
            model.addConstr(gp.quicksum(variables['car']) == 1, "must_rent_car")
            if self.verbose:
                print("Constraint added: Must rent a car.")
        
        if query_json.get("equipment") or "equipment" in query_json.get("special_requirements", []):
            num_equipment_sets = people
            model.addConstr(gp.quicksum(variables['equipment']) >= num_equipment_sets, "must_rent_equipment")
            if self.verbose:
                print(f"Constraint added: Must rent at least {num_equipment_sets} sets of equipment.")

        # Define total cost expression
        total_cost = (
            gp.quicksum(variables['resort'][i] * filtered_resorts.iloc[i]['Price_day'] for i in range(len(filtered_resorts))) * days +
            gp.quicksum(variables['car'][i] * car_options.iloc[i]['Price_day'] for i in range(len(car_options))) * days +
            gp.quicksum(variables['equipment'][i] * equipment_options.iloc[i]['Price_day'] for i in range(len(equipment_options))) * days * people
        )

        # Budget constraint
        budget_val = query_json.get('budget')
        try:
            if budget_val is not None and budget_val != "":
                budget = float(budget_val)
                model.addConstr(total_cost <= budget, "budget")
                if self.verbose:
                    print(f"Constraint added: Budget <= {budget}")
        except (ValueError, TypeError):
            if self.verbose:
                print("No valid budget constraint applied.")

        # Objective function
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        if self.verbose:
            print("--- Gurobi Model Built ---")

        return model, variables

    def extract_plan_from_model(self, query_json):
        if not self.model or self.model.status != GRB.OPTIMAL:
            return "No optimal solution found."

        total_cost = self.model.objVal
        selected_resort_indices = [i for i, var in enumerate(self.variables['resort']) if self.variables['resort'][i].x > 0.5]
        selected_car_indices = [i for i, var in enumerate(self.variables['car']) if self.variables['car'][i].x > 0.5]
        selected_equipment_indices = [i for i, var in enumerate(self.variables['equipment']) if self.variables['equipment'][i].x > 0.5]

        selected_resort_df = self.filtered_resorts.iloc[selected_resort_indices]
        selected_car_df = self.car_options.iloc[selected_car_indices]
        selected_equipment_df = self.equipment_options.iloc[selected_equipment_indices]

        days = query_json.get('days', 1)
        people = query_json.get('people', 1)
        
        accommodation_cost = (selected_resort_df['Price_day'].sum() * days) if not selected_resort_df.empty else 0
        car_cost = (selected_car_df['Price_day'].sum() * days) if not selected_car_df.empty else 0
        equipment_cost = (selected_equipment_df['Price_day'].sum() * days * people) if not selected_equipment_df.empty else 0

        plan = "### GUROBI SKI TRIP PLAN\n"
        plan += f"**Query:** {query_json.get('query', 'N/A')}\n"
        plan += "**Result:** Optimal solution found!\n"
        
        # Add selected resort name prominently at the top
        selected_resort_name = "Unknown Resort"
        if not selected_resort_df.empty:
            resort = selected_resort_df.iloc[0]
            selected_resort_name = resort['Resort']
            plan += f"**Selected Resort:** {selected_resort_name}\n"
        
        plan += f"**Total Cost:** €{total_cost:.2f}\n"
        plan += "**Cost Breakdown:**\n"
        plan += f"- Accommodation: €{accommodation_cost:.2f}\n"
        if not selected_car_df.empty:
            plan += f"- Car Rental: €{car_cost:.2f}\n"
        if not selected_equipment_df.empty:
            plan += f"- Equipment Rental: €{equipment_cost:.2f}\n"
        
        plan += "\n"
        
        plan += "#### Accommodation\n"
        if not selected_resort_df.empty:
            for _, acc in selected_resort_df.iterrows():
                plan += f"- Stay at {acc['Resort']} for {days} days.\n"
        else:
            plan += "- No accommodation selected.\n"

        plan += "#### Car Rental\n"
        if not selected_car_df.empty:
            car = selected_car_df.iloc[0]
            plan += f"- Rented Car: {car['Type']} ({car['Fuel']}) for {days} days.\n"
        else:
            plan += "- No car rental selected.\n"

        plan += "#### Equipment Rental\n"
        if not selected_equipment_df.empty:
            equipment_summary = selected_equipment_df['Equipment'].value_counts().to_dict()
            plan += f"- Rented Equipment for {people} people for {days} days:\n"
            for item, count in equipment_summary.items():
                plan += f"  - {item} (x{count})\n"
        else:
            plan += "- No equipment rental selected.\n"
            
        return plan

    def analyze_infeasible_model(self, query_json):
        if not self.model or self.model.status != GRB.INFEASIBLE:
            return "Model is not infeasible."

        print("Model is infeasible. Computing IIS to find conflicting constraints...")
        self.model.computeIIS()
        
        iis_file = f"iis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ilp"
        self.model.write(iis_file)
        print(f"IIS report saved to {iis_file}")

        conflicting_constraints = []
        for constr in self.model.getConstrs():
            if constr.IISConstr:
                conflicting_constraints.append(constr.constrName)
        
        for qconstr in self.model.getQConstrs():
            if qconstr.IISQConstr:
                conflicting_constraints.append(qconstr.QCName)

        if not conflicting_constraints:
            return "Gurobi model is infeasible, but no specific conflicting constraints were identified by IIS."

        suggestion = self.generate_suggestion_from_iis(conflicting_constraints, query_json)
        
        return f"The query is infeasible. {suggestion}"

    def generate_suggestion_from_iis(self, conflicting_constraints, query_json):
        suggestions = []
        if "budget" in conflicting_constraints:
            suggestions.append(f"the budget of €{query_json.get('budget')} is too low for the requested trip.")
        if "one_resort" in conflicting_constraints:
            suggestions.append(f"no resorts could be found for the destination '{query_json.get('destination')}' that meet all criteria.")
        if "must_rent_car" in conflicting_constraints:
            suggestions.append("car rental is required but might not be possible within the budget or other constraints.")
        if "must_rent_equipment" in conflicting_constraints:
            suggestions.append("equipment rental is required but might not be possible within the budget or other constraints.")

        if not suggestions:
            return f"The combination of constraints ({', '.join(conflicting_constraints)}) is too restrictive."

        return "Suggestion: try relaxing some constraints, for example, " + " or ".join(suggestions)

    def run_gurobi_planner(self, query_json):
        destination = query_json.get("destination")
        if not destination:
            print("Destination not specified in the query.")
            return

        # Apply fallback parsing for special requirements if needed
        special_requirements = query_json.get('special_requirements', [])
        if not special_requirements:
            original_query = query_json.get('query', '')
            fallback_requirements = extract_special_requirements_fallback(original_query)
            if fallback_requirements:
                query_json['special_requirements'] = fallback_requirements
                if self.verbose:
                    print(f"🔄 Using fallback parsing for special requirements: {fallback_requirements}")

        # Filter resorts and reset index to ensure it's 0-based
        self.filtered_resorts = self.ski_resorts.run(destination=destination, query=f"Find ski resorts in or near {destination}").reset_index(drop=True)
        
        if self.filtered_resorts.empty:
            print(f"No resorts found for destination: {destination}")
            return

        if self.verbose:
            print(f"Debug: Found {len(self.filtered_resorts)} resorts matching destination '{destination}'")
            print(f"Debug: Filtered resorts: {self.filtered_resorts['Resort'].tolist()}")

        # Reset indices for car and equipment options as well
        self.car_options.reset_index(drop=True, inplace=True)
        self.equipment_options.reset_index(drop=True, inplace=True)

        self.model, self.variables = self.build_gurobi_model(query_json, self.filtered_resorts, self.car_options, self.equipment_options)
        self.model.optimize()

    def run_and_save_plan(self, query: str, output_dir: str, test_name: str):
        if self.verbose:
            print("Starting main logic...")
        
        query_json = self.query_to_json(query)
        if not query_json:
            print("Failed to convert query to JSON. Aborting.")
            return

        self.run_gurobi_planner(query_json)

        if self.model and self.model.status == GRB.OPTIMAL:
            plan = self.extract_plan_from_model(query_json)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_str = "gurobi"
            if hasattr(self.llm_client, 'model_name'):
                model_name_str = self.llm_client.model_name.replace('/', '_')

            save_path = os.path.join(output_dir, model_name_str, test_name, f"{timestamp}_plan.txt")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(plan)

            print(f"Plan saved to: {save_path}")
            print("\nGenerated Ski Plan:")
            print(plan)
        else:
            print("Failed to generate a plan.")

# Ensure the main logic is executed
print("SKI PLANNER - Using Real GitHub Models API (Gurobi version)")

def pipeline_ski(query: str, mode: str, model: str, index: int, model_version: str = None, verbose: bool = False, fallback_models: list = None) -> str:
    """
    Main pipeline for the Gurobi ski planner.
    This function is the entry point for the benchmarking script.
    """
    try:
        planner = SkiPlannerGurobi(use_mock=(mode == 'mock'), verbose=verbose)
        
        if hasattr(planner.llm_client, 'model_name'):
            planner.llm_client.model_name = model
        if hasattr(planner.llm_client, 'fallback_models'):
            planner.llm_client.fallback_models = fallback_models or []

        query_json = planner.query_to_json(query)
        if not query_json:
            return "Failed to convert query to JSON."
        
        # Store original query for fallback parsing
        query_json['query'] = query

        planner.run_gurobi_planner(query_json)

        if planner.model and planner.model.status == GRB.OPTIMAL:
            plan = planner.extract_plan_from_model(query_json)
            return plan
        elif planner.model and planner.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Analyzing IIS for suggestions...")
            suggestion = planner.analyze_infeasible_model(query_json)
            return suggestion
        elif planner.model:
            # Return a descriptive message for other non-optimal statuses
            return f"Gurobi optimization failed with status: {planner.model.status}"
        else:
            return "Gurobi planner failed to run or find a solution."

    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        return f"An error occurred in the Gurobi pipeline: {str(e)}"

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

def extract_special_requirements_fallback(query: str) -> list:
    """Fallback method to extract special requirements using regex"""
    requirements = []
    query_lower = query.lower()
    
    # Check for car rental requirements
    car_patterns = [
        r'need.*car.*rental', r'car.*rental', r'rent.*car', r'rental.*car',
        r'need.*car', r'want.*car', r'require.*car'
    ]
    if any(re.search(pattern, query_lower) for pattern in car_patterns):
        requirements.append('car rental')
    
    # Check for equipment requirements
    equipment_patterns = [
        r'need.*equipment', r'equipment.*rental', r'rent.*equipment', 
        r'ski.*equipment', r'equipment.*rent', r'need.*ski.*gear'
    ]
    if any(re.search(pattern, query_lower) for pattern in equipment_patterns):
        requirements.append('equipment')
    
    return requirements
