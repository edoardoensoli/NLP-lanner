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

load_dotenv()

# Global LLM client for fallback
llm_client = None

# Initialize data loaders
class SkiResorts:
    def __init__(self):
        try:
            self.data = pd.read_csv('dataset_ski/resorts/resorts.csv')
        except FileNotFoundError:
            print("Warning: resorts.csv not found, using mock data")
            self.data = pd.DataFrame({
                'Resort': ['Mock Resort'],
                'Country': ['Mock Country'],
                'Price_day': [150],
                'Rating': [4.0],
                'Beds': [2],
                'Access': ['Road']
            })

class SkiCar:
    def __init__(self):
        try:
            self.data = pd.read_csv('dataset_ski/car/ski_car.csv')
        except FileNotFoundError:
            print("Warning: ski_car.csv not found, using mock data")
            self.data = pd.DataFrame({
                'Brand': ['Mock Brand'],
                'Model': ['Mock Model'],
                'Type': ['SUV'],
                'Fuel': ['Gasoline'],
                'Price_day': [80]
            })

class SkiRent:
    def __init__(self):
        try:
            self.data = pd.read_csv('dataset_ski/rent/ski_rent.csv')
        except FileNotFoundError:
            print("Warning: ski_rent.csv not found, using mock data")
            self.data = pd.DataFrame({
                'Equipment': ['Skis', 'Boots', 'Helmet', 'Poles'],
                'Price_day': [25, 15, 10, 8]
            })

class SkiSlopes:
    def __init__(self):
        try:
            self.data = pd.read_csv('dataset_ski/slopes/ski_slopes.csv')
        except FileNotFoundError:
            print("Warning: ski_slopes.csv not found, using mock data")
            self.data = pd.DataFrame({
                'Slope': ['Mock Slope'],
                'Difficulty': ['Intermediate'],
                'Length': [2000]
            })

class LLMClient:
    """Client for interacting with a generic LLM API."""
    def __init__(self, model_name: str = None, fallback_models: list = None):
        load_dotenv()
        self.api_key = os.getenv("API_TOKEN")
        if not self.api_key:
            raise ValueError("API_TOKEN not found in .env file.")
        
        self.base_url = "https://models.inference.ai.azure.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.model_name = model_name or "gpt-4o-mini"
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
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in ski trip planning."},
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
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    def _query_api(self, prompt):
        """Query API with current model"""
        response, model = self._query_api_with_fallback(prompt)
        return response

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
        
        return self._query_api(prompt)

class SkiPlannerGurobiExtended:
    def __init__(self, use_mock=False, verbose=False):
        print(f"SKI PLANNER EXTENDED - Using {'Mock' if use_mock else 'Real GitHub Models API'} (Gurobi version)")
        self.verbose = verbose
        self.llm_client = LLMClient() if not use_mock else None
        self.ski_resorts = SkiResorts()
        self.ski_car = SkiCar()
        self.ski_rent = SkiRent()
        self.ski_slopes = SkiSlopes()
        self.all_resorts = self.ski_resorts.data
        self.car_options = self.ski_car.data
        self.equipment_options = self.ski_rent.data
        self.slope_options = self.ski_slopes.data
        self.model = None
        self.variables = None
        self.filtered_resorts = None
        if self.verbose:
            print("Ski Resorts, Car, Equipment, and Slopes data loaded.")

    def query_to_json(self, query: str) -> Dict[str, Any]:
        """Convert natural language query to JSON"""
        if self.verbose:
            print("Converting natural language query to JSON...")
        
        if self.llm_client:
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
        else:
            # Fallback parsing for mock mode
            return self._fallback_parse_query(query)

    def _fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Fallback query parsing when LLM is not available"""
        query_lower = query.lower()
        
        # Extract destination
        destination = "Unknown"
        countries = ["france", "italy", "switzerland", "austria", "germany"]
        for country in countries:
            if country in query_lower:
                destination = country.title()
                break
        
        # Extract basic parameters
        days = 3
        if "day" in query_lower:
            days_match = re.search(r'(\d+)\s*day', query_lower)
            if days_match:
                days = int(days_match.group(1))
        
        people = 2
        if "people" in query_lower:
            people_match = re.search(r'(\d+)\s*people', query_lower)
            if people_match:
                people = int(people_match.group(1))
        
        budget = None
        if "budget" in query_lower or "euro" in query_lower:
            budget_match = re.search(r'(\d+)\s*euro', query_lower)
            if budget_match:
                budget = int(budget_match.group(1))
        
        # Extract car preferences
        car_type = None
        if "suv" in query_lower:
            car_type = "SUV"
        elif "sedan" in query_lower:
            car_type = "Sedan"
        elif "hatchback" in query_lower:
            car_type = "Hatchback"
        
        fuel_type = None
        if "diesel" in query_lower:
            fuel_type = "Diesel"
        elif "gasoline" in query_lower or "petrol" in query_lower:
            fuel_type = "Gasoline"
        elif "electric" in query_lower:
            fuel_type = "Electric"
        
        # Extract equipment
        equipment = None
        if "equipment" in query_lower or "ski" in query_lower:
            equipment_list = []
            if "skis" in query_lower:
                equipment_list.append("Skis")
            if "boots" in query_lower:
                equipment_list.append("Boots")
            if "helmet" in query_lower:
                equipment_list.append("Helmet")
            if "poles" in query_lower:
                equipment_list.append("Poles")
            
            if equipment_list:
                equipment = ", ".join(equipment_list)
        
        # Extract slope difficulty
        slope_difficulty = None
        if "easy" in query_lower or "beginner" in query_lower:
            slope_difficulty = "Easy"
        elif "intermediate" in query_lower:
            slope_difficulty = "Intermediate"
        elif "difficult" in query_lower or "advanced" in query_lower:
            slope_difficulty = "Difficult"
        
        return {
            "domain": "ski",
            "destination": destination,
            "days": days,
            "people": people,
            "budget": budget,
            "car_type": car_type,
            "fuel_type": fuel_type,
            "equipment": equipment,
            "slope_difficulty": slope_difficulty,
            "query": query
        }

    def filter_data_by_preferences(self, query_json):
        """Filter data based on user preferences"""
        destination = query_json.get('destination', '').lower()
        car_type = query_json.get('car_type', '').lower()
        fuel_type = query_json.get('fuel_type', '').lower()
        
        # Handle equipment (can be list or string)
        equipment_req = query_json.get('equipment', '')
        if isinstance(equipment_req, list):
            equipment = ' '.join(equipment_req).lower()
        elif isinstance(equipment_req, str):
            equipment = equipment_req.lower()
        else:
            equipment = ''
            
        slope_difficulty = query_json.get('slope_difficulty', '').lower()
        
        # Filter resorts by destination
        filtered_resorts = self.all_resorts.copy()
        if destination and destination != "unknown":
            filtered_resorts = filtered_resorts[
                filtered_resorts['Country'].str.lower().str.contains(destination, na=False)
            ]
        
        # Filter cars by type and fuel
        filtered_cars = self.car_options.copy()
        if car_type:
            filtered_cars = filtered_cars[
                filtered_cars['Type'].str.lower().str.contains(car_type, na=False)
            ]
        if fuel_type:
            filtered_cars = filtered_cars[
                filtered_cars['Fuel'].str.lower().str.contains(fuel_type, na=False)
            ]
        
        # Filter equipment by type
        filtered_equipment = self.equipment_options.copy()
        if equipment:
            equipment_mask = pd.Series([False] * len(filtered_equipment))
            for eq_type in ['skis', 'boots', 'helmet', 'poles']:
                if eq_type in equipment:
                    equipment_mask |= filtered_equipment['Equipment'].str.lower().str.contains(eq_type, na=False)
            filtered_equipment = filtered_equipment[equipment_mask]
        
        # Filter slopes by difficulty
        filtered_slopes = self.slope_options.copy()
        if slope_difficulty:
            filtered_slopes = filtered_slopes[
                filtered_slopes['Difficulty'].str.lower().str.contains(slope_difficulty, na=False)
            ]
        
        return {
            'resorts': filtered_resorts,
            'cars': filtered_cars,
            'equipment': filtered_equipment,
            'slopes': filtered_slopes
        }

    def build_gurobi_model(self, query_json, filtered_data):
        """Build Gurobi optimization model"""
        model = gp.Model("ski_trip_planner_extended")
        variables = {}
        
        if self.verbose:
            print("--- Building Gurobi Extended Model ---")
            print(f"Query JSON: {json.dumps(query_json, indent=2)}")

        filtered_resorts = filtered_data['resorts']
        filtered_cars = filtered_data['cars']
        filtered_equipment = filtered_data['equipment']
        filtered_slopes = filtered_data['slopes']

        # Decision variables
        variables['resort'] = model.addVars(len(filtered_resorts), vtype=GRB.BINARY, name="resort")
        variables['car'] = model.addVars(len(filtered_cars), vtype=GRB.BINARY, name="car")
        variables['equipment'] = model.addVars(len(filtered_equipment), vtype=GRB.BINARY, name="equipment")
        variables['slope'] = model.addVars(len(filtered_slopes), vtype=GRB.BINARY, name="slope")

        # Constraints
        
        # 1. Exactly one resort must be selected
        model.addConstr(gp.quicksum(variables['resort'][i] for i in range(len(filtered_resorts))) == 1, "select_one_resort")
        
        # 2. At most one car can be selected
        model.addConstr(gp.quicksum(variables['car'][i] for i in range(len(filtered_cars))) <= 1, "select_max_one_car")
        
        # 3. Equipment constraints - if equipment is requested, ensure basic equipment is selected
        equipment_requested = query_json.get('equipment') is not None
        if equipment_requested and len(filtered_equipment) > 0:
            # Ensure at least basic equipment (skis and boots) is selected
            skis_indices = [i for i, eq in enumerate(filtered_equipment.to_dict('records')) if 'skis' in eq.get('Equipment', '').lower()]
            boots_indices = [i for i, eq in enumerate(filtered_equipment.to_dict('records')) if 'boots' in eq.get('Equipment', '').lower()]
            
            if skis_indices:
                model.addConstr(gp.quicksum(variables['equipment'][i] for i in skis_indices) >= 1, "select_skis")
            if boots_indices:
                model.addConstr(gp.quicksum(variables['equipment'][i] for i in boots_indices) >= 1, "select_boots")
        
        # 4. Slope selection (informational - can select multiple)
        # No strict constraints on slopes as they're informational
        
        # Cost calculation
        days = query_json.get('days', 3)
        people = query_json.get('people', 2)
        budget = query_json.get('budget', 1000000)  # Large number if no budget specified
        
        # Resort cost
        resort_cost = gp.quicksum(
            variables['resort'][i] * filtered_resorts.iloc[i]['Price_day'] * days
            for i in range(len(filtered_resorts))
        )
        
        # Car cost
        car_cost = gp.quicksum(
            variables['car'][i] * filtered_cars.iloc[i]['Price_day'] * days
            for i in range(len(filtered_cars))
        )
        
        # Equipment cost
        equipment_cost = gp.quicksum(
            variables['equipment'][i] * filtered_equipment.iloc[i]['Price_day'] * days * people
            for i in range(len(filtered_equipment))
        )
        
        # Total cost
        total_cost = resort_cost + car_cost + equipment_cost
        
        # Budget constraint
        model.addConstr(total_cost <= budget, "budget_constraint")
        
        # Objective: minimize total cost
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        # Store data for result interpretation
        self.model = model
        self.variables = variables
        self.filtered_data = filtered_data
        self.query_json = query_json
        
        return model, variables

    def solve_and_generate_plan(self):
        """Solve the optimization problem and generate plan"""
        if self.model is None:
            return "Error: Model not built yet"
        
        # Solve the model
        self.model.optimize()
        
        if self.model.status == GRB.Status.OPTIMAL:
            if self.verbose:
                print("✅ Gurobi found an optimal solution!")
            
            # Extract solution
            filtered_resorts = self.filtered_data['resorts']
            filtered_cars = self.filtered_data['cars']
            filtered_equipment = self.filtered_data['equipment']
            filtered_slopes = self.filtered_data['slopes']
            
            selected_resort = None
            selected_cars = []
            selected_equipment = []
            selected_slopes = []
            
            # Get selected resort
            for i in range(len(filtered_resorts)):
                if self.variables['resort'][i].x > 0.5:
                    selected_resort = filtered_resorts.iloc[i].to_dict()
                    break
            
            # Get selected cars
            for i in range(len(filtered_cars)):
                if self.variables['car'][i].x > 0.5:
                    selected_cars.append(filtered_cars.iloc[i].to_dict())
            
            # Get selected equipment
            for i in range(len(filtered_equipment)):
                if self.variables['equipment'][i].x > 0.5:
                    selected_equipment.append(filtered_equipment.iloc[i].to_dict())
            
            # Get available slopes (informational)
            if not filtered_slopes.empty:
                selected_slopes = filtered_slopes.head(3).to_dict('records')  # Show first 3 slopes
            
            # Calculate costs
            days = self.query_json.get('days', 3)
            people = self.query_json.get('people', 2)
            budget = self.query_json.get('budget', 1000000)
            
            resort_cost = selected_resort['Price_day'] * days if selected_resort else 0
            car_cost = sum(car['Price_day'] * days for car in selected_cars)
            equipment_cost = sum(eq['Price_day'] * days * people for eq in selected_equipment)
            total_cost = resort_cost + car_cost + equipment_cost
            
            # Generate comprehensive plan
            plan = f"""GUROBI EXTENDED SKI TRIP PLAN:

DESTINATION: {self.query_json.get('destination', 'Unknown')}
DURATION: {days} days
PEOPLE: {people}
BUDGET: €{budget if budget < 1000000 else 'Not specified'}

SELECTED RESORT:
- Resort: {selected_resort.get('Resort', 'Unknown') if selected_resort else 'None'}
- Country: {selected_resort.get('Country', 'Unknown') if selected_resort else 'None'}
- Price per day: €{selected_resort.get('Price_day', 0) if selected_resort else 0}
- Rating: {selected_resort.get('Rating', 'Unknown') if selected_resort else 'Unknown'}/5.0
- Beds: {selected_resort.get('Beds', 'Unknown') if selected_resort else 'Unknown'}
- Access: {selected_resort.get('Access', 'Unknown') if selected_resort else 'Unknown'}

SELECTED CAR(S):
"""
            
            if selected_cars:
                for car in selected_cars:
                    plan += f"- {car.get('Brand', 'Unknown')} {car.get('Model', 'Unknown')} ({car.get('Type', 'Unknown')})\n"
                    plan += f"  Fuel: {car.get('Fuel', 'Unknown')}, Price: €{car.get('Price_day', 0)}/day\n"
            else:
                plan += "- No car rental selected\n"
            
            plan += "\nSELECTED EQUIPMENT:\n"
            if selected_equipment:
                for eq in selected_equipment:
                    plan += f"- {eq.get('Equipment', 'Unknown')}: €{eq.get('Price_day', 0)}/day per person\n"
            else:
                plan += "- No equipment rental selected\n"
            
            plan += "\nAVAILABLE SLOPES:\n"
            if selected_slopes:
                for slope in selected_slopes:
                    plan += f"- {slope.get('Slope', 'Unknown')}: {slope.get('Difficulty', 'Unknown')} ({slope.get('Length', 'Unknown')}m)\n"
            else:
                plan += "- No slope information available\n"
            
            plan += f"""
COST BREAKDOWN:
- Resort: €{resort_cost:.2f}
- Car rental: €{car_cost:.2f}
- Equipment: €{equipment_cost:.2f}
- TOTAL COST: €{total_cost:.2f}

BUDGET STATUS: {'✅ Within budget' if total_cost <= budget else '❌ Over budget'}

Generated by: Gurobi Extended Optimization Solver
Optimization Status: Optimal
Data Sources: Real CSV data
"""
            
            return plan
        
        elif self.model.status == GRB.Status.INFEASIBLE:
            return "INFEASIBLE: No solution found within the given budget and constraints. Try increasing the budget or relaxing requirements."
        
        else:
            return f"SOLVER ERROR: Gurobi status {self.model.status}"

    def plan_ski_trip(self, query: str):
        """Main function to plan ski trip"""
        if self.verbose:
            print(f"Planning ski trip for query: {query}")
        
        # Step 1: Convert query to JSON
        query_json = self.query_to_json(query)
        if not query_json:
            return "Error: Could not parse query"
        
        # Step 2: Filter data based on preferences
        filtered_data = self.filter_data_by_preferences(query_json)
        
        # Check if we have data
        if filtered_data['resorts'].empty:
            return "Error: No resorts found for the specified destination"
        
        # Step 3: Build Gurobi model
        model, variables = self.build_gurobi_model(query_json, filtered_data)
        
        # Step 4: Solve and generate plan
        plan = self.solve_and_generate_plan()
        
        return plan

def main():
    """Test the Gurobi extended ski planner"""
    import sys
    
    if len(sys.argv) > 1:
        test_query = sys.argv[1]
    else:
        test_query = "Plan a 5-day ski trip to Switzerland for 3 people with budget 4000 euros. We need an SUV with diesel fuel, ski equipment including skis and boots, and prefer intermediate slopes."
    
    print("Testing Gurobi Extended Ski Planner...")
    print(f"Query: {test_query}")
    
    # Initialize planner
    planner = SkiPlannerGurobiExtended(use_mock=False, verbose=True)
    
    # Plan trip
    result = planner.plan_ski_trip(test_query)
    
    if result:
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(result)
    else:
        print("❌ Failed to generate plan")

if __name__ == "__main__":
    main()
