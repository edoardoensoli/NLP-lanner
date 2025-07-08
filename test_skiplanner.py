import re, string, os, sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tools/planner")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../tools/planner")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import importlib
from typing import List, Dict, Any
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
from utils.func import load_line_json_data, save_file
import sys
import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
from datasets import load_dataset
import os
import pdb
import json
from z3 import *
import numpy as np

# Use tools_ski for ski-specific APIs
from tools_ski.apis import *

# Load ski resorts mapping dynamically from CSV
def load_ski_destinations_mapping():
    """Load dynamic country/region to resorts mapping from CSV data"""
    try:
        df = pd.read_csv('dataset_ski/resorts/resorts.csv')
        mapping = {}
        
        # Group resorts by country
        for _, row in df.iterrows():
            country = row['Country']
            resort = row['Resort']
            
            if country not in mapping:
                mapping[country] = []
            if resort not in mapping[country]:
                mapping[country].append(resort)
        
        # Create a flat lookup for easy access
        lookup = {}
        
        # Add country-based lookups (lowercase for matching)
        for country, resorts in mapping.items():
            lookup[country.lower()] = resorts[0]  # Default to first resort
            
            # Add specific resort lookups
            for resort in resorts:
                lookup[resort.lower()] = resort
        
        # Add common aliases and region names
        aliases = {
            "italy": "Italy",
            "italia": "Italy", 
            "norway": "Norway",
            "norvegia": "Norway",
            "austria": "Austria",
            "france": "France", 
            "francia": "France",
            "canada": "Canada",
            "united states": "United States",
            "usa": "United States",
            "germany": "Germany",
            "germania": "Germany", 
            "united kingdom": "United Kingdom",
            "uk": "United Kingdom",
            "scotland": "United Kingdom",
            "switzerland": "Switzerland",
            "japan": "Japan",
            "australia": "Australia", 
            "new zealand": "New Zealand",
            "chile": "Chile",
            "argentina": "Argentina",
            "finland": "Finland",
            "sweden": "Sweden",
            "south korea": "South Korea"
        }
        
        # Add aliases to lookup
        for alias, country in aliases.items():
            if country in mapping and alias not in lookup:
                lookup[alias] = mapping[country][0]  # Default to first resort
        
        return lookup, mapping
        
    except Exception as e:
        print(f"Error loading ski destinations: {e}")
        # Fallback to basic mapping
        return {
            "livigno": "Livigno",
            "italy": "Livigno",
            "hemsedal": "Hemsedal", 
            "norway": "Hemsedal"
        }, {}

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
            print("Mock LLM initialized - Using real prompts, no API costs!")
        self.prompts = load_ski_prompts(verbose)
        self.verbose = verbose
    def query_to_json_response(self, query):
        """Mock response for query to JSON conversion using real prompt structure"""
        # Enhanced parsing for ski queries to extract car rental information
        query_lower = query.lower()
        
        # Load dynamic ski destinations mapping
        ski_destinations, country_resorts_mapping = load_ski_destinations_mapping()
        
        # Extract destination from ski resorts using dynamic mapping
        available_resorts = []  # List of possible resorts
        destination_country = None  # Matched country
        
        for country, resorts_list in country_resorts_mapping.items():
            if country.lower() in query_lower:
                available_resorts = resorts_list
                destination_country = country
                break
        
        # If no country found, try to match specific resort names
        if not available_resorts:
            # Read all resorts to find specific resort mentions
            all_resorts = self._get_all_resort_names()
            for resort in all_resorts:
                if resort.lower() in query_lower:
                    available_resorts = [resort]
                    destination_country = f"specific:{resort}"
                    break
        
        # Default fallback
        if not available_resorts:
            available_resorts = ["Livigno"]
            destination_country = "italy"
        
        # Store for dynamic code generation  
        self._last_resorts = available_resorts
        self._last_country = destination_country
        
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
        
        # Extract equipment with better detection - handle partial equipment rental
        equipment = []
        equipment_people = None
        if "equipment" in query_lower or "rental" in query_lower or "rent" in query_lower:
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
            "destination": available_resorts[0] if available_resorts else "Livigno",  # Primary destination
            "available_resorts": available_resorts,  # All available resorts for this region
            "destination_country": destination_country,  # Country or region matched
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
            'Destination cities': self._get_destination_code(),
            
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

    def _get_destination_code(self):
        """Generate dynamic destination code based on the last parsed destination"""
        # Get the country from destination
        destination = getattr(self, '_last_destination', 'Livigno')
        
        # Map destination to country and resort list
        country_resort_mapping = {
            # Norway
            'Hemsedal': ('Norway', ['Hemsedal', 'Geilosiden Geilo', 'Voss']),
            'Geilosiden Geilo': ('Norway', ['Hemsedal', 'Geilosiden Geilo', 'Voss']),
            'Voss': ('Norway', ['Hemsedal', 'Geilosiden Geilo', 'Voss']),
            
            # Canada
            'Fernie': ('Canada', ['Fernie', 'Sun Peaks', 'Panorama']),
            'Sun Peaks': ('Canada', ['Fernie', 'Sun Peaks', 'Panorama']),
            'Panorama': ('Canada', ['Fernie', 'Sun Peaks', 'Panorama']),
            'Red Mountain Resort-Rossland': ('Canada', ['Red Mountain Resort-Rossland', 'Fernie', 'Sun Peaks']),
            
            # Austria
            'Golm': ('Austria', ['Golm']),
            
            # Switzerland  
            'Zermatt': ('Switzerland', ['Zermatt', 'St. Moritz', 'Verbier']),
            'St. Moritz': ('Switzerland', ['Zermatt', 'St. Moritz', 'Verbier']),
            'Verbier': ('Switzerland', ['Zermatt', 'St. Moritz', 'Verbier']),
            'Davos': ('Switzerland', ['Davos', 'Klosters', 'Saas-Fee']),
            'Klosters': ('Switzerland', ['Davos', 'Klosters', 'Saas-Fee']),
            'Saas-Fee': ('Switzerland', ['Davos', 'Klosters', 'Saas-Fee']),
            'Engelberg': ('Switzerland', ['Engelberg', 'Crans-Montana']),
            'Crans-Montana': ('Switzerland', ['Engelberg', 'Crans-Montana']),
            
            # United States
            'Jackson Hole': ('United States', ['Jackson Hole', 'Steamboat', 'Crested Butte']),
            'Steamboat': ('United States', ['Jackson Hole', 'Steamboat', 'Crested Butte']),
            'Crested Butte': ('United States', ['Jackson Hole', 'Steamboat', 'Crested Butte']),
            
            # Germany  
            'Hochschwarzeck': ('Germany', ['Hochschwarzeck', 'Rossfeld - Berchtesgaden - Oberau', 'Brauneck Lenggries']),
            
            # Italy (default)
            'Livigno': ('Italy', ['Livigno', 'Cortina', 'Madonna di Campiglio']),
            'Cortina d\'Ampezzo': ('Italy', ['Livigno', 'Cortina', 'Madonna di Campiglio']),
            'Val Gardena': ('Italy', ['Val Gardena', 'Gressoney']),
            'Madonna di Campiglio': ('Italy', ['Livigno', 'Cortina', 'Madonna di Campiglio']),
            'Sestriere': ('Italy', ['Sestriere', 'La Thuile', 'Courmayeur']),
        }
        
        # Get country and resorts for destination
        if destination in country_resort_mapping:
            country, resorts = country_resort_mapping[destination]
        else:
            # Default fallback
            country, resorts = 'Italy', ['Livigno', 'Cortina d\'Ampezzo', 'Madonna di Campiglio']
        
        # Generate dynamic code
        resort_list = str(resorts)  # Keep single quotes for Python list
        return f"""# Destination setup
destination_country = '{country}'
selected_resorts = {resort_list}
all_resorts = selected_resorts"""

# Mock functions for API responses using real prompts
def GPT_response(prompt, model_version=None, verbose=True):
    """Mock GPT response using real prompt structure"""
    mock_llm = MockLLM(verbose=verbose)
    
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
    """Mock Claude response using real prompt structure"""
    return GPT_response(prompt, verbose=verbose)

def Mixtral_response(prompt, format_type=None, verbose=True):
    """Mock Mixtral response using real prompt structure"""
    return GPT_response(prompt, verbose=verbose)

if __name__ == "__main__" or True:  # Ensure it's printed once
    if False:  # Disable the print statement
        print("🆓 Mock LLM functions loaded - Using real prompts, no API costs!")

def convert_to_int(real):
    out = ToInt(real)
    out += If(real == out, 0, 1)
    return out

def generate_ski_plan(s, variables, query, ski_apis):
    """Generate ski plan from Z3 solution using existing API instances"""
    SkiResortSearch, SkiSlopeSearch, SkiRentSearch, SkiCarSearch = ski_apis
    
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
        
        # Get slope details for the destination
        query_text = query.get('query', '').lower()
        if "advanced" in query_text or "expert" in query_text:
            skill_level = "advanced"
        elif "intermediate" in query_text:
            skill_level = "intermediate"
        elif "not expert" in query_text or "beginner" in query_text:
            skill_level = "beginner"
        else:
            skill_level = "intermediate"  # default
        
        slope_details = get_slope_details(query.get('destination'), skill_level, query.get('people', 2))
        
        # Format match quality message
        match_quality_msg = ""
        if slope_details['match_quality'] == 'perfect':
            match_quality_msg = "[PERFECT] Perfect match for your requirements!"
        elif slope_details['match_quality'] == 'good':
            match_quality_msg = "[GOOD] Good match with minor limitations"
        elif slope_details['match_quality'] == 'partial':
            match_quality_msg = "[PARTIAL] Partial match - some requirements not met"
        else:
            match_quality_msg = "[LIMITED] Limited information available"
        
        # Format missing requirements
        missing_req_msg = ""
        if slope_details['missing_requirements']:
            missing_req_msg = f"\nMissing Requirements: {', '.join(slope_details['missing_requirements'])}"
        
        plan = f"""SKI TRIP PLAN:
Destination: {query.get('destination', 'Unknown')}
Best Resort: {slope_details['resort_name']} (Rating: {slope_details.get('rating', 0):.1f}/5.0)
Duration: {query.get('days', 3)} days
People: {query.get('people', 2)}
Budget: €{query.get('budget', 1500)}

Resort Match: {match_quality_msg}{missing_req_msg}

Ski Slopes: {', '.join(slopes) if slopes else 'Available slopes at destination'}
Slope Details: {slope_details['total_slopes']} total slopes, longest run {slope_details['longest_run']} km
Available Difficulty: {slope_details['difficulty_available']}
Recommended for You: {slope_details['recommended_difficulty']} slopes
Equipment Rental: {', '.join(equipment) if equipment else 'Local rental shops'}
Transportation: {transportation_str if transportation else 'To be arranged'}
Accommodation: {accommodation if accommodation else f"{slope_details['resort_name']} Lodge"}
"""
        
        return plan
        
    except Exception as e:
        print(f"Error generating ski plan: {e}")
        return f"Ski trip plan for {query['dest']} - {query['days']} days"

def get_plan(query, number, args):
    # This function will replace the pipeline_ski function
    # It will take the query, number, and args as input
    # and return the plan and results
    pass

def pipeline_ski(query, mode, model, index, model_version=None, verbose=False):
    """Pipeline for ski trip planning using real prompt files"""
    path = f'output_ski/{mode}/mock_llm/{index}/'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+'codes/')
        os.makedirs(path+'plans/')
    
    # Load ski-specific prompts from real files (silently)
    with open('prompts/ski/query_to_json_ski.txt', 'r', encoding='utf-8') as file:
        query_to_json_prompt = file.read()
    with open('prompts/ski/constraint_to_step_ski.txt', 'r', encoding='utf-8') as file:
        constraint_to_step_prompt = file.read()
    
    # Load step-to-code prompts from real files
    with open('prompts/ski/step_to_code_destination.txt', 'r', encoding='utf-8') as file:
        step_to_code_destination_prompt = file.read()  
    with open('prompts/ski/step_to_code_resort.txt', 'r', encoding='utf-8') as file:
        step_to_code_resort_prompt = file.read()
    with open('prompts/ski/step_to_code_slopes.txt', 'r', encoding='utf-8') as file:
        step_to_code_slopes_prompt = file.read()
    with open('prompts/ski/step_to_code_equipment.txt', 'r', encoding='utf-8') as file:
        step_to_code_equipment_prompt = file.read()
    with open('prompts/ski/step_to_code_car.txt', 'r', encoding='utf-8') as file:
        step_to_code_car_prompt = file.read()
    with open('prompts/ski/step_to_code_budget.txt', 'r', encoding='utf-8') as file:
        step_to_code_budget_prompt = file.read()
    
    # Initialize ski APIs (silently to avoid redundant messages)
    print("Initializing ski planner...")
    SkiResortSearch = SkiResorts()
    SkiSlopeSearch = SkiSlopes()
    SkiRentSearch = SkiRent()
    SkiCarSearch = SkiCar()
    print("Ready to find your perfect ski trip!")
    print("="*60)
    
    s = Optimize()
    variables = {}
    times = []
    
    # Map step-to-code prompts using real files
    step_to_code_prompts = {
        'Destination cities': step_to_code_destination_prompt,
        'Ski resort': step_to_code_resort_prompt,
        'Ski slopes': step_to_code_slopes_prompt,
        'Ski equipment': step_to_code_equipment_prompt,
        'Car rental': step_to_code_car_prompt,
        'Budget': step_to_code_budget_prompt,
        'Departure dates': '',  # Can be added if needed
        'Transportation methods': '',  # Can be added if needed  
        'Accommodation': ''  # Can be added if needed
    }
    
    if verbose:
        print("Loaded ski prompts from real files (not hardcoded!)")
    
    plan = ''
    codes = ''
    success = False
    
    try:
        # Convert query to JSON using mock LLM
        query_json_str = GPT_response(query_to_json_prompt + '{' + query + '}\n' + 'JSON:\n', model_version, verbose)
        query_json = json.loads(query_json_str.replace('```json', '').replace('```', ''))
        
        with open(path+'plans/' + 'query.txt', 'w') as f:
            f.write(query)
        f.close()
        
        with open(path+'plans/' + 'query.json', 'w') as f:
            json.dump(query_json, f)
        f.close()
        
        if verbose:
            print('-----------------query in json format-----------------\n', query_json)
        
        # Generate steps using mock LLM
        start = time.time()
        steps = GPT_response(constraint_to_step_prompt + query + '\n' + 'Steps:\n', model_version, verbose)
        json_step = time.time()
        times.append(json_step - start)
        
        with open(path+'plans/' + 'steps.txt', 'w') as f:
            f.write(steps)
        f.close()
        
        steps = steps.split('\n\n')
        
        # Process each step
        for step in steps:
            if verbose:
                print('!!!!!!!!!!STEP!!!!!!!!!!\n', step, '\n')
            try:
                lines = step.split('# \n')[1]
            except:
                try:
                    lines = step.split('#\n')[1]
                except:
                    lines = step
            
            prompt = ''
            step_key = ''
            for key in step_to_code_prompts.keys():
                if key.lower() in step.lower():
                    if verbose:
                        print('!!!!!!!!!!KEY!!!!!!!!!!\n', key, '\n')
                    prompt = step_to_code_prompts[key]
                    step_key = key
                    break
            
            start = time.time()
            code = GPT_response(prompt + lines, model_version, verbose)
            step_code = time.time()
            times.append(step_code - start)
            
            code = code.replace('```python', '')
            code = code.replace('```', '')
            code = code.replace('\\_', '_')
            
            # No automatic indentation for mock code - keep it clean
            if verbose:
                print('!!!!!!!!!!CODE!!!!!!!!!!\n', code, '\n')
            codes += code + '\n'
            
            with open(path+'codes/' + f'{step_key}.txt', 'w') as f:
                f.write(code)
            f.close()
        
        # Add solver code
        with open(f'prompts/ski/solve_ski_{query_json["days"]}.txt', 'r') as f:
            codes += f.read()
        
        start = time.time()
        # Execute the code and capture the result
        local_vars = {
            's': s, 'variables': variables, 'query_json': query_json,
            'SkiResortSearch': SkiResortSearch, 'SkiSlopeSearch': SkiSlopeSearch,
            'SkiRentSearch': SkiRentSearch, 'SkiCarSearch': SkiCarSearch
        }
        exec(codes, globals(), local_vars)
        
        # Generate and save the plan
        if 's' in local_vars and local_vars['s'].check() == sat:
            ski_apis = (SkiResortSearch, SkiSlopeSearch, SkiRentSearch, SkiCarSearch)
            plan = generate_ski_plan(local_vars['s'], local_vars['variables'], query_json, ski_apis)
            success = True
            if verbose:
                print(f"\n{'='*50}")
                print("SKI PLAN GENERATO CON SUCCESSO:")
                print(f"{'='*50}")
                print(plan)
                print(f"{'='*50}\n")
            
            # Save the plan to file
            with open(path+'plans/' + 'plan.txt', 'w') as f:
                f.write(plan)
            f.close()
        else:
            if verbose:
                print("\nNO SOLUTION FOUND - Constraints cannot be satisfied")
            with open(path+'plans/' + 'no_solution.txt', 'w') as f:
                f.write("No solution found - constraints cannot be satisfied")
            f.close()
        
        exec_code = time.time()
        times.append(exec_code - start)
        
    except Exception as e:
        with open(path+'codes/' + 'codes.txt', 'w') as f:
            f.write(codes)
        f.close()
        print(f"\nERROR during Z3 code execution:")
        print(f"Error: {str(e)}")
        with open(path+'plans/' + 'error.txt', 'w') as f:
            f.write(str(e))
        f.close()
    
    # Save timing information
    with open(path+'plans/' + 'time.txt', 'w') as f:
        for line in times:
            f.write(f"{line}\n")
    
    return plan if success else None

def calculate_detailed_costs(query_json):
    """Calculate detailed costs for each service based on real data from CSV files"""
    costs = {}
    days = query_json.get('days', 3)
    people = query_json.get('people', 2)
    destination = query_json.get('destination', 'Livigno')
    
    # Get available resorts for this destination
    available_resorts = query_json.get('available_resorts', [destination])
    if not available_resorts:
        available_resorts = [destination]
    
    try:
        # Get the best resort for the skill level
        query_text = query_json.get('query', '').lower()
        if "advanced" in query_text or "expert" in query_text:
            skill_level = "advanced"
        elif "intermediate" in query_text:
            skill_level = "intermediate"
        elif "not expert" in query_text or "beginner" in query_text:
            skill_level = "beginner"
        else:
            skill_level = "intermediate"  # default
            
        best_resort_info = find_best_resort_by_skill(available_resorts, skill_level, people, query_json.get('budget', 3000), days)
        
        # Resort selection quality is already shown by find_best_resort_by_skill function
        if best_resort_info['match_quality'] == 'no_data':
            print(f"ERROR: No ski resorts found in {destination}.")
            print("Please check the destination name or try a different location.")
            # Return early to avoid further processing
            return {
                'resort': 0,
                'equipment': 0,
                'car_rental': 0,
                'accommodation': 0,
                'transportation': 0,
                'total': 0
            }, best_resort_info
        
        # Use the best resort name for API calls
        best_resort_name = best_resort_info['resort_name']
        
        # Initialize APIs to get real data
        from tools_ski.apis import SkiResorts, SkiCar, SkiRent, SkiSlopes
        
        # Get real resort data for the best resort
        ski_resorts = SkiResorts()
        resort_data = ski_resorts.run(best_resort_name)
        
        if isinstance(resort_data, pd.DataFrame) and len(resort_data) > 0:
            # Use real resort price from the best matched resort
            resort_price_per_day = best_resort_info['price_per_day']
            costs['resort'] = int(resort_price_per_day * days)
        else:
            # Fallback to hardcoded if resort not found
            costs['resort'] = 80 * days * people
        
        # Get real equipment rental costs for the destination area
        ski_rent = SkiRent()
        rent_data = ski_rent.run(destination)
        if isinstance(rent_data, pd.DataFrame) and len(rent_data) > 0:
            equipment_cost_per_day = rent_data['Price_day'].mean()
            equipment_people = query_json.get('equipment_people', people)
            if query_json.get('equipment'):
                costs['equipment'] = int(equipment_cost_per_day * days * equipment_people)
            else:
                costs['equipment'] = 0
        else:
            # Fallback
            equipment_people = query_json.get('equipment_people', people)
            costs['equipment'] = 25 * days * equipment_people if query_json.get('equipment') else 0
        
        # Get real car rental costs for the destination area
        if query_json.get('car_type') or query_json.get('access') == 'Car':
            ski_car = SkiCar()
            car_data = ski_car.run(destination)
            if isinstance(car_data, pd.DataFrame) and len(car_data) > 0:
                # Filter by car type if specified
                car_type = query_json.get('car_type')
                if car_type and car_type in car_data['Type'].values:
                    filtered_car_data = car_data[car_data['Type'] == car_type]
                    if len(filtered_car_data) > 0:
                        car_price_per_day = filtered_car_data['Price_day'].mean()
                    else:
                        car_price_per_day = car_data['Price_day'].mean()
                else:
                    car_price_per_day = car_data['Price_day'].mean()
                
                costs['car_rental'] = int(car_price_per_day * days)
            else:
                # Fallback to hardcoded
                car_base_cost = 75 if query_json.get('car_type') == 'Pick up' else 50
                costs['car_rental'] = car_base_cost * days
        else:
            costs['car_rental'] = 0
        
        # Accommodation costs (using best resort price as base)
        if 'resort' in costs and costs['resort'] > 0:
            # Use resort price as accommodation base (per person per day)
            base_accommodation = costs['resort'] // days  # Per day total
            costs['accommodation'] = int(base_accommodation * 0.75)  # 75% of resort cost for accommodation
        else:
            costs['accommodation'] = 60 * days * people
            
    except Exception as e:
        print(f"Error getting real prices, using fallback: {e}")
        # Fallback to original hardcoded logic
        costs['resort'] = 80 * days * people
        costs['equipment'] = 25 * days * query_json.get('equipment_people', people) if query_json.get('equipment') else 0
        costs['car_rental'] = 75 * days if query_json.get('car_type') == 'Pick up' else 50 * days
        costs['accommodation'] = 60 * days * people
        
        # Create fallback resort info
        best_resort_info = {
            'resort_name': destination,
            'match_quality': 'fallback',
            'total_slopes': 15,
            'longest_run': 2.5,
            'price_per_day': 80,
            'rating': 4.0
        }
    
    # Transportation costs (if not car rental)
    if query_json.get('access') == 'Train':
        costs['transportation'] = 80 * people
    elif query_json.get('access') == 'Bus':
        costs['transportation'] = 40 * people
    else:
        costs['transportation'] = 0
    
    # Total cost
    costs['total'] = sum(costs.values())
    
    # Return both costs and resort information
    return costs, best_resort_info

def get_resort_details(destination):
    """Get detailed information about a ski resort"""
    try:
        import sys
        import io
        from contextlib import redirect_stdout
        from tools_ski.apis import SkiResorts
        
        # Capture stdout to suppress loading messages
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            ski_resorts = SkiResorts()
        
        # Try to find resort by name
        resort_data = ski_resorts.run(destination)
        if isinstance(resort_data, pd.DataFrame) and len(resort_data) > 0:
            resort = resort_data.iloc[0]
            return {
                'name': resort['Resort'],
                'country': resort['Country'],
                'access': resort['Access'],
                'rating': resort['Rating'],
                'beds': resort['Beds'],
                'price_day': resort['Price_day']
            }
        else:
            # Try to find by country if direct resort search fails
            potential_countries = ['Norway', 'Italy', 'Austria', 'Switzerland', 'France']
            for country in potential_countries:
                if country.lower() in destination.lower():
                    country_data = ski_resorts.get_resort_by_country(country)
                    if isinstance(country_data, pd.DataFrame) and len(country_data) > 0:
                        resort = country_data.iloc[0]  # Get first resort in country
                        return {
                            'name': resort['Resort'],
                            'country': resort['Country'],
                            'access': resort['Access'],
                            'rating': resort['Rating'],
                            'beds': resort['Beds'],
                            'price_day': resort['Price_day']
                        }
            
            # Fallback - return default info
            return {
                'name': destination,
                'country': 'Unknown',
                'access': 'Car',
                'rating': 4.0,
                'beds': 4,
                'price_day': 80
            }
    except Exception as e:
        return None

def get_slope_details(destination, skill_level="beginner", people=2):
            'total_slopes': resort_match['total_slopes'],
            'longest_run': resort_match['longest_run'],
            'difficulty_available': resort_match['available_difficulty'],
            'recommended_difficulty': resort_match.get('required_difficulty', 'Blue'),
            'resort_name': resort_match['resort_name'],
            'match_quality': resort_match['match_quality'],
            'missing_requirements': resort_match['missing_requirements'],
            'price_per_day': resort_match['price_per_day'],
            'rating': resort_match.get('rating', 0)
        }
        
        return slope_info
        
    except Exception as e:
        print(f"Error getting slope details: {e}")
        return {
            'total_slopes': 15,
            'longest_run': 2.5,
            'difficulty_available': "Blue",
            'recommended_difficulty': "Blue",
            'resort_name': destination,
            'match_quality': 'error',
            'missing_requirements': [f'Error: {str(e)}'],
            'price_per_day': 100,
            'rating': 0
        }

def find_best_resort_by_skill(resort_candidates, skill_level="beginner", people=2, budget=3000, days=3):
    """Find the best resort from a list of candidates using comprehensive evaluation"""
    try:
        from tools_ski.apis import SkiResorts, SkiSlopes
        import pandas as pd
        
        # Map skill level to difficulty
        difficulty_mapping = {
            "beginner": "Blue",
            "not expert": "Blue", 
            "intermediate": "Red",
            "advanced": "Black",
            "expert": "Black"
        }
        
        required_difficulty = difficulty_mapping.get(skill_level.lower(), "Blue")
        
        # If resort_candidates is a string (old API), convert to list
        if isinstance(resort_candidates, str):
            resort_candidates = [resort_candidates]
        
        # Evaluate each candidate resort (silently)
        evaluated_resorts = []
        
        # Evaluate each candidate resort
        for resort_name in resort_candidates:
            # Get resort data for this specific resort (silently)
            try:
                from contextlib import redirect_stdout
                import io
                
                # Capture all output to avoid spam
                captured_output = io.StringIO()
                with redirect_stdout(captured_output):
                    ski_resorts = SkiResorts()
                    resort_data = ski_resorts.run(resort_name)
                    
                    # Get slopes data for this resort
                    ski_slopes = SkiSlopes()
                    slope_data = ski_slopes.run(resort_name)
                
                if not isinstance(resort_data, pd.DataFrame) or len(resort_data) == 0:
                    continue  # Skip this resort if no data
                
                # Filter data for this specific resort
                resort_info = resort_data[resort_data['Resort'] == resort_name]
                if len(resort_info) == 0:
                    continue
            except Exception:
                continue
            
            # Initialize evaluation metrics
            constraints_satisfied = 0
            total_constraints = 0
            missing_requirements = []
            
            # Get basic resort info
            avg_price = resort_info['Price_day'].mean()
            avg_rating = resort_info['Rating'].mean()
            
            # CONSTRAINT 1: Budget feasibility (most important)
            total_constraints += 1
            estimated_resort_cost = avg_price * days
            if estimated_resort_cost <= budget * 0.6:  # Resort should not exceed 60% of budget
                constraints_satisfied += 1
                budget_feasible = True
            else:
                budget_feasible = False
                missing_requirements.append(f"Too expensive: €{estimated_resort_cost:.0f} > €{budget*0.6:.0f}")
            
            # CONSTRAINT 2: Skill level requirement
            total_constraints += 1
            has_required_difficulty = False
            available_difficulties = []
            total_slopes = 0
            longest_run = 0
            
            if isinstance(slope_data, pd.DataFrame) and len(slope_data) > 0:
                resort_slopes = slope_data[slope_data['Resort'] == resort_name]
                if len(resort_slopes) > 0:
                    available_difficulties = resort_slopes['Difficult_Slope'].unique().tolist()
                    has_required_difficulty = required_difficulty in available_difficulties
                    total_slopes = resort_slopes['Total_Slopes'].iloc[0] if len(resort_slopes) > 0 else 0
                    longest_run = resort_slopes['Longest_Run'].max() if len(resort_slopes) > 0 else 0
                    
                    if has_required_difficulty:
                        constraints_satisfied += 1
                    else:
                        missing_requirements.append(f"No {required_difficulty} slopes")
                else:
                    missing_requirements.append("No slope data for this resort")
            else:
                missing_requirements.append("No slope data available")
                available_difficulties = ['Unknown']
            
            # CONSTRAINT 3: Accommodation capacity
            total_constraints += 1
            suitable_beds = resort_info[resort_info['Beds'] >= people]
            if len(suitable_beds) > 0:
                constraints_satisfied += 1
            else:
                missing_requirements.append(f"No accommodation for {people} people")
            
            # Calculate constraint satisfaction ratio
            constraint_ratio = constraints_satisfied / total_constraints if total_constraints > 0 else 0
            
            # Calculate comprehensive score
            score = 0
            
            # 1. Constraint satisfaction (40% of score) - Maximum 400 points
            score += constraint_ratio * 400
            
            # 2. Budget efficiency (20% of score) - Maximum 200 points
            if budget_feasible:
                # Better score for cheaper options (more budget left for other activities)
                budget_efficiency = max(0, (budget * 0.6 - estimated_resort_cost) / (budget * 0.6))
                score += budget_efficiency * 200
            
            # 3. Rating quality (20% of score) - Maximum 200 points
            rating_score = (avg_rating / 5.0) * 200
            score += rating_score
            
            # 4. Slope quality (10% of score) - Maximum 100 points
            slope_score = 0
            if total_slopes > 0:
                slope_score += min(total_slopes / 50, 0.5) * 100  # More slopes = better
            if longest_run > 0:
                slope_score += min(longest_run / 10, 0.5) * 100  # Longer runs = better
            score += slope_score
            
            # 5. Bonus points (10% of score) - Maximum 100 points
            bonus_score = 0
            if has_required_difficulty and constraint_ratio == 1.0:
                bonus_score += 50  # Perfect match bonus
            if avg_rating >= 4.5:
                bonus_score += 30  # Excellent rating bonus
            if budget_feasible and estimated_resort_cost <= budget * 0.4:
                bonus_score += 20  # Great value bonus
            score += bonus_score
            
            evaluated_resorts.append({
                'resort_name': resort_name,
                'score': score,
                'constraints_satisfied': constraints_satisfied,
                'total_constraints': total_constraints,
                'constraint_ratio': constraint_ratio,
                'budget_feasible': budget_feasible,
                'has_required_difficulty': has_required_difficulty,
                'available_difficulties': available_difficulties,
                'missing_requirements': missing_requirements,
                'total_slopes': total_slopes,
                'longest_run': longest_run,
                'price_per_day': avg_price,
                'estimated_resort_cost': estimated_resort_cost,
                'rating': avg_rating
            })
        
        if not evaluated_resorts:
            # Fallback to first resort name if no data found
            fallback_name = resort_candidates[0] if resort_candidates else "Livigno"
            print(f"❌ No suitable resorts found, using fallback: {fallback_name}")
            return {
                'resort_name': fallback_name,
                'match_quality': 'no_data',
                'available_difficulty': 'Unknown',
                'missing_requirements': ['No resort data available'],
                'total_slopes': 0,
                'longest_run': 0,
                'price_per_day': 100,
                'constraints_satisfied': 0,
                'total_constraints': 3,
                'constraint_ratio': 0.0
            }
        
        # Sort by score (best first)
        evaluated_resorts.sort(key=lambda x: x['score'], reverse=True)
        
        # Show top 3 candidates for transparency (only once)
        if len(evaluated_resorts) > 1:  # Only show ranking if multiple resorts were evaluated
            print(f"\n🏆 TOP 3 RESORT CANDIDATES ({len(evaluated_resorts)} evaluated):")
            for i, resort in enumerate(evaluated_resorts[:3]):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"  {emoji} {resort['resort_name']} - Score: {resort['score']:.1f}/1000")
            print()
        
        best_resort = evaluated_resorts[0]
        
        # Determine match quality based on comprehensive evaluation (only once)
        if best_resort['constraint_ratio'] == 1.0 and best_resort['budget_feasible']:
            match_quality = 'perfect'
            print(f"🎯 PERFECT MATCH: {best_resort['resort_name']} satisfies all requirements!")
        elif best_resort['constraint_ratio'] >= 0.67 and best_resort['budget_feasible']:
            match_quality = 'excellent'
            print(f"⭐ EXCELLENT CHOICE: {best_resort['resort_name']} meets most requirements!")
        elif best_resort['constraint_ratio'] >= 0.33 and best_resort['budget_feasible']:
            match_quality = 'good'
            print(f"👍 GOOD OPTION: {best_resort['resort_name']} is a reasonable choice.")
        elif best_resort['budget_feasible']:
            match_quality = 'budget_friendly'
            print(f"💰 BUDGET FRIENDLY: {best_resort['resort_name']} fits your budget.")
        else:
            match_quality = 'limited'
            print(f"⚠️  LIMITED OPTIONS: {best_resort['resort_name']} has some limitations.")
        
        return {
            'resort_name': best_resort['resort_name'],
            'match_quality': match_quality,
            'available_difficulty': ', '.join(best_resort['available_difficulties']),
            'required_difficulty': required_difficulty,
            'missing_requirements': best_resort['missing_requirements'],
            'total_slopes': best_resort['total_slopes'],
            'longest_run': best_resort['longest_run'],
            'price_per_day': best_resort['price_per_day'],
            'estimated_resort_cost': best_resort['estimated_resort_cost'],
            'rating': best_resort['rating'],
            'constraints_satisfied': best_resort['constraints_satisfied'],
            'total_constraints': best_resort['total_constraints'],
            'constraint_ratio': best_resort['constraint_ratio'],
            'score': best_resort['score']
        }
        
    except Exception as e:
        print(f"Error finding best resort: {e}")
        return {
            'resort_name': resort_candidates[0] if resort_candidates else "Livigno",
            'match_quality': 'error',
            'available_difficulty': 'Unknown',
            'missing_requirements': [f'Error: {str(e)}'],
            'total_slopes': 0,
            'longest_run': 0,
            'price_per_day': 100
        }

# ...existing code...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ski Planner Test with Mock LLM")
    parser.add_argument("--set_type", type=str, default="test_ski")
    parser.add_argument("--model_name", type=str, default="mock_llm")
    parser.add_argument("--use_dataset_queries", action="store_true", help="Use queries from dataset_ski")
    parser.add_argument("--max_queries", type=int, default=5, help="Maximum number of queries to test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--clean_output", action="store_true", help="Enable clean output to file")
    parser.add_argument("--query", type=str, help="Single query to test directly from command line")
    parser.add_argument("--query_file", type=str, help="File containing a single query to test")
    args = parser.parse_args()

    print(f"SKI PLANNER - Using Mock LLM (No API costs)")
    
    # Clean output preparation
    clean_results = [] if args.clean_output else None
    
    # Handle single query from command line or file
    if args.query:
        ski_queries = [args.query]
        print(f"Single query from command line")
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
                result_plan = pipeline_ski(query, args.set_type, args.model_name, number, "mock", args.verbose)
                
                # Load the generated JSON to get parameters and calculate costs
                json_path = f'output_ski/{args.set_type}/{args.model_name}/{number}/plans/query.json'
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        query_json = json.load(f)
                    
                    # Calculate detailed costs and get best resort information
                    costs, best_resort_info = calculate_detailed_costs(query_json)
                    
                    # Check if a valid resort was found
                    if costs['total'] == 0 and costs['resort'] == 0:
                        print(f"\nNo suitable ski resorts found for your request.")
                        print(f"Unable to generate a ski trip plan for {query_json.get('destination', 'N/A')}.")
                        print("Please try a different destination.")
                        continue  # Skip to next query
                    
                    # Use resort information already obtained from calculate_detailed_costs
                    selected_resort_name = best_resort_info['resort_name']
                    
                    if result_plan:
                        print(f"\n{query_json.get('days', 3)}-day ski trip solution found!")
                        print(f"Destination: {selected_resort_name}")
                        print(f"Duration: {query_json.get('days', 'N/A')} days")
                        print(f"People: {query_json.get('people', 'N/A')}")
                        print(f"Budget: €{query_json.get('budget', 'N/A')}")
                        
                        # Get detailed resort information using the selected resort
                        resort_info = get_resort_details(selected_resort_name)
                        if resort_info:
                            print(f"\nRESORT DETAILS:")
                            print(f"  Resort Name: {resort_info['name']}")
                            print(f"  Country: {resort_info['country']}")
                            print(f"  Access Method: {resort_info['access']}")
                            print(f"  Rating: {resort_info['rating']}/5.0")
                        
                        # Use slope information from best resort selection
                        print(f"\nSLOPE DETAILS:")
                        print(f"  Total Slopes: {best_resort_info['total_slopes']}")
                        print(f"  Longest Run: {best_resort_info['longest_run']} km")
                        print(f"  Available Difficulty: {best_resort_info['available_difficulty']}")
                        print(f"  Recommended for You: {best_resort_info['required_difficulty']} slopes")
                        
                        print(f"\nDETAILED COSTS:")
                        print(f"  Resort: €{costs['resort']}")
                        print(f"  Equipment: €{costs['equipment']}")
                        if costs['car_rental'] > 0:
                            car_type = query_json.get('car_type', 'Standard')
                            fuel_type = query_json.get('fuel_type', '')
                            fuel_str = f" ({fuel_type})" if fuel_type else ""
                            print(f"  Car Rental ({car_type}{fuel_str}): €{costs['car_rental']}")
                        if costs['transportation'] > 0:
                            print(f"  Transportation: €{costs['transportation']}")
                        print(f"  Accommodation: €{costs['accommodation']}")
                        print(f"  ───────────────────")
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
    
    print(f"\nProcessing completed!")
    
    if args.verbose:
        print(f"Completed processing {len(numbers)} ski samples with Mock LLM!")
