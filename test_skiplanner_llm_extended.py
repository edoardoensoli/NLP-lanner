import pandas as pd
import json
import os
import argparse
import re
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, model_name="gpt-4o", fallback_models=None):
        self.model_name = model_name
        self.fallback_models = fallback_models or ["Meta-Llama-3.1-405B-Instruct", "Phi-3-mini-4k-instruct"]
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError(f"GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
        self.base_url = "https://models.inference.ai.azure.com"
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def _get_token_for_model(self, model_name):
        if model_name == "openai/gpt-4.1":
            return os.getenv("GPT4_API_KEY")
        elif model_name.startswith("meta/meta-llama"):
            return os.getenv("LLAMA_API_KEY")
        elif model_name.startswith("microsoft/phi-4"):
            return os.getenv("PHI4_API_KEY")
        else:
            return None

    def _query_api(self, prompt):
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to extract information from user queries about ski resort planning."
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
            print(f"Error calling LLM API with model {self.model_name}: {e}")
            return None

    def query_with_fallback(self, prompt):
        result = self._query_api(prompt)
        if result:
            return result

        for fallback_model in self.fallback_models:
            print(f"Trying fallback model: {fallback_model}")
            self.model_name = fallback_model
            self.token = self._get_token_for_model(fallback_model)
            self.headers['Authorization'] = f'Bearer {self.token}'
            result = self._query_api(prompt)
            if result:
                return result

        print("All models failed.")
        return None

    def query_to_json_response(self, query):
        # Extended prompt to include all new parameters
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

Query: {query}

Please extract all available information and return a valid JSON object:"""
        
        return self.query_with_fallback(prompt)


def get_llm_response(query, llm):
    response = llm.query_to_json_response(query)
    return response

def parse_and_extract_json(response_text):
    # Regular expression to find a JSON object
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"```(json)?\n(.*?)\n```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"{\n.*}", response_text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        # if no json found, try to load the entire string
        try:
            parsed_json = json.loads(response_text)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"No JSON object found in the response. Error decoding JSON: {e}")
            return None

def get_resort_details(resort_name, country, resorts_df):
    """
    Retrieves the details of a specific ski resort from the dataframe.
    """
    resort_info = resorts_df[(resorts_df['Resort'] == resort_name) & (resorts_df['Country'] == country)]
    if not resort_info.empty:
        return resort_info.iloc[0].to_dict()
    return None

def get_country_resorts(country, resorts_df):
    """
    Retrieves all ski resorts in a specific country from the dataframe.
    """
    country_resorts = resorts_df[resorts_df['Country'] == country]
    return country_resorts

def get_car_options(car_type=None, fuel_type=None, car_df=None):
    """
    Retrieves car options based on type and fuel preferences.
    """
    if car_df is None:
        return None
    
    filtered_cars = car_df.copy()
    
    if car_type:
        filtered_cars = filtered_cars[filtered_cars['Type'].str.contains(car_type, case=False, na=False)]
    
    if fuel_type:
        filtered_cars = filtered_cars[filtered_cars['Fuel'].str.contains(fuel_type, case=False, na=False)]
    
    return filtered_cars

def get_equipment_options(equipment_type=None, equipment_df=None):
    """
    Retrieves ski equipment options based on type.
    """
    if equipment_df is None:
        return None
    
    filtered_equipment = equipment_df.copy()
    
    if equipment_type:
        # Match any equipment type mentioned
        if isinstance(equipment_type, str):
            equipment_types = equipment_type.split(',')
        elif isinstance(equipment_type, list):
            equipment_types = equipment_type
        else:
            equipment_types = [str(equipment_type)]
        
        mask = False
        for eq_type in equipment_types:
            eq_type_str = eq_type.strip() if isinstance(eq_type, str) else str(eq_type)
            mask |= filtered_equipment['Equipment'].str.contains(eq_type_str, case=False, na=False)
        filtered_equipment = filtered_equipment[mask]
    
    return filtered_equipment

def get_slope_options(difficulty=None, slopes_df=None):
    """
    Retrieves slope options based on difficulty level.
    """
    if slopes_df is None:
        return None
    
    filtered_slopes = slopes_df.copy()
    
    if difficulty:
        filtered_slopes = filtered_slopes[filtered_slopes['Difficult_Slope'].str.contains(difficulty, case=False, na=False)]
    
    return filtered_slopes

def find_best_resort(resorts, budget, preferences):
    """
    Finds the best resort based on user preferences and budget.
    Uses the actual CSV structure: Resort, Country, Continent, Season, Beds, Price_day, Access, Rating
    """
    best_resort = None
    max_score = -1

    for index, resort in resorts.iterrows():
        score = 0
        
        # Check if resort is within budget
        if budget and resort['Price_day'] > budget:
            continue
            
        # Preference: Rating (higher is better)
        if preferences.get('rating'):
            if resort['Rating'] >= preferences.get('rating'):
                score += 2
        else:
            # Give bonus for higher rated resorts
            score += resort['Rating'] / 5.0
            
        # Preference: Access method
        if preferences.get('access') and resort['Access'] == preferences.get('access'):
            score += 1
            
        # Preference: Number of beds
        if preferences.get('beds') and resort['Beds'] == preferences.get('beds'):
            score += 1

        if score > max_score:
            max_score = score
            best_resort = resort

    return best_resort.to_dict() if best_resort is not None else None

def calculate_total_cost(resort, num_people, num_days, car_cost=0, equipment_cost=0):
    """
    Calculates the total cost for the ski trip including resort, car, and equipment.
    """
    resort_cost = resort['Price_day'] * num_people * num_days
    total_cost = resort_cost + car_cost + equipment_cost
    return total_cost, resort_cost, car_cost, equipment_cost

def pipeline_ski_extended(query, llm, resorts_df, car_df=None, equipment_df=None, slopes_df=None):
    # Step 1: Query to JSON
    json_response = get_llm_response(query, llm)
    if not json_response:
        return "Failed to get a response from the LLM."
    
    parsed_json = parse_and_extract_json(json_response)
    if not parsed_json:
        return "Could not parse the query."

    # Debug: print parsed JSON
    print(f"DEBUG: Parsed JSON: {parsed_json}")

    # Extracting details from JSON
    country = parsed_json.get('destination')
    budget = parsed_json.get('budget')
    num_people = parsed_json.get('people')
    num_days = parsed_json.get('days')
    car_type = parsed_json.get('car_type')
    fuel_type = parsed_json.get('fuel_type')
    equipment_type = parsed_json.get('equipment')
    slope_difficulty = parsed_json.get('slope_difficulty')
    
    print(f"DEBUG: Equipment type extracted: {equipment_type}")
    print(f"DEBUG: Equipment type is None: {equipment_type is None}")
    
    preferences = {}
    if parsed_json.get('rating'):
        preferences['rating'] = parsed_json.get('rating')
    if parsed_json.get('access'):
        preferences['access'] = parsed_json.get('access')
    if parsed_json.get('beds'):
        preferences['beds'] = parsed_json.get('beds')

    # Step 2: Get all resorts in the specified country
    country_resorts = get_country_resorts(country, resorts_df)
    if country_resorts.empty:
        return f"No ski resorts found for {country}."

    # Step 3: Find the best resort based on preferences and budget
    best_resort = find_best_resort(country_resorts, budget, preferences)
    if not best_resort:
        return "No suitable resort found within the given budget and preferences."

    # Step 4: Handle car selection
    car_cost = 0
    selected_car = None
    if car_df is not None and (car_type or fuel_type):
        car_options = get_car_options(car_type, fuel_type, car_df)
        if not car_options.empty:
            # Select the cheapest car that meets criteria
            selected_car = car_options.loc[car_options['Price_day'].idxmin()]
            car_cost = selected_car['Price_day'] * num_days

    # Step 5: Handle equipment selection
    equipment_cost = 0
    selected_equipment = []
    if equipment_df is not None and equipment_type:
        print(f"DEBUG: Processing equipment: {equipment_type}")
        
        # Parse equipment from string or list
        equipment_list = []
        if isinstance(equipment_type, list):
            for item in equipment_type:
                item_str = item.lower()
                if 'skis' in item_str or 'ski' in item_str:
                    equipment_list.append('skis')
                if 'boots' in item_str or 'boot' in item_str:
                    equipment_list.append('boots')
                if 'helmet' in item_str:
                    equipment_list.append('helmet')
                if 'poles' in item_str or 'pole' in item_str:
                    equipment_list.append('poles')
        elif isinstance(equipment_type, str):
            equipment_str = equipment_type.lower()
            if 'skis' in equipment_str or 'ski' in equipment_str:
                equipment_list.append('skis')
            if 'boots' in equipment_str or 'boot' in equipment_str:
                equipment_list.append('boots')
            if 'helmet' in equipment_str:
                equipment_list.append('helmet')
            if 'poles' in equipment_str or 'pole' in equipment_str:
                equipment_list.append('poles')
        
        print(f"DEBUG: Equipment list parsed: {equipment_list}")
        
        for eq_type in equipment_list:
            # Filter equipment by country if possible
            country_equipment = equipment_df[equipment_df['Country'] == country] if country else equipment_df
            if country_equipment.empty:
                country_equipment = equipment_df  # Fall back to all equipment
            
            # Find equipment of this type
            eq_options = country_equipment[country_equipment['Equipment'].str.contains(eq_type, case=False, na=False)]
            if not eq_options.empty:
                # Select the cheapest option
                cheapest_eq = eq_options.loc[eq_options['Price_day'].idxmin()]
                selected_equipment.append(cheapest_eq.to_dict())
                equipment_cost += cheapest_eq['Price_day'] * num_people * num_days
                print(f"DEBUG: Selected {eq_type}: {cheapest_eq['Equipment']} at {cheapest_eq['Price_day']} EUR/day")

    # Step 6: Handle slope selection
    available_slopes = None
    if slopes_df is not None and slope_difficulty:
        available_slopes = get_slope_options(slope_difficulty, slopes_df)

    # Step 7: Calculate total cost
    total_cost, resort_cost, car_cost, equipment_cost = calculate_total_cost(
        best_resort, num_people, num_days, car_cost, equipment_cost
    )

    # Step 8: Generate comprehensive response
    response = f"=== SKI TRIP PLAN ===\n\n"
    response += f"**Resort:** {best_resort['Resort']} in {best_resort['Country']}\n"
    response += f"**Rating:** {best_resort['Rating']}/5.0\n"
    response += f"**Resort cost:** {best_resort['Price_day']} EUR per day\n"
    response += f"**Access:** {best_resort['Access']}\n"
    response += f"**Beds available:** {best_resort['Beds']}\n\n"

    if selected_car is not None:
        response += f"**Car:** {selected_car['Type']}\n"
        response += f"**Fuel type:** {selected_car['Fuel']}\n"
        response += f"**Car cost:** {selected_car['Price_day']} EUR per day\n\n"

    if selected_equipment:
        response += f"**Equipment:**\n"
        for equipment in selected_equipment:
            response += f"- {equipment['Equipment']}: {equipment['Price_day']} EUR per day per person\n"
        response += f"**Total equipment cost:** {equipment_cost} EUR\n\n"

    if available_slopes is not None and not available_slopes.empty:
        response += f"**Available slopes ({slope_difficulty}):**\n"
        for _, slope in available_slopes.iterrows():
            response += f"- {slope['Resort']}: {slope['Difficult_Slope']} ({slope['Longest_Run']}km)\n"
        response += "\n"

    response += f"**COST BREAKDOWN:**\n"
    response += f"- Resort: {resort_cost} EUR ({num_people} people × {num_days} days × {best_resort['Price_day']} EUR)\n"
    if car_cost > 0:
        response += f"- Car rental: {car_cost} EUR ({num_days} days × {selected_car['Price_day']} EUR)\n"
    if equipment_cost > 0:
        response += f"- Equipment rental: {equipment_cost} EUR\n"
    response += f"**TOTAL COST:** {total_cost} EUR\n"

    if budget and total_cost > budget:
        response += f"\n⚠️ **Warning:** Total cost ({total_cost} EUR) exceeds budget ({budget} EUR)"

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, 
                       default="I want to go skiing in France with my family of 4 for 7 days. Our budget is 5000 EUR. We need an SUV with diesel fuel, ski equipment including skis and boots, and prefer intermediate slopes.",
                       help="Input query for the ski planner.")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                       help="Model to use for generating responses.")
    parser.add_argument("--resorts_csv", type=str, default="dataset_ski/resorts/resorts.csv",
                       help="Path to the ski resorts CSV file.")
    parser.add_argument("--car_csv", type=str, default="dataset_ski/car/ski_car.csv",
                       help="Path to the car CSV file.")
    parser.add_argument("--equipment_csv", type=str, default="dataset_ski/rent/ski_rent.csv",
                       help="Path to the equipment CSV file.")
    parser.add_argument("--slopes_csv", type=str, default="dataset_ski/slopes/ski_slopes.csv",
                       help="Path to the slopes CSV file.")
    args = parser.parse_args()

    # Load all data
    resorts_df = pd.read_csv(args.resorts_csv)
    car_df = pd.read_csv(args.car_csv) if os.path.exists(args.car_csv) else None
    equipment_df = pd.read_csv(args.equipment_csv) if os.path.exists(args.equipment_csv) else None
    slopes_df = pd.read_csv(args.slopes_csv) if os.path.exists(args.slopes_csv) else None

    # Initialize the LLM
    try:
        llm = LLMClient(args.model)
    except ValueError as e:
        print(e)
        return

    # Run the extended pipeline
    result = pipeline_ski_extended(args.query, llm, resorts_df, car_df, equipment_df, slopes_df)
    print(result)

if __name__ == "__main__":
    main()
