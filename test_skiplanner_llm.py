import pandas as pd
import json
import os
import argparse
import re
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, model_name="DeepSeek-R1"):
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
                    "content": "You are a helpful assistant designed to extract information from user queries."
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

    def query_to_json_response(self, query):
        # Load the prompt template
        try:
            with open("prompts/ski/query_to_json_ski.txt", "r") as f:
                prompt_template = f.read()
            prompt = prompt_template + query
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
        return self._query_api(prompt)


def get_llm_response(query, llm):
    response = llm.query_to_json_response(query)
    return response

def parse_and_extract_json(response_text):
    # Regular expression to find a JSON object
    # It looks for a string that starts with '{' and ends with '}'
    # The regex is non-greedy (.*?) to find the first valid JSON object.
    match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"```(json)?\n(.*?)\n```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"{\n.*}", response_text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
        try:
            # Attempt to parse the extracted string as JSON
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

def calculate_total_cost(resort, num_people, num_days):
    """
    Calculates the total cost for the ski trip.
    Uses Price_day from the CSV structure.
    """
    price_per_day = resort['Price_day']
    total_cost = price_per_day * num_people * num_days
    return total_cost

def pipeline_ski(query, llm, resorts_df):
    # Step 1: Query to JSON
    json_response = get_llm_response(query, llm)
    if not json_response:
        return "Failed to get a response from the LLM."
    parsed_json = parse_and_extract_json(json_response)

    if not parsed_json:
        return "Could not parse the query."

    # Extracting details from JSON
    country = parsed_json.get('destination')  # Changed from 'country' to 'destination' 
    budget = parsed_json.get('budget')        # Changed from 'budget_eur' to 'budget'
    num_people = parsed_json.get('people')    # Changed from 'num_people' to 'people'
    num_days = parsed_json.get('days')        # Changed from 'num_days' to 'days'
    preferences = {}
    
    # Extract preferences from the parsed JSON
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

    # Step 4: Calculate total cost
    total_cost = calculate_total_cost(best_resort, num_people, num_days)

    # Step 5: Generate response
    response = f"The best ski resort for you is {best_resort['Resort']} in {best_resort['Country']}.\n"
    response += f"It has a rating of {best_resort['Rating']} and costs {best_resort['Price_day']} EUR per day.\n"
    response += f"Access is by {best_resort['Access']} and it has {best_resort['Beds']} beds.\n"
    response += f"The total estimated cost for {num_people} people for {num_days} days is {total_cost} EUR."

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="I want to go skiing in France with my family of 4 for 7 days. Our budget is 5000 EUR and we prefer easy slopes and good scenery.", help="Input query for the travel planner.")
    parser.add_argument("--model", type=str, default="jambda-1.5-large", help="Model to use for generating responses (e.g., jambda-1.5-large, DeepSeek-R1).")
    parser.add_argument("--resorts_csv", type=str, default="dataset_ski/resorts/resorts.csv", help="Path to the ski resorts CSV file.")
    args = parser.parse_args()

    # Load the ski resorts data
    resorts_df = pd.read_csv(args.resorts_csv)

    # Initialize the LLM
    try:
        llm = LLMClient(args.model)
    except ValueError as e:
        print(e)
        return

    # Run the pipeline
    result = pipeline_ski(args.query, llm, resorts_df)
    print(result)

if __name__ == "__main__":
    main()
