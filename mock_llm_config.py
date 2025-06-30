# Mock LLM Configuration for NLP-lanner
# This configuration enables testing without API costs

import os
import json
from typing import Dict, Any

class MockLLMConfig:
    """Configuration for Mock LLM responses"""
    
    def __init__(self):
        self.ski_responses = {
            "query_to_json": {
                "livigno": {
                    "org": "Milano",
                    "dest": "Livigno",
                    "days": 3,
                    "visiting_city_number": 1,
                    "date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                    "people_number": 2,
                    "local_constraint": {},
                    "budget": 1500,
                    "level": "easy"
                },
                "cortina": {
                    "org": "Roma",
                    "dest": "Cortina d'Ampezzo",
                    "days": 5,
                    "visiting_city_number": 1,
                    "date": ["2024-02-10", "2024-02-11", "2024-02-12", "2024-02-13", "2024-02-14"],
                    "people_number": 4,
                    "local_constraint": {},
                    "budget": 3000,
                    "level": "intermediate"
                },
                "valdisere": {
                    "org": "Milano",
                    "dest": "Val d'Isère",
                    "days": 7,
                    "visiting_city_number": 1,
                    "date": ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07"],
                    "people_number": 6,
                    "local_constraint": {},
                    "budget": 5000,
                    "level": "advanced"
                }
            },
            "constraint_to_step": """# Destination cities
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
        }
        
        self.travel_responses = {
            "query_to_json": {
                "default": {
                    "org": "Milano",
                    "dest": ["Roma", "Milano"],
                    "days": 3,
                    "visiting_city_number": 2,
                    "date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                    "people_number": 2,
                    "local_constraint": {},
                    "budget": 1000
                }
            }
        }
    
    def get_ski_json_response(self, query: str) -> Dict[str, Any]:
        """Get JSON response for ski queries"""
        query_lower = query.lower()
        
        if "livigno" in query_lower:
            return self.ski_responses["query_to_json"]["livigno"]
        elif "cortina" in query_lower:
            return self.ski_responses["query_to_json"]["cortina"]
        elif "val d'isère" in query_lower or "valdisere" in query_lower:
            return self.ski_responses["query_to_json"]["valdisere"]
        else:
            # Default ski response
            return {
                "org": "Milano",
                "dest": "Livigno",
                "days": 3,
                "visiting_city_number": 1,
                "date": ["2024-01-15", "2024-01-16", "2024-01-17"],
                "people_number": 2,
                "local_constraint": {},
                "budget": 1500,
                "level": "intermediate"
            }
    
    def get_ski_step_response(self) -> str:
        """Get step response for ski queries"""
        return self.ski_responses["constraint_to_step"]
    
    def get_code_response(self, step_type: str) -> str:
        """Get code response for different step types"""
        
        code_templates = {
            "destination": """if query_json['visiting_city_number'] == 1:
    pass
else:
    city = [Int(f'city_{i}') for i in range(query_json['visiting_city_number'])]
    variables['city'] = city
    for i in range(query_json['visiting_city_number']):
        s.add(city[i] >= 0)
        s.add(city[i] < 10)""",
        
            "departure": """departure_dates = [Int(f'departure_date_{i}') for i in range(query_json['days']-1)]
variables['departure_dates'] = departure_dates
for i in range(query_json['days']-1):
    s.add(departure_dates[i] >= 0)
    s.add(departure_dates[i] < len(query_json['date']))""",
        
            "transportation": """car_rental = [Bool(f'car_rental_{i}') for i in range(query_json['days']-1)]
taxi = [Bool(f'taxi_{i}') for i in range(query_json['days']-1)]
flight = [Bool(f'flight_{i}') for i in range(query_json['days']-1)]
variables['car_rental'] = car_rental
variables['taxi'] = taxi  
variables['flight'] = flight
for i in range(query_json['days']-1):
    s.add(Or(car_rental[i], taxi[i], flight[i]))
    s.add(Not(And(car_rental[i], taxi[i])))
    s.add(Not(And(car_rental[i], flight[i])))
    s.add(Not(And(taxi[i], flight[i])))""",
        
            "car": """car_rental_index = [Int(f'car_rental_index_{i}') for i in range(query_json['days']-1)]
variables['car_rental_index'] = car_rental_index
for i in range(query_json['days']-1):
    s.add(car_rental_index[i] >= 0)
    s.add(car_rental_index[i] < 5)""",
        
            "resort": """resort_index = [Int(f'resort_index_{i}') for i in range(query_json['days'])]
variables['resort_index'] = resort_index
for i in range(query_json['days']):
    s.add(resort_index[i] >= 0)
    s.add(resort_index[i] < 10)""",
        
            "slopes": """slopes_index = [Int(f'slopes_index_{i}') for i in range(query_json['days'])]
variables['slopes_index'] = slopes_index
for i in range(query_json['days']):
    s.add(slopes_index[i] >= 0)
    s.add(slopes_index[i] < 15)""",
        
            "equipment": """rent_index = [Int(f'rent_index_{i}') for i in range(query_json['people_number'])]
variables['rent_index'] = rent_index
for i in range(query_json['people_number']):
    s.add(rent_index[i] >= 0)
    s.add(rent_index[i] < 8)""",
        
            "accommodation": """accommodation_index = Int('accommodation_index')
variables['accommodation_index'] = accommodation_index
s.add(accommodation_index >= 0)
s.add(accommodation_index < 12)""",
        
            "budget": """total_cost = Int('total_cost')
variables['total_cost'] = total_cost
s.add(total_cost <= query_json['budget'])
s.add(total_cost >= 0)"""
        }
        
        # Match step type to code template
        step_lower = step_type.lower()
        
        if "destination" in step_lower:
            return code_templates["destination"]
        elif "departure" in step_lower:
            return code_templates["departure"]
        elif "transportation" in step_lower:
            return code_templates["transportation"]
        elif "car" in step_lower:
            return code_templates["car"]
        elif "resort" in step_lower:
            return code_templates["resort"]
        elif "slope" in step_lower:
            return code_templates["slopes"]
        elif "equipment" in step_lower or "rent" in step_lower:
            return code_templates["equipment"]
        elif "accommodation" in step_lower:
            return code_templates["accommodation"]
        elif "budget" in step_lower:
            return code_templates["budget"]
        else:
            return f"# Mock code for: {step_type}"

# Global instance
mock_config = MockLLMConfig()

def get_mock_response(prompt: str, response_type: str = "auto") -> str:
    """Get mock response based on prompt type"""
    
    if response_type == "json" or "JSON:" in prompt:
        # Extract query from prompt and return JSON
        query = prompt.split("JSON:")[0] if "JSON:" in prompt else prompt
        json_response = mock_config.get_ski_json_response(query)
        return json.dumps(json_response, indent=2)
    
    elif response_type == "steps" or "Steps:" in prompt:
        return mock_config.get_ski_step_response()
    
    elif response_type == "code":
        # Extract step type from prompt
        step_type = prompt
        return mock_config.get_code_response(step_type)
    
    else:
        # Auto-detect response type
        if "JSON:" in prompt:
            return get_mock_response(prompt, "json")
        elif "Steps:" in prompt:
            return get_mock_response(prompt, "steps")
        else:
            return get_mock_response(prompt, "code")

# Enable/disable mock mode
MOCK_MODE_ENABLED = True

def set_mock_mode(enabled: bool):
    """Enable or disable mock mode"""
    global MOCK_MODE_ENABLED
    MOCK_MODE_ENABLED = enabled
    if enabled:
        print("Mock LLM mode ENABLED - No API costs!")
    else:
        print("Mock LLM mode DISABLED - Using real APIs")

def is_mock_mode() -> bool:
    """Check if mock mode is enabled"""
    return MOCK_MODE_ENABLED

# Export functions for use in other modules
__all__ = ['MockLLMConfig', 'mock_config', 'get_mock_response', 'set_mock_mode', 'is_mock_mode']
