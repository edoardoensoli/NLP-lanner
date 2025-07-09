# Ski Resort Planning Framework

## Overview

This project extends the LLM-based formal travel planner framework from [LLM_Formal_Travel_Planner](https://github.com/yih301/LLM_Formal_Travel_Planner) to support ski resort planning domain. The original framework has been adapted to handle ski-specific requirements including resort selection, slope difficulty matching, equipment rental, and transportation planning for winter sports destinations.

## Key Differences from Original Framework

The base architecture from the LLM_Formal_Travel_Planner has been modified to support:

- **Ski Resort Domain**: Specialized for winter sports planning instead of general travel
- **Winter-Specific Constraints**: Slope difficulty, snow conditions, seasonal availability
- **Equipment Integration**: Ski gear rental as a core planning component
- **Resort vs Accommodation Distinction**: 
  - **Resort**: Ski resort location with ski pass costs included
  - **Accommodation**: Lodging/hotel costs separate from resort activities

## Environment Setup

To run the ski planning framework, you need to set up a Python virtual environment and install the required dependencies.

### Creating Virtual Environment

#### On Windows (PowerShell)
```powershell
# Navigate to the project directory
cd C:\path\to\NLP-lanner

# Create virtual environment
python -m venv nlp-lanner-env

# Activate virtual environment
.\nlp-lanner-env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### On macOS/Linux
```bash
# Navigate to the project directory
cd /path/to/NLP-lanner

# Create virtual environment
python3 -m venv nlp-lanner-env

# Activate virtual environment
source nlp-lanner-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages (specified in `requirements.txt`):

- `langchain==0.1.4` - LLM framework
- `langchain-community` - Community extensions for Langchain
- `pandas==2.0.1` - Data manipulation and analysis
- `tiktoken==0.4.0` - Tokenizer for OpenAI models
- `openai==0.27.2` - OpenAI API client
- `langchain_google_genai==0.0.4` - Google Generative AI integration
- `gradio==3.50.2` - Web interface framework
- `datasets==2.15.0` - Dataset handling library
- `z3-solver` - SMT constraint solver
- `anthropic` - Anthropic API client
- `mistralai==0.4.2` - Mistral AI API client

### Activating Environment

Always activate the virtual environment before running the ski planner:

```powershell
# Windows
.\nlp-lanner-env\Scripts\Activate.ps1

# macOS/Linux
source nlp-lanner-env/bin/activate
```

You should see `(nlp-lanner-env)` in your terminal prompt indicating the environment is active.

### Running the Ski Planner

Once the environment is activated and dependencies are installed, you can run the ski planner:

```bash
python test_skiplanner.py --query "Plan a 3-day ski trip to Cortina for 2 people with budget 1500 euros"
```

## Framework Architecture

### Core Components

```
tools_ski/
├── __init__.py
└── apis.py                 # API classes for ski dataset access

dataset_ski/
├── resorts/resorts.csv     # Resort accommodations (1,009 entries)
├── slopes/ski_slopes.csv   # Slope information (762 entries)
├── rent/ski_rent.csv       # Equipment rental (1,996 entries)
└── car/ski_car.csv         # Car rental options (1,499 entries)

prompts/ski/
├── query_to_json_ski.txt           # Query parsing for ski domain
├── constraint_to_step_ski.txt      # Step generation for ski planning
├── step_to_code_destination.txt    # Destination selection constraints
├── step_to_code_resort.txt         # Resort accommodation constraints
├── step_to_code_slopes.txt         # Slope difficulty matching
├── step_to_code_equipment.txt      # Equipment rental constraints
├── step_to_code_car.txt            # Car rental constraints
├── step_to_code_budget.txt         # Budget calculation
├── solve_ski_3.txt                 # 3-day trip solver
├── solve_ski_5.txt                 # 5-day trip solver
└── solve_ski_7.txt                 # 7-day trip solver
```

### API Classes

#### SkiResorts
Manages ski resort accommodations with the following data structure:
- Resort, Country, Continent, Season, Beds, Price_day, Access, Rating

Key methods:
- `get_resort_by_country(country)` - Filter by country
- `get_resort_by_access(access)` - Filter by access method (Train/Car/Bus)
- `get_resort_by_rating(min_rating)` - Filter by minimum rating
- `run(resort)` - Get accommodations for specific resort

#### SkiSlopes
Manages slope information and difficulty matching:
- Resort, Slope_Name, Difficulty, Length_km, Total_km_slopes, Longest_slope_km

Key methods:
- `get_slopes_by_difficulty(difficulty)` - Filter by difficulty (Blue/Red/Black)
- `run(resort)` - Get slope information for specific resort

#### SkiRent
Handles equipment rental options:
- Resort, Shop_Name, Equipment_Type, Price_day, Available_sizes

Key methods:
- `get_equipment_by_type(equipment_type)` - Filter by equipment type
- `run(resort)` - Get rental options for specific resort

#### SkiCar
Manages car rental services:
- Resort, Company, Car_Type, Fuel_Type, Price_day, Available

Key methods:
- `get_cars_by_type(car_type)` - Filter by car type (SUV/Sedan/Pick up/Cabriolet)
- `get_cars_by_fuel(fuel_type)` - Filter by fuel type (Petrol/Diesel/Hybrid/Electric)
- `run(resort)` - Get car rental options for specific resort

## Cost Structure

The framework calculates detailed costs for each service component:

### Resort Costs
- Base ski pass cost per person per day: 80 EUR
- Luxury resorts: 150 EUR per person per day
- Premium resorts: 120 EUR per person per day

### Equipment Rental
- Base cost: 25 EUR per person per day
- Supports partial equipment rental (e.g., equipment for only 2 out of 3 people)

### Car Rental
- Standard car: 50 EUR per day
- SUV: 80 EUR per day
- Pick up: 75 EUR per day
- Cabriolet: 100 EUR per day
- Electric fuel: +20 EUR per day
- Hybrid fuel: +10 EUR per day

### Accommodation
- Base cost: 60 EUR per person per day
- Luxury accommodation: 120 EUR per person per day

### Transportation
- Train: 80 EUR per person (one-time)
- Bus: 40 EUR per person (one-time)

## Usage Examples

### Command Line Interface

```bash
# Single query from command line
python test_skiplanner.py --query "Plan a 7-day ski vacation to Switzerland for 3 people with SUV rental"

# Query from file
python test_skiplanner.py --query_file query.txt

# Batch processing from dataset
python test_skiplanner.py --use_dataset_queries --max_queries 10

# Verbose output for debugging
python test_skiplanner.py --query "..." --verbose

# Clean output with cost breakdown
python test_skiplanner.py --query "..." --clean_output
```

### Sample Output

```
SKI PLANNER - Using Mock LLM (No API costs)
============================================================
QUERY 1:
Plan a luxury 5-day ski vacation to Cortina d'Ampezzo for 4 people with premium SUV diesel rental and complete equipment. Budget 4500 euros.
------------------------------------------------------------
SOLUTION FOUND
Destination: Cortina d'Ampezzo
Duration: 5 days
People: 4
Budget: 4500 EUR

DETAILED COSTS:
  Resort (ski pass): 3000 EUR
  Equipment: 500 EUR
  Car Rental (SUV Diesel): 450 EUR
  Accommodation: 2400 EUR
  ───────────────────
  TOTAL COST: 6350 EUR
  Over budget by 1850 EUR
============================================================
```

## Testing

### API Testing
```bash
python test_ski_apis.py
```
Tests all API classes and their filtering methods.

### Prompt Testing
```bash
python test_ski_prompts.py
```
Verifies that all prompt files load correctly and contain valid templates.

### Dataset Query Testing
```bash
python test_ski_dataset_queries.py
```
Tests the framework with realistic ski planning queries.

## Query Format Support

The framework supports natural language queries in multiple languages:

### English Examples
- "Plan a 7-day ski trip to Switzerland for 3 people with transportation and equipment rental for 2 people"
- "Luxury 5-day ski vacation to Austria with SUV rental and intermediate slopes"

### Supported Parameters
- Destination (resort/country)
- Duration (days)
- Group size (people)
- Budget (EUR)
- Skill level (beginner/intermediate/advanced)
- Equipment needs (partial or complete)
- Transportation preferences (car/train/bus)
- Car specifications (type, fuel)
- Accommodation requirements (beds, rating)

## Integration Notes

This ski planning framework operates independently from the original travel planner but uses the same:
- Z3 constraint solver architecture
- LLM prompt engineering patterns
- JSON query parsing structure
- Multi-step code generation approach

The framework can be integrated into the main NLP-lanner system or used as a standalone ski planning tool.
