import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.func import extract_before_parenthesis
from z3 import *
import numpy as np

class SkiResorts:
    def __init__(self, path="dataset_ski/resorts/resorts.csv"):
        self.path = path
        # Resort,Country,Continent,Season,Beds,Price_day,Access,Rating
        self.data = pd.read_csv(self.path).dropna()
        print("Ski Resorts loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def get_resort_by_country(self, country: str) -> DataFrame:
        """Search for ski resorts by country."""
        results = self.data[self.data["Country"] == country]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no ski resorts in {country}."
        return results

    def get_resort_by_continent(self, continent: str) -> DataFrame:
        """Search for ski resorts by continent."""
        results = self.data[self.data["Continent"] == continent]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no ski resorts in {continent}."
        return results

    def get_resort_by_beds(self, beds: int) -> DataFrame:
        """Search for ski resorts by number of beds."""
        results = self.data[self.data["Beds"] == beds]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no ski resorts with {beds} beds."
        return results

    def get_resort_by_access(self, access: str) -> DataFrame:
        """Search for ski resorts by access method."""
        results = self.data[self.data["Access"] == access]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no ski resorts accessible by {access}."
        return results

    def get_resort_by_rating(self, min_rating: float) -> DataFrame:
        """Search for ski resorts by minimum rating."""
        results = self.data[self.data["Rating"] >= min_rating]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no ski resorts with rating >= {min_rating}."
        return results

    def run(self, destination: str, query: str = None) -> DataFrame:
        """Search for ski resort accommodations by resort name with fuzzy matching and country fallback."""
        # First try exact match on resort name
        results = self.data[self.data["Resort"].str.lower() == destination.lower()]
        
        # If no exact match, try fuzzy matching (contains) on the resort name
        if len(results) == 0:
            results = self.data[self.data["Resort"].str.contains(destination, case=False, na=False)]
        
        # If still no match, try to find by country
        if len(results) == 0:
            country = self.get_country_from_destination(destination)
            if country:
                results = self.get_resort_by_country(country)

        return results.reset_index(drop=True)

    def get_country_from_destination(self, destination: str) -> Optional[str]:
        """Get the country from a destination name using a mapping."""
        country_mappings = {
            'zermatt': 'Switzerland',
            'livigno': 'Italy',
            'cortina': 'Italy',
            'val d\'isère': 'France',
            'val disere': 'France',
            'st. moritz': 'Switzerland',
            'st moritz': 'Switzerland',
            'chamonix': 'France',
            'verbier': 'Switzerland',
            'davos': 'Switzerland',
            'klosters': 'Switzerland',
            'whistler': 'Canada',
            'aspen': 'United States',
            'vail': 'United States',
            'gstaad': 'Switzerland',
            'interlaken': 'Switzerland',
            'grindelwald': 'Switzerland',
            'andermatt': 'Switzerland',
            'laax': 'Switzerland',
            'flims': 'Switzerland',
            'arosa': 'Switzerland',
            'wengen': 'Switzerland',
            'murren': 'Switzerland',
            'saas-fee': 'Switzerland',
            'les gets': 'France',
            'morzine': 'France',
            'courchevel': 'France',
            'meribel': 'France',
            'tignes': 'France',
            'la plagne': 'France',
            'les arcs': 'France',
            'alpe d\'huez': 'France',
            'avoriaz': 'France',
            'serre chevalier': 'France',
            'megève': 'France',
            'courmayeur': 'Italy',
            'sestriere': 'Italy',
            'madonna di campiglio': 'Italy',
            'val gardena': 'Italy',
            'alta badia': 'Italy',
            'plan de corones': 'Italy',
            'kronplatz': 'Italy',
            'solden': 'Austria',
            'ischgl': 'Austria',
            'st. anton': 'Austria',
            'st anton': 'Austria',
            'kitzbuhel': 'Austria',
            'lecn': 'Austria',
            'mayrhofen': 'Austria',
            'obergurgl': 'Austria',
            'saalbach': 'Austria',
            'hinterglemm': 'Austria',
            'zell am see': 'Austria',
            'kaprun': 'Austria',
            'schladming': 'Austria',
            'flachau': 'Austria',
            'wagrain': 'Austria',
            'st johann': 'Austria',
            'crans-montana': 'Switzerland',
            'switzerland': 'Switzerland',
            'italy': 'Italy',
            'france': 'France',
            'austria': 'Austria',
            'canada': 'Canada',
            'united states': 'United States'
        }
        return country_mappings.get(destination.lower())

class SkiSlopes:
    def __init__(self, path="dataset_ski/slopes/slopes.csv"):
        self.path = path
        # ID,Resort,Country,Continent,Difficult_Slope,Total_Slopes,Longest_Run
        self.data = pd.read_csv(self.path).dropna()
        print("Ski Slopes loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def get_slopes_by_difficulty(self, difficulty: str) -> DataFrame:
        """Search for slopes by difficulty (Black, Red, Blue)."""
        results = self.data[self.data["Difficult_Slope"] == difficulty]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no {difficulty} slopes available."
        return results

    def run(self, resort: str) -> DataFrame:
        """Search for slopes by resort name."""
        results = self.data[self.data["Resort"] == resort]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no slopes data for {resort}."
        return results

    def run_for_all_resorts(self, all_resorts, resorts: list) -> tuple:
        """Get slope information for all resorts."""
        results_difficulty = Array('slope_difficulty', IntSort(), IntSort())
        results_total = Array('slope_total', IntSort(), IntSort())
        results_longest = Array('slope_longest', IntSort(), IntSort())
        
        difficulty_types = ['Blue', 'Red', 'Black']  # Easy to Hard
        
        for i, resort in enumerate(resorts):
            result = self.data[self.data["Resort"] == resort]
            if len(result) != 0:
                resort_idx = all_resorts.index(resort)
                
                # Store most common difficulty (encoded as int: 0=Blue, 1=Red, 2=Black)
                most_common_difficulty = result['Difficult_Slope'].mode().iloc[0]
                difficulty_idx = difficulty_types.index(most_common_difficulty)
                results_difficulty = Store(results_difficulty, resort_idx, IntVal(difficulty_idx))
                
                # Store average total slopes
                avg_total = int(result['Total_Slopes'].mean())
                results_total = Store(results_total, resort_idx, IntVal(avg_total))
                
                # Store average longest run
                avg_longest = int(result['Longest_Run'].mean())
                results_longest = Store(results_longest, resort_idx, IntVal(avg_longest))
            else:
                resort_idx = all_resorts.index(resort)
                results_difficulty = Store(results_difficulty, resort_idx, IntVal(-1))
                results_total = Store(results_total, resort_idx, IntVal(-1))
                results_longest = Store(results_longest, resort_idx, IntVal(-1))
        
        return results_difficulty, results_total, results_longest

    def get_info(self, info_array, resort_idx):
        """Get slope information for a specific resort."""
        return Select(info_array, resort_idx)

class SkiRent:
    def __init__(self, path="dataset_ski/rent/ski_rent.csv"):
        self.path = path
        # Resort,Country,Continent,Equipment,Price_day
        self.data = pd.read_csv(self.path).dropna()
        print("Ski Equipment Rental loaded.")

    def get_equipment_types(self) -> list:
        """Returns a list of unique equipment types."""
        return self.data['Equipment'].unique().tolist()

    def get_equipment_by_type(self, type: str) -> DataFrame:
        """Search for ski equipment by type."""
        results = self.data[self.data["Equipment"] == type]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There is no {type} rental available."
        return results

    def run(self, resort: str) -> DataFrame:
        """Search for equipment rental by resort name."""
        results = self.data[self.data["Resort"] == resort]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There is no equipment rental in {resort}."
        return results

    def run_for_all_resorts(self, all_resorts, resorts: list) -> dict:
        """Get equipment rental prices for all resorts."""
        equipment_types = ['Skis', 'Boots', 'Helmet', 'Poles']
        results = {}
        
        for equipment in equipment_types:
            equipment_array = Array(f'rent_{equipment.lower()}', IntSort(), IntSort())
            
            for resort in resorts:
                result = self.data[(self.data["Resort"] == resort) & (self.data["Equipment"] == equipment)]
                resort_idx = all_resorts.index(resort)
                
                if len(result) != 0:
                    avg_price = int(result['Price_day'].mean())
                    equipment_array = Store(equipment_array, resort_idx, IntVal(avg_price))
                else:
                    equipment_array = Store(equipment_array, resort_idx, IntVal(-1))
            
            results[equipment.lower()] = equipment_array
        
        return results

    def get_equipment_price(self, equipment_arrays, resort_idx, equipment_type):
        """Get equipment rental price for a specific resort and equipment type."""
        return Select(equipment_arrays[equipment_type], resort_idx)

class SkiCar:
    def __init__(self, path="dataset_ski/car/ski_car.csv"):
        self.path = path
        # Resort,Country,Continent,Type,Fuel,Price_day
        self.data = pd.read_csv(self.path).dropna()
        print("Ski Car Rental loaded.")

    def load_db(self):
        self.data = pd.read_csv(self.path)

    def get_cars_by_type(self, car_type: str) -> DataFrame:
        """Search for cars by type (SUV, Sedan, Pick up, Cabriolet)."""
        results = self.data[self.data["Type"] == car_type]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no {car_type} cars available."
        return results

    def get_cars_by_fuel(self, fuel_type: str) -> DataFrame:
        """Search for cars by fuel type (Petrol, Diesel, Hybrid, Electric)."""
        results = self.data[self.data["Fuel"] == fuel_type]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There are no {fuel_type} cars available."
        return results

    def run(self, resort: str) -> DataFrame:
        """Search for car rental by resort name."""
        results = self.data[self.data["Resort"] == resort]
        results = results.reset_index(drop=True)
        if len(results) == 0:
            return f"There is no car rental in {resort}."
        return results

    def run_for_all_resorts(self, all_resorts, resorts: list) -> tuple:
        """Get car rental information for all resorts."""
        results_type = Array('car_type', IntSort(), IntSort())
        results_fuel = Array('car_fuel', IntSort(), IntSort())
        results_price = Array('car_price', IntSort(), IntSort())
        
        car_types = ['SUV', 'Sedan', 'Pick up', 'Cabriolet']
        fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
        
        for resort in resorts:
            result = self.data[self.data["Resort"] == resort]
            resort_idx = all_resorts.index(resort)
            
            if len(result) != 0:
                # Store most common car type (encoded as int)
                most_common_type = result['Type'].mode().iloc[0]
                type_idx = car_types.index(most_common_type)
                results_type = Store(results_type, resort_idx, IntVal(type_idx))
                
                # Store most common fuel type (encoded as int)
                most_common_fuel = result['Fuel'].mode().iloc[0]
                fuel_idx = fuel_types.index(most_common_fuel)
                results_fuel = Store(results_fuel, resort_idx, IntVal(fuel_idx))
                
                # Store average price
                avg_price = int(result['Price_day'].mean())
                results_price = Store(results_price, resort_idx, IntVal(avg_price))
            else:
                results_type = Store(results_type, resort_idx, IntVal(-1))
                results_fuel = Store(results_fuel, resort_idx, IntVal(-1))
                results_price = Store(results_price, resort_idx, IntVal(-1))
        
        return results_type, results_fuel, results_price

    def get_info(self, info_array, resort_idx):
        """Get car rental information for a specific resort."""
        return Select(info_array, resort_idx)
