#!/usr/bin/env python3
"""
Test script to verify that the ski dataset APIs work correctly.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from tools_ski.apis import SkiResorts, SkiSlopes, SkiRent, SkiCar
    print("Ski APIs imported successfully")
except ImportError as e:
    print(f"Error importing ski APIs: {e}")
    sys.exit(1)

def test_ski_apis():
    print("\nTesting Ski APIs...")
    
    # Test SkiResorts
    print("\n--- Testing SkiResorts ---")
    try:
        ski_resorts = SkiResorts()
        print(f"SkiResorts loaded with {len(ski_resorts.data)} entries")
        
        # Test search by country
        italy_resorts = ski_resorts.get_resort_by_country("Italy")
        if isinstance(italy_resorts, str):
            print(f"Warning: {italy_resorts}")
        else:
            print(f"Found {len(italy_resorts)} resorts in Italy")
            
        # Test search by access method
        car_resorts = ski_resorts.get_resort_by_access("Car")
        if isinstance(car_resorts, str):
            print(f"Warning: {car_resorts}")
        else:
            print(f"Found {len(car_resorts)} resorts accessible by car")
            
    except Exception as e:
        print(f"SkiResorts test failed: {e}")
    
    # Test SkiSlopes
    print("\n--- Testing SkiSlopes ---")
    try:
        ski_slopes = SkiSlopes()
        print(f"SkiSlopes loaded with {len(ski_slopes.data)} entries")
        
        # Test search by difficulty
        black_slopes = ski_slopes.get_slopes_by_difficulty("Black")
        if isinstance(black_slopes, str):
            print(f"Warning: {black_slopes}")
        else:
            print(f"Found {len(black_slopes)} black slopes")
            
    except Exception as e:
        print(f"SkiSlopes test failed: {e}")
    
    # Test SkiRent
    print("\n--- Testing SkiRent ---")
    try:
        ski_rent = SkiRent()
        print(f"SkiRent loaded with {len(ski_rent.data)} entries")
        
        # Test search by equipment
        skis_rent = ski_rent.get_equipment_by_type("Skis")
        if isinstance(skis_rent, str):
            print(f"Warning: {skis_rent}")
        else:
            print(f"Found {len(skis_rent)} skis rental options")
            
    except Exception as e:
        print(f"SkiRent test failed: {e}")
    
    # Test SkiCar
    print("\n--- Testing SkiCar ---")
    try:
        ski_car = SkiCar()
        print(f"SkiCar loaded with {len(ski_car.data)} entries")
        
        # Test search by car type
        suv_cars = ski_car.get_cars_by_type("SUV")
        if isinstance(suv_cars, str):
            print(f"Warning: {suv_cars}")
        else:
            print(f"Found {len(suv_cars)} SUV rental options")
            
        # Test search by fuel type
        electric_cars = ski_car.get_cars_by_fuel("Electric")
        if isinstance(electric_cars, str):
            print(f"Warning: {electric_cars}")
        else:
            print(f"Found {len(electric_cars)} electric car options")
            
    except Exception as e:
        print(f"SkiCar test failed: {e}")
    
    print("\nSki API tests completed!")

if __name__ == "__main__":
    test_ski_apis()
