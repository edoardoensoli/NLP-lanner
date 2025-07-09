#!/usr/bin/env python3
"""Test script for city/country mapping functionality"""

from tools_ski.apis import SkiResorts
import pandas as pd

def test_mapping_functionality():
    # Initialize the ski resorts API
    ski_resorts = SkiResorts()
    
    print('=== TESTING CITY/COUNTRY MAPPING ===')
    
    # Test 1: Search by specific city
    print('\n1. Search by City (Livigno):')
    livigno_results = ski_resorts.run(destination='Livigno', query='Find ski resorts in Livigno')
    print(f'   Found {len(livigno_results)} resorts for Livigno')
    if not livigno_results.empty:
        print(f'   Countries: {livigno_results["Country"].unique()}')
        print(f'   Resorts: {livigno_results["Resort"].unique()[:3]}')
    
    # Test 2: Search by country
    print('\n2. Search by Country (Switzerland):')
    swiss_results = ski_resorts.run(destination='Switzerland', query='Find ski resorts in Switzerland')
    print(f'   Found {len(swiss_results)} resorts for Switzerland')
    if not swiss_results.empty:
        print(f'   Sample resorts: {swiss_results["Resort"].unique()[:5]}')
    
    # Test 3: Test fallback for unknown city
    print('\n3. Search for Italian city with fallback (Milano):')
    milano_results = ski_resorts.run(destination='Milano', query='Find ski resorts near Milano')
    print(f'   Found {len(milano_results)} resorts for Milano')
    if not milano_results.empty:
        print(f'   Countries: {milano_results["Country"].unique()}')
    else:
        print('   No results - fallback may not be working')
    
    # Test 4: Check the mapping function directly
    print('\n4. Direct mapping function test:')
    test_destinations = ['Livigno', 'Zermatt', 'Switzerland', 'Italy', 'Milano', 'Random City']
    for dest in test_destinations:
        country = ski_resorts.get_country_from_destination(dest)
        print(f'   {dest} -> {country}')
    
    # Test 5: Available countries in dataset
    print('\n5. Available countries in dataset:')
    all_countries = ski_resorts.data['Country'].unique()
    print(f'   Countries: {sorted(all_countries)}')
    
    return {
        'livigno_count': len(livigno_results),
        'swiss_count': len(swiss_results),
        'milano_count': len(milano_results),
        'available_countries': sorted(all_countries)
    }

if __name__ == "__main__":
    results = test_mapping_functionality()
    print(f'\n=== SUMMARY ===')
    print(f'Livigno results: {results["livigno_count"]}')
    print(f'Switzerland results: {results["swiss_count"]}')
    print(f'Milano fallback results: {results["milano_count"]}')
    print(f'Available countries: {len(results["available_countries"])}')
