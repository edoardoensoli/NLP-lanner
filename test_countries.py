#!/usr/bin/env python3
"""
Script per testare tutti i paesi disponibili nel dataset ski resorts
e verificare se il sistema di mapping funziona correttamente.
"""

import pandas as pd
import json
import os
from test_skiplanner import MockLLM, find_best_resort_by_skill

def test_all_countries():
    """Test tutti i paesi nel dataset per verificare il mapping"""
    
    # Leggi tutti i paesi disponibili
    df = pd.read_csv('dataset_ski/resorts/resorts.csv')
    countries = sorted(df['Country'].unique())
    
    print("="*80)
    print("TESTING ALL COUNTRIES IN SKI RESORTS DATASET")
    print("="*80)
    print(f"Total countries found: {len(countries)}")
    print("Countries:", ", ".join(countries))
    print("="*80)
    
    # Inizializza il planner
    planner = MockLLM()
    
    results = {
        'working': [],
        'not_mapped': [],
        'no_data': [],
        'errors': []
    }
    
    for country in countries:
        print(f"\nTesting: {country}")
        print("-" * 50)
        
        try:
            # Test query per questo paese
            test_query = f"i want to go to a ski resort in {country} 3 days. we are in 3 people, we need a transportation method like a pick up, we want to rent the equipment only for 3 person. we also are beginner skiers."
            
            # Prova a fare il parsing
            json_response = planner.query_to_json_response(test_query)
            
            try:
                query_json = json.loads(json_response)
            except:
                print(f"  ❌ ERROR: Failed to parse JSON response")
                results['errors'].append(country)
                continue
            
            destination = query_json.get('destination', 'Unknown')
            
            print(f"  Mapped to: {destination}")
            
            # Controlla se il mapping è corretto
            if destination == 'Livigno':  # Default fallback
                print(f"  ❌ WARNING: {country} not properly mapped, using fallback!")
                results['not_mapped'].append(country)
                continue
            
            # Test se riesce a trovare resort per questo paese
            resort_info = find_best_resort_by_skill(destination, "beginner", 3)
            
            if resort_info['match_quality'] == 'no_data':
                print(f"  ❌ ERROR: No resort data found for {destination}")
                results['no_data'].append(country)
            elif resort_info['match_quality'] == 'error':
                print(f"  ❌ ERROR: Exception occurred while processing {destination}")
                results['errors'].append(country)
            else:
                print(f"  ✅ SUCCESS: Found {resort_info['resort_name']} (Quality: {resort_info['match_quality']})")
                print(f"     Available slopes: {resort_info['available_difficulty']}")
                print(f"     Price per day: €{resort_info['price_per_day']:.0f}")
                results['working'].append(country)
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            results['errors'].append(country)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"\n✅ WORKING COUNTRIES ({len(results['working'])}):")
    for country in results['working']:
        print(f"  - {country}")
    
    print(f"\n❌ NOT MAPPED COUNTRIES ({len(results['not_mapped'])}):")
    for country in results['not_mapped']:
        print(f"  - {country}")
    
    print(f"\n❌ NO DATA COUNTRIES ({len(results['no_data'])}):")
    for country in results['no_data']:
        print(f"  - {country}")
    
    print(f"\n❌ ERROR COUNTRIES ({len(results['errors'])}):")
    for country in results['errors']:
        print(f"  - {country}")
    
    print(f"\nCOVERAGE: {len(results['working'])}/{len(countries)} countries working ({len(results['working'])/len(countries)*100:.1f}%)")
    
    # Suggerimenti per migliorare il mapping
    if results['not_mapped']:
        print(f"\n📝 SUGGESTED MAPPINGS TO ADD:")
        for country in results['not_mapped']:
            # Trova un resort di esempio per questo paese
            country_resorts = df[df['Country'] == country]['Resort'].unique()
            if len(country_resorts) > 0:
                example_resort = country_resorts[0]
                key = country.lower().replace(' ', ' ')
                print(f'  "{key}": "{example_resort}",  # Default {country} resort')
    
    return results

def test_specific_country(country_name):
    """Test un paese specifico in dettaglio"""
    print(f"DETAILED TEST FOR: {country_name}")
    print("="*60)
    
    # Verifica se il paese esiste nel dataset
    df = pd.read_csv('dataset_ski/resorts/resorts.csv')
    country_data = df[df['Country'] == country_name]
    
    if len(country_data) == 0:
        print(f"❌ Country '{country_name}' not found in dataset!")
        return
    
    print(f"Resorts in {country_name}:")
    resorts = country_data['Resort'].unique()
    for i, resort in enumerate(resorts, 1):
        print(f"  {i}. {resort}")
    
    # Test mapping
    planner = MockLLM()
    test_query = f"i want to go to a ski resort in {country_name} 3 days. we are in 3 people, we need a transportation method like a pick up, we want to rent the equipment only for 3 person. we also are beginner skiers."
    
    json_response = planner.query_to_json_response(test_query)
    
    try:
        query_json = json.loads(json_response)
        mapped_destination = query_json.get('destination', 'Unknown')
    except:
        mapped_destination = 'Parse Error'
    
    print(f"\nMapping test:")
    print(f"  Input: '{country_name}'")
    print(f"  Mapped to: '{mapped_destination}'")
    
    if mapped_destination in resorts:
        print(f"  ✅ Correct mapping!")
    else:
        print(f"  ❌ Incorrect mapping!")
        print(f"  💡 Should map to one of: {', '.join(resorts[:3])}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specifico paese
        country = " ".join(sys.argv[1:])
        test_specific_country(country)
    else:
        # Test tutti i paesi
        test_all_countries()
