"""
SKI DATASET TEST QUERIES
Realistic queries based on actual dataset information
Used for testing the ski planning framework
"""

SKI_TEST_QUERIES = [
    # Italian Alps - Premium destinations
    {
        "query": "Plan a 3-day ski trip to Livigno for 2 people with a budget of 1800 euros, departing from Milano. We need ski equipment rental and prefer car access.",
        "expected_resort": "Livigno",
        "expected_days": 3,
        "expected_people": 2,
        "expected_budget": 1800,
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_access": "car"
    },
    
    {
        "query": "Organize a 5-day ski vacation to Cortina d'Ampezzo for 4 people with a budget of 3500 euros. We are intermediate skiers and need accommodation for 4 beds.",
        "expected_resort": "Cortina d'Ampezzo",
        "expected_days": 5,
        "expected_people": 4,
        "expected_budget": 3500,
        "expected_level": "intermediate",
        "expected_beds": 4
    },
    
    {
        "query": "Plan a weekend ski trip to Val Gardena for 2 people, budget 1200 euros. We want to rent skis, boots and helmets, and prefer train access.",
        "expected_resort": "Val Gardena",
        "expected_days": 2,
        "expected_people": 2,
        "expected_budget": 1200,
        "expected_equipment": ["skis", "boots", "helmet"],
        "expected_access": "train"
    },
    
    {
        "query": "Create a 7-day ski adventure to Madonna di Campiglio for 6 people with a budget of 5000 euros. We are advanced skiers and need SUV rental for equipment transport.",
        "expected_resort": "Madonna di Campiglio",
        "expected_days": 7,
        "expected_people": 6,
        "expected_budget": 5000,
        "expected_level": "advanced",
        "expected_car_type": "SUV"
    },
    
    {
        "query": "Plan a 4-day family ski trip to Sestriere for 4 people, budget 2800 euros. We need complete ski equipment rental and prefer highly rated accommodation.",
        "expected_resort": "Sestriere",
        "expected_days": 4,
        "expected_people": 4,
        "expected_budget": 2800,
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_rating": "high"
    },
    
    # Alpine regions - Advanced skiing
    {
        "query": "Organize a 3-day ski trip to La Thuile for 2 experienced skiers with a budget of 1600 euros. We need red and black slopes, car rental preferred.",
        "expected_resort": "La Thuile",
        "expected_days": 3,
        "expected_people": 2,
        "expected_budget": 1600,
        "expected_level": "expert",
        "expected_slopes": ["red", "black"]
    },
    
    {
        "query": "Plan a 6-day ski vacation to Val Senales Glacier for 3 people, budget 3200 euros. We want to ski from October to May and need equipment rental.",
        "expected_resort": "Val Senales Glacier",
        "expected_days": 6,
        "expected_people": 3,
        "expected_budget": 3200,
        "expected_season": "October - May",
        "expected_equipment": ["skis", "boots", "helmet", "poles"]
    },
    
    {
        "query": "Create a 5-day ski trip to Courmayeur for 4 people with a budget of 4000 euros. We are intermediate skiers and need 4-bed accommodation with car access.",
        "expected_resort": "Courmayeur",
        "expected_days": 5,
        "expected_people": 4,
        "expected_budget": 4000,
        "expected_level": "intermediate",
        "expected_beds": 4,
        "expected_access": "car"
    },
    
    # Beginner-friendly options
    {
        "query": "Plan a 2-day ski weekend to Gressoney for 2 people, budget 900 euros. We need blue slopes for beginners and complete equipment rental.",
        "expected_resort": "Gressoney",
        "expected_days": 2,
        "expected_people": 2,
        "expected_budget": 900,
        "expected_level": "beginner",
        "expected_slopes": ["blue"],
        "expected_equipment": ["skis", "boots", "helmet", "poles"]
    },
    
    {
        "query": "Organize a 4-day ski trip to Valchiavenna Madesimo for 3 people, budget 2200 euros. We are beginners needing blue slopes and bus access option.",
        "expected_resort": "Valchiavenna Madesimo",
        "expected_days": 4,
        "expected_people": 3,
        "expected_budget": 2200,
        "expected_level": "beginner",
        "expected_slopes": ["blue"],
        "expected_access": "bus"
    },
    
    # International destinations
    {
        "query": "Plan a 6-day ski adventure in Hemsedal, Norway for 4 people with a budget of 4800 euros. We need long season skiing and car rental for flexibility.",
        "expected_resort": "Hemsedal",
        "expected_country": "Norway",
        "expected_days": 6,
        "expected_people": 4,
        "expected_budget": 4800,
        "expected_season": "November - May",
        "expected_access": "car"
    },
    
    {
        "query": "Create a 3-day ski trip to Golm, Austria for 2 people, budget 1500 euros. We want mix of blue and red slopes with train access preferred.",
        "expected_resort": "Golm",
        "expected_country": "Austria",
        "expected_days": 3,
        "expected_people": 2,
        "expected_budget": 1500,
        "expected_slopes": ["blue", "red"],
        "expected_access": "train"
    },
    
    {
        "query": "Plan a 5-day ski vacation to Geilosiden Geilo, Norway for 3 people with a budget of 3000 euros. We need equipment rental and hybrid car option.",
        "expected_resort": "Geilosiden Geilo",
        "expected_country": "Norway",
        "expected_days": 5,
        "expected_people": 3,
        "expected_budget": 3000,
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_car_fuel": "hybrid"
    },
    
    {
        "query": "Organize a 4-day ski trip to Voss, Norway for 2 people, budget 2000 euros. We want May skiing availability and complete equipment package.",
        "expected_resort": "Voss",
        "expected_country": "Norway",
        "expected_days": 4,
        "expected_people": 2,
        "expected_budget": 2000,
        "expected_season": "November - May",
        "expected_equipment": ["skis", "boots", "helmet", "poles"]
    },
    
    {
        "query": "Plan a 7-day ski holiday to Red Mountain Resort in Canada for 4 people with a budget of 6000 euros. We need extensive slopes and electric car rental.",
        "expected_resort": "Red Mountain Resort-Rossland",
        "expected_country": "Canada",
        "expected_days": 7,
        "expected_people": 4,
        "expected_budget": 6000,
        "expected_slopes": "extensive",
        "expected_car_fuel": "electric"
    },
    
    # Complex multi-constraint queries
    {
        "query": "Plan a 5-day family ski vacation to Livigno for 4 people (2 adults, 2 kids) with a budget of 3800 euros. We need beginner-friendly blue slopes, complete equipment rental for all, and 4-bed accommodation with car access.",
        "expected_resort": "Livigno",
        "expected_days": 5,
        "expected_people": 4,
        "expected_budget": 3800,
        "expected_level": "beginner",
        "expected_slopes": ["blue"],
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_beds": 4,
        "expected_access": "car",
        "expected_group_type": "family"
    },
    
    {
        "query": "Organize a 4-day ski trip to Cortina d'Ampezzo for 3 intermediate skiers with a budget of 2800 euros. We want a mix of blue and black slopes, equipment rental, and prefer accommodation with rating above 3.5.",
        "expected_resort": "Cortina d'Ampezzo",
        "expected_days": 4,
        "expected_people": 3,
        "expected_budget": 2800,
        "expected_level": "intermediate",
        "expected_slopes": ["blue", "black"],
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_rating": 3.5
    },
    
    {
        "query": "Plan a 6-day luxury ski holiday to Val Senales Glacier for 2 people with a budget of 4200 euros. We want premium accommodation, complete equipment rental, and electric car for eco-friendly travel.",
        "expected_resort": "Val Senales Glacier",
        "expected_days": 6,
        "expected_people": 2,
        "expected_budget": 4200,
        "expected_accommodation": "premium",
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_car_fuel": "electric",
        "expected_category": "luxury"
    },
    
    {
        "query": "Create a 3-day budget ski trip to Madonna di Campiglio for 2 people with a maximum budget of 1300 euros. We need basic accommodation, equipment rental, and car access to the resort.",
        "expected_resort": "Madonna di Campiglio",
        "expected_days": 3,
        "expected_people": 2,
        "expected_budget": 1300,
        "expected_accommodation": "basic",
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_access": "car",
        "expected_category": "budget"
    },
    
    {
        "query": "Plan a 8-day extended ski vacation to Sestriere (Via Lattea) for 6 people with a budget of 7500 euros. We want access to the extensive 400-slope network, equipment rental for all, and multiple SUV rentals for the large group.",
        "expected_resort": "Sestriere",
        "expected_days": 8,
        "expected_people": 6,
        "expected_budget": 7500,
        "expected_slopes": "400-slope network",
        "expected_equipment": ["skis", "boots", "helmet", "poles"],
        "expected_car_type": "SUV",
        "expected_car_quantity": "multiple",
        "expected_group_size": "large"
    }
]

# Summary statistics
QUERY_STATS = {
    "total_queries": len(SKI_TEST_QUERIES),
    "countries": ["Italy", "Norway", "Austria", "Canada"],
    "resort_count": 15,
    "budget_range": (900, 7500),
    "duration_range": (2, 8),
    "people_range": (2, 6),
    "complexity_levels": ["simple", "intermediate", "complex"],
    "skill_levels": ["beginner", "intermediate", "advanced", "expert"],
    "equipment_types": ["skis", "boots", "helmet", "poles"],
    "car_types": ["SUV", "sedan", "pick_up", "cabriolet"],
    "fuel_types": ["petrol", "diesel", "hybrid", "electric"],
    "access_methods": ["car", "train", "bus"]
}

print(f"Generated {QUERY_STATS['total_queries']} test queries covering {len(QUERY_STATS['countries'])} countries and {QUERY_STATS['resort_count']} resorts")
