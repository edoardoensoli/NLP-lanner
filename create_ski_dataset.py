import pandas as pd
import random
import numpy as np

# Read the original resorts data
try:
    df_original = pd.read_csv('resorts.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df_original = pd.read_csv('resorts.csv', encoding='latin-1')
    except UnicodeDecodeError:
        df_original = pd.read_csv('resorts.csv', encoding='cp1252')

# Create the new resorts.csv with your requested structure
print("Creating enhanced resorts.csv...")

# Create multiple entries per resort with different bed configurations
resorts_data = []

# Access types with realistic distribution
access_types = ['Train', 'Car', 'Bus']
access_weights = [0.3, 0.6, 0.1]  # Most accessible by car

for _, row in df_original.iterrows():
    resort_name = row['Resort']
    country = row['Country']
    continent = row['Continent']
    season = row['Season']
    
    # Create 1-3 accommodation options per resort
    num_accommodations = random.randint(1, 3)
    
    for _ in range(num_accommodations):
        beds = random.randint(1, 4)
        
        # Price varies by continent and bed count
        base_price = {
            'Europe': random.randint(80, 300),
            'North America': random.randint(100, 350),
            'South America': random.randint(50, 200),
            'Oceania': random.randint(120, 400),
            'Asia': random.randint(60, 250)
        }.get(continent, random.randint(80, 300))
        
        # Adjust price by bed count
        price_day = base_price + (beds * random.randint(20, 50))
        
        access = np.random.choice(access_types, p=access_weights)
        rating = round(random.uniform(1.0, 5.0), 1)
        
        resorts_data.append({
            'Resort': resort_name,
            'Country': country,
            'Continent': continent,
            'Season': season,
            'Beds': beds,
            'Price_day': price_day,
            'Access': access,
            'Rating': rating
        })

# Create DataFrame and save
df_resorts = pd.DataFrame(resorts_data)
df_resorts.to_csv('resorts_new.csv', index=False)
print(f"Created resorts_new.csv with {len(df_resorts)} entries")

# Create Ski Slopes dataset
print("Creating ski_slopes.csv...")
slopes_data = []

for _, row in df_original.iterrows():
    resort_name = row['Resort']
    country = row['Country']
    continent = row['Continent']
    
    # Use existing slope data if available, otherwise generate realistic data
    total_slopes = row.get('Total slopes', random.randint(10, 150))
    if pd.isna(total_slopes):
        total_slopes = random.randint(10, 150)
    
    longest_run = row.get('Longest run', random.randint(2, 20))
    if pd.isna(longest_run):
        longest_run = random.randint(2, 20)
    
    # Create 1-2 entries per resort with different difficulty focuses
    num_entries = random.randint(1, 2)
    
    for _ in range(num_entries):
        difficult_slope = random.choice(['Black', 'Red', 'Blue'])
        
        slopes_data.append({
            'ID': len(slopes_data) + 1,
            'Resort': resort_name,
            'Country': country,
            'Continent': continent,
            'Difficult_Slope': difficult_slope,
            'Total_Slopes': int(total_slopes),
            'Longest_Run': int(longest_run)
        })

df_slopes = pd.DataFrame(slopes_data)
df_slopes.to_csv('ski_slopes.csv', index=False)
print(f"Created ski_slopes.csv with {len(df_slopes)} entries")

# Create Ski Rent dataset
print("Creating ski_rent.csv...")
rent_data = []

equipment_types = ['Skis', 'Boots', 'Helmet', 'Poles']
equipment_base_prices = {'Skis': 35, 'Boots': 20, 'Helmet': 15, 'Poles': 10}

for _, row in df_original.iterrows():
    resort_name = row['Resort']
    country = row['Country']
    continent = row['Continent']
    
    # Create entries for each equipment type
    for equipment in equipment_types:
        base_price = equipment_base_prices[equipment]
        
        # Price varies by continent
        continent_multiplier = {
            'Europe': random.uniform(1.2, 1.8),
            'North America': random.uniform(1.3, 2.0),
            'South America': random.uniform(0.8, 1.3),
            'Oceania': random.uniform(1.5, 2.2),
            'Asia': random.uniform(0.9, 1.5)
        }.get(continent, 1.0)
        
        price_day = int(base_price * continent_multiplier * random.uniform(0.8, 1.4))
        
        rent_data.append({
            'Resort': resort_name,
            'Country': country,
            'Continent': continent,
            'Equipment': equipment,
            'Price_day': price_day
        })

df_rent = pd.DataFrame(rent_data)
df_rent.to_csv('ski_rent.csv', index=False)
print(f"Created ski_rent.csv with {len(df_rent)} entries")

# Create Ski Car dataset
print("Creating ski_car.csv...")
car_data = []

car_types = ['SUV', 'Sedan', 'Pick up', 'Cabriolet']
fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']

# Realistic combinations - SUVs more common in ski resorts
car_fuel_combinations = [
    ('SUV', 'Petrol'), ('SUV', 'Diesel'), ('SUV', 'Hybrid'),
    ('Sedan', 'Petrol'), ('Sedan', 'Diesel'), ('Sedan', 'Hybrid'),
    ('Pick up', 'Diesel'), ('Pick up', 'Petrol'),
    ('Cabriolet', 'Petrol'), ('SUV', 'Electric'), ('Sedan', 'Electric')
]

for _, row in df_original.iterrows():
    resort_name = row['Resort']
    country = row['Country']
    continent = row['Continent']
    
    # Create 2-4 car options per resort
    num_cars = random.randint(2, 4)
    selected_combinations = random.sample(car_fuel_combinations, min(num_cars, len(car_fuel_combinations)))
    
    for car_type, fuel_type in selected_combinations:
        # Base price varies by car type
        base_prices = {'SUV': 80, 'Sedan': 60, 'Pick up': 75, 'Cabriolet': 90}
        base_price = base_prices[car_type]
        
        # Continent price adjustment
        continent_multiplier = {
            'Europe': random.uniform(1.0, 1.5),
            'North America': random.uniform(1.2, 1.8),
            'South America': random.uniform(0.7, 1.2),
            'Oceania': random.uniform(1.3, 2.0),
            'Asia': random.uniform(0.8, 1.4)
        }.get(continent, 1.0)
        
        # Electric/Hybrid premium
        fuel_multiplier = {'Electric': 1.3, 'Hybrid': 1.1, 'Petrol': 1.0, 'Diesel': 0.95}
        
        price_day = int(base_price * continent_multiplier * fuel_multiplier[fuel_type] * random.uniform(0.8, 1.3))
        
        car_data.append({
            'Resort': resort_name,
            'Country': country,
            'Continent': continent,
            'Type': car_type,
            'Fuel': fuel_type,
            'Price_day': price_day
        })

df_cars = pd.DataFrame(car_data)
df_cars.to_csv('ski_car.csv', index=False)
print(f"Created ski_car.csv with {len(df_cars)} entries")

print("\nDataset creation completed!")
print("\nFiles created:")
print("- resorts_new.csv (accommodations)")
print("- ski_slopes.csv (slope information)")
print("- ski_rent.csv (equipment rental)")
print("- ski_car.csv (car rental)")

print("\n📊 Dataset Statistics:")
print(f"Total resorts: {len(df_original)}")
print(f"Accommodation entries: {len(df_resorts)}")
print(f"Slope entries: {len(df_slopes)}")
print(f"Equipment rental entries: {len(df_rent)}")
print(f"Car rental entries: {len(df_cars)}")
