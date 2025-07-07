import pandas as pd

# Analyze resorts dataset
resorts_df = pd.read_csv('dataset_ski/resorts/resorts.csv')
print('=== RESORTS DATASET ===')
print(f'Total resorts: {len(resorts_df)}')
print(f'Countries: {resorts_df["Country"].nunique()}')
print('Countries list:', sorted(resorts_df['Country'].unique()))
print(f'Average rating: {resorts_df["Rating"].mean():.2f}')
print(f'Price range: €{resorts_df["Price_day"].min():.0f} - €{resorts_df["Price_day"].max():.0f}')
print()

# Analyze slopes dataset
slopes_df = pd.read_csv('dataset_ski/slopes/ski_slopes.csv')
print('=== SLOPES DATASET ===')
print(f'Total slope records: {len(slopes_df)}')
print(f'Unique resorts with slopes: {slopes_df["Resort"].nunique()}')
print('Difficulty levels:', sorted(slopes_df['Difficult_Slope'].unique()))
print(f'Average total slopes per resort: {slopes_df["Total_Slopes"].mean():.0f}')
print(f'Longest run: {slopes_df["Longest_Run"].max():.0f} km')
print()

# Analyze car rental dataset
car_df = pd.read_csv('dataset_ski/car/ski_car.csv')
print('=== CAR RENTAL DATASET ===')
print(f'Total car rental records: {len(car_df)}')
print(f'Unique resorts: {car_df["Resort"].nunique()}')
print('Car types:', sorted(car_df['Type'].unique()))
print('Fuel types:', sorted(car_df['Fuel'].unique()))
print(f'Price range: €{car_df["Price_day"].min():.0f} - €{car_df["Price_day"].max():.0f}')
print()

# Analyze equipment rental dataset
rent_df = pd.read_csv('dataset_ski/rent/ski_rent.csv')
print('=== EQUIPMENT RENTAL DATASET ===')
print(f'Total equipment rental records: {len(rent_df)}')
print(f'Unique resorts: {rent_df["Resort"].nunique()}')
print('Equipment types:', sorted(rent_df['Equipment'].unique()))
print(f'Price range: €{rent_df["Price_day"].min():.0f} - €{rent_df["Price_day"].max():.0f}')
print()

print('=== SUMMARY ===')
print(f'Total records across all datasets: {len(resorts_df) + len(slopes_df) + len(car_df) + len(rent_df)}')
print(f'Coverage: {resorts_df["Country"].nunique()} countries, {slopes_df["Resort"].nunique()} resorts with slope data')
