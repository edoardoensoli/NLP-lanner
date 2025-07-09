from ski_planner_metrics import QueryParameterExtractor

extractor = QueryParameterExtractor()

test_queries = [
    'Plan a 3-day ski trip to Livigno for 2 people with a budget of 2000 euros',
    'Organize a 5-day ski vacation to Zermatt for 4 people with budget 4500 euros and car rental'
]

print("Testing parameter extraction:")
for query in test_queries:
    params = extractor.extract_parameters(query)
    print(f'\nQuery: {query}')
    print(f'  people: {params.get("people")} (type: {type(params.get("people"))})')
    print(f'  days: {params.get("days")}')
    print(f'  destination: {params.get("destination")}')
    print(f'  car_required: {params.get("car_required")}')
