# SKI PLANNER METRICS REPORT
Generated: 2025-07-09 23:09:47
Total queries evaluated: 20

## OVERALL SUMMARY

### Z3
- Queries processed: 20
- Final Pass Rate: 60.0% (avg: 0.600)
- Delivery Rate: 75.0% (avg: 0.750)
- Hard Constraints: 0.0% (avg: 0.000)
- Commonsense Constraints: 55.0% (avg: 0.575)
- Repair Success: 85.0% (avg: 0.850)
- Optimality: 60.0% (avg: 0.570)
- Runtime Efficiency: 100.0% (avg: 1.000)
- Cost Reasonableness: 60.0% (avg: 0.590)

## DETAILED RESULTS

### Query 1: Z3
**Query:** 1. I want to go to a ski resort in Austria for 5 days. We are 3 people, we need a SUV with diesel fuel and we want to rent skis and boots for 2 people. We also are expert skiers looking for black slopes. We have a budget of 3000 euros.
**Result:** optimal
**Cost:** €2205.00
**Runtime:** 2.40s
**Resort:** - Resort: Zillertal Arena-Zell am Ziller-?Gerlos-?Ko?nigsleiten-?Hochkrimml

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2205.00 ≤ €3700.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.40s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €147.00/person/day

### Query 2: Z3
**Query:** 2. Planning a ski trip to France for 7 days. We are 2 people, need a sedan with petrol fuel and want to rent helmet and poles for both. We prefer blue slopes as beginners. Budget is 2500 euros.
**Result:** optimal
**Cost:** €2408.00
**Runtime:** 2.11s
**Resort:** - Resort: Molines-en-Queyras-?Saint-Ve?ran

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided
- Hard Constraint Micro: ✗ 0.67 - Passed 2/3 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2408.00 ≤ €3150.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.11s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €172.00/person/day

### Query 3: Z3
**Query:** 3. Looking for a ski vacation in Switzerland for 4 days. We are 4 people, need a pick up truck with hybrid fuel and want to rent skis for 3 people. We are intermediate skiers wanting red slopes. Budget is 4000 euros.
**Result:** optimal
**Cost:** €2132.00
**Runtime:** 2.17s
**Resort:** - Resort: Scuol-Motta Naluns

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✗ 0.50 - Resort '- Resort: Scuol-Motta Naluns' may not match destination 'Switzerland'
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2132.00 ≤ €3880.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.17s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €133.25/person/day

### Query 4: Z3
**Query:** 4. I want to ski in Italy for 6 days. We are 1 person, need a cabriolet with electric fuel and want to rent boots and helmet. I prefer black slopes for advanced skiing. Budget is 2000 euros.
**Result:** optimal
**Cost:** €1734.00
**Runtime:** 1.45s
**Resort:** - Resort: Cortina d'Ampezzo

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 0.80 - Good cost optimization: €1734.00 vs €1680.00 baseline
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.45s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €289.00/person/day

### Query 5: Z3
**Query:** 5. Planning a trip to Norway for 8 days. We are 5 people, need a SUV with petrol fuel and want to rent skis and poles for 4 people. We like blue slopes for easy skiing. Budget is 3500 euros.
**Result:** infeasible
**Runtime:** 2.14s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: infeasible
- Delivery Rate: ✓ 1.00 - Delivered: infeasible
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✗ 0.00 - No suggestions provided for infeasible query
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.14s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 6: Z3
**Query:** 6. Looking for skiing in Canada for 3 days. We are 2 people, need a sedan with diesel fuel and want to rent helmet for 1 person. We prefer red slopes for intermediate level. Budget is 1800 euros.
**Result:** optimal
**Cost:** €1314.00
**Runtime:** 1.96s
**Resort:** - Resort: Castle Mountain

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided
- Hard Constraint Micro: ✗ 0.67 - Passed 2/3 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €1314.00 ≤ €1350.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.96s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €219.00/person/day

### Query 7: Z3
**Query:** 7. I want to go skiing in Germany for 5 days. We are 3 people, need a pick up with electric fuel and want to rent skis, boots and helmet for all. We are beginners looking for blue slopes. Budget is 2800 euros.
**Result:** optimal
**Cost:** €2455.00
**Runtime:** 2.25s
**Resort:** - Resort: Go?tschen-Bischofswiesen

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2455.00 ≤ €3700.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.25s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €163.67/person/day

### Query 8: Z3
**Query:** 8. Planning a ski vacation in Japan for 9 days. We are 4 people, need a SUV with hybrid fuel and want to rent poles for 2 people. We are expert skiers wanting black slopes. Budget is 5000 euros.
**Result:** optimal
**Cost:** €4950.00
**Runtime:** 2.13s
**Resort:** - Resort: Gala Yuzawa-?Ishiuchi Maruyama

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €4950.00 ≤ €8730.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.13s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €137.50/person/day

### Query 9: Z3
**Query:** 9. Looking for a ski trip to Sweden for 6 days. We are 2 people, need a cabriolet with petrol fuel and want to rent boots and poles for both. We prefer red slopes. Budget is 3200 euros.
**Result:** optimal
**Cost:** €2646.00
**Runtime:** 1.76s
**Resort:** - Resort: Idre Fja?ll

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided
- Hard Constraint Micro: ✗ 0.67 - Passed 2/3 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2646.00 ≤ €2700.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.76s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €220.50/person/day

### Query 10: Z3
**Query:** 10. I want to ski in Chile for 7 days. We are 3 people, need a sedan with diesel fuel and want to rent skis and helmet for 2 people. We like blue slopes for easy skiing. Budget is 2600 euros.
**Result:** optimal
**Cost:** €2499.00
**Runtime:** 1.81s
**Resort:** - Resort: Valle Nevado

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2499.00 ≤ €5180.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.81s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €119.00/person/day

### Query 11: Z3
**Query:** 11. Planning skiing in Finland for 4 days. We are 1 person, need a pick up with electric fuel and want to rent all equipment (skis, boots, helmet, poles). I prefer black slopes. Budget is 2200 euros.
**Result:** optimal
**Cost:** €1632.00
**Runtime:** 1.68s
**Resort:** - Resort: Ruka

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 0.60 - Reasonable cost: €1632.00 vs €1120.00 baseline
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.68s
- Cost Reasonableness: ✓ 0.80 - Premium cost: €408.00/person/day

### Query 12: Z3
**Query:** 12. Looking for a ski vacation in New Zealand for 8 days. We are 6 people, need a SUV with petrol fuel and want to rent skis for 4 people. We are intermediate skiers wanting red slopes. Budget is 4500 euros.
**Result:** infeasible
**Runtime:** 2.62s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: infeasible
- Delivery Rate: ✓ 1.00 - Delivered: infeasible
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✗ 0.00 - No suggestions provided for infeasible query
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.62s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 13: Z3
**Query:** 13. I want to go skiing in Spain for 5 days. We are 2 people, need a cabriolet with hybrid fuel and want to rent boots and helmet for both. We prefer blue slopes as beginners. Budget is 2400 euros.
**Result:** optimal
**Cost:** €2120.00
**Runtime:** 1.97s
**Resort:** - Resort: Baqueira / ?Beret

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided
- Hard Constraint Micro: ✗ 0.67 - Passed 2/3 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €2120.00 ≤ €2250.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.97s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €212.00/person/day

### Query 14: Z3
**Query:** 14. Planning a trip to Argentina for 10 days. We are 4 people, need a sedan with diesel fuel and want to rent poles for 3 people. We are expert skiers looking for black slopes. Budget is 3800 euros.
**Result:** infeasible
**Runtime:** 2.39s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: infeasible
- Delivery Rate: ✓ 1.00 - Delivered: infeasible
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✗ 0.00 - No suggestions provided for infeasible query
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 2.39s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 15: Z3
**Query:** 15. Looking for skiing in United States for 6 days. We are 3 people, need a pick up with electric fuel and want to rent skis and boots for all. We prefer red slopes. Budget is 3600 euros.
**Result:** optimal
**Cost:** €3042.00
**Runtime:** 1.49s
**Resort:** - Resort: Sugarloaf

**Metrics:**
- Final Pass Rate: ✓ 1.00 - Result: optimal
- Delivery Rate: ✓ 1.00 - Delivered: optimal
- Hard Constraint Pass Rate: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Hard Constraint Micro: ✗ 0.50 - Passed 2/4 constraints: budget, accommodation
- Hard Constraint Macro: ✗ 0.00 - Car rental required but not provided; Equipment rental required but not provided
- Commonsense Constraint Pass Rate: ✓ 1.00 - All commonsense constraints satisfied
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✓ 1.00 - Excellent cost optimization: €3042.00 ≤ €4440.00
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 1.49s
- Cost Reasonableness: ✓ 1.00 - Reasonable cost: €169.00/person/day

### Query 16: Z3
**Query:** 16. I want to ski in Slovenia for 4 days. We are 2 people, need a SUV with petrol fuel and want to rent helmet for 1 person. We like blue slopes for easy skiing. Budget is 2000 euros.
**Result:** failed
**Runtime:** 0.57s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: failed
- Delivery Rate: ✗ 0.00 - Delivered: failed
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 0.57s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 17: Z3
**Query:** 17. Planning a ski vacation in Poland for 7 days. We are 5 people, need a cabriolet with diesel fuel and want to rent skis and poles for 3 people. We are intermediate skiers wanting red slopes. Budget is 3400 euros.
**Result:** failed
**Runtime:** 0.52s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: failed
- Delivery Rate: ✗ 0.00 - Delivered: failed
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 0.52s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 18: Z3
**Query:** 18. Looking for a ski trip to Czech Republic for 5 days. We are 3 people, need a sedan with hybrid fuel and want to rent boots for 2 people. We prefer black slopes for advanced skiing. Budget is 2700 euros.
**Result:** failed
**Runtime:** 0.60s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: failed
- Delivery Rate: ✗ 0.00 - Delivered: failed
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 0.60s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 19: Z3
**Query:** 19. I want to go skiing in Bulgaria for 8 days. We are 4 people, need a pick up with electric fuel and want to rent helmet and poles for all. We are beginners looking for blue slopes. Budget is 3000 euros.
**Result:** failed
**Runtime:** 0.61s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: failed
- Delivery Rate: ✗ 0.00 - Delivered: failed
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 0.61s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate

### Query 20: Z3
**Query:** 20. Planning skiing in Romania for 6 days. We are 2 people, need a SUV with petrol fuel and want to rent skis for both. We prefer red slopes for intermediate level. Budget is 2500 euros.
**Result:** failed
**Runtime:** 0.56s

**Metrics:**
- Final Pass Rate: ✗ 0.00 - Result: failed
- Delivery Rate: ✗ 0.00 - Delivered: failed
- Hard Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Micro: ✗ 0.00 - No optimal solution to evaluate
- Hard Constraint Macro: ✗ 0.00 - No optimal solution to evaluate
- Commonsense Constraint Pass Rate: ✗ 0.00 - No optimal solution to evaluate
- Repair Success: ✓ 1.00 - Not applicable (feasible query)
- Optimality: ✗ 0.00 - No optimal solution with cost to evaluate
- Runtime Efficiency: ✓ 1.00 - Excellent runtime: 0.56s
- Cost Reasonableness: ✗ 0.00 - No optimal solution with cost to evaluate
