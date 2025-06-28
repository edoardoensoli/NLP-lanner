#!/usr/bin/env python3
"""Test script per verificare il caricamento del dataset TravelPlanner"""

from datasets import load_dataset

try:
    print("Caricamento del dataset TravelPlanner...")
    ds = load_dataset('osunlp/TravelPlanner', 'validation')
    print("Dataset caricato con successo!")
    print(f"Numero di sample nel validation set: {len(ds['validation'])}")
    
    # Mostra il primo esempio
    first_query = ds['validation'][0]['query']
    print("\nPrimo esempio (primi 200 caratteri):")
    print(first_query[:200])
    print("...")
    
    # Mostra la struttura
    print("\nChiavi disponibili nel primo esempio:")
    print(list(ds['validation'][0].keys()))
    
except Exception as e:
    print(f"Errore nel caricamento del dataset: {e}")
