import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "Meta-Llama-3.1-405B-Instruct",
    "Meta-Llama-3.1-70B-Instruct",
    "Phi-3-mini-4k-instruct",
    "Phi-3-medium-4k-instruct"
]

BASE_URL = "https://models.inference.ai.azure.com"

for model in MODELS:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Test message"
            }
        ],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f"Status: {response.status_code}, Response: {response.text}")
        response.raise_for_status()
        print(f"Model {model} works: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Model {model} failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Error details: {e.response.text}")
