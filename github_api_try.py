import requests
from dotenv import load_dotenv
import os

class GitHubModelsClient:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Access the TOKEN from environment variables
        self.token = os.getenv("API_TOKEN")
        if not self.token:
            raise ValueError("API_TOKEN not found. Please ensure it is set in your .env file.")
            
        self.base_url = "https://models.inference.ai.azure.com"
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def query_text(self, prompt, model="jambda-1.5-large"):
        """
        Fa una query testuale a GitHub Models
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a brilliant assistant, specialized in software development and AI research. Your task is to provide insightful, innovative, and safe responses to user queries."
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": ""
                }

            ],
            "model": model,
            "temperature": 1.0,
            "max_tokens": 2000,
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                print(f"Errore API: {response.status_code}")
                print(f"Risposta: {response.text}")
                return None
                
        except Exception as e:
            print(f"Errore nella richiesta: {e}")
            return None

# Esempio di utilizzo
if __name__ == "__main__":
    try:
        client = GitHubModelsClient()
        
        # Query di esempio
        prompt = "Can you write a simple hello world in python?"
        print("🤖 Inviando richiesta a GitHub Models...")
        result = client.query_text(prompt, model="DeepSeek-R1")
        
        if result:
            print("\n📖 Risposta:")
            print("="*50)
            print(result)
        else:
            print("❌ Errore nella richiesta")
    except ValueError as e:
        print(f"❌ Errore di configurazione: {e}")