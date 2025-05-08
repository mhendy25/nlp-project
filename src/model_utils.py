import requests
import json
import time

class OllamaModel:
    def __init__(self, model_name):
        """Initialize with an Ollama model name."""
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        if not self.check_model_availability():
                self.download_model()
    def generate(self, prompt, system, max_tokens=100, temperature=0.7):
        """Generate text from the model using Ollama API."""
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": self.model_name,
            "system": system,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            # response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None
            
    def check_model_availability(self):
        """Check if the model is available in Ollama."""
        url = f"{self.base_url}/tags"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            return self.model_name in available_models
        except requests.exceptions.RequestException as e:
            print(f"Error checking model availability: {e}")
            return False
            
    def download_model(self):
        """Download the model if not already available."""
        if self.check_model_availability():
            print(f"Model {self.model_name} is already available")
            return True
            
        url = f"{self.base_url}/pull"
        
        payload = {
            "name": self.model_name
        }
        
        try:
            print(f"Downloading {self.model_name}. This may take a while...")
            response = requests.post(url, json=payload, stream=True)
            
            for line in response.iter_lines():
                if line:
                    status = json.loads(line.decode('utf-8'))
                    if 'status' in status:
                        print(f"Download progress: {status['status']}")
            
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading model: {e}")
            return False

if __name__ == "__main__":
    model = OllamaModel(model_name = "llama3.2:1b")
    print(model.generate(prompt = "how are you?"))