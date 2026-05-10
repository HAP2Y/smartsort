import json
import requests

class LocalAIClassifier:
    def __init__(self, model="qwen2.5:14b"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
        self.base_url = "http://localhost:11434"

    def is_running(self) -> tuple[bool, str]:
        """Checks if Ollama is running and the model is pulled."""
        try:
            # 1. Check if Ollama is alive
            health = requests.get(self.base_url, timeout=3)
            if health.status_code != 200:
                return False, f"Ollama server returned HTTP {health.status_code}"
                
            # 2. Check if the model is pulled
            tags = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if tags.status_code == 200:
                models = [m['name'] for m in tags.json().get('models', [])]
                # Check for model existence
                if not any(self.model in m for m in models):
                    return False, f"Model '{self.model}' not found. Open terminal and run: ollama pull {self.model}"
            
            return True, f"Ollama is running and {self.model} is loaded."
            
        except requests.exceptions.ConnectionError:
            return False, "Ollama connection refused. Is the Ollama app open?"
        except Exception as e:
            return False, f"Ollama health check failed: {str(e)}"

    def classify(self, filename: str, snippet: str, categories: list) -> tuple[str, int, str]:
        prompt = f"""
        You are a highly intelligent document classification AI. Classify the provided file into EXACTLY ONE of the following categories:
        {', '.join(categories)}
        
        Filename: {filename}
        Extracted Text Snippet: {snippet}
        
        CRITICAL CONTEXT & RULES:
        1. Resumes_Career_Tech vs Guidewire_PSE_Work: Any resume, CV, or document highlighting AI automation, Kubernetes, or Software Development Engineering skills belongs in Resumes_Career_Tech. Guidewire_PSE_Work is strictly for internal operational logs, platform support tickets, flex work agreements, and system specs.
        2. Canadian_PR_Docs vs Career: Standard employment verification letters (even if they mention Canadian or Indian offices) belong in Resumes_Career_Tech unless they are explicitly an IRCC immigration form (like an IMM form, Express Entry profile, or ITA).
        3. Prioritize text content over file names.
        
        Return ONLY valid JSON in this exact format, with no markdown formatting or backticks:
        {{"category": "Category_Name", "confidence": 95, "reason": "brief explanation based on text content"}}
        """

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(self.url, json=payload, timeout=60)
            
            if response.status_code != 200:
                return "Unknown_Unsorted", 0, f"HTTP {response.status_code}: {response.text[:40]}"
                
            raw_reply = response.json().get('response', '').strip()
            if not raw_reply:
                return "Unknown_Unsorted", 0, "Ollama returned an empty string"
            
            # Clean up potential markdown formatting
            if raw_reply.startswith("```json"):
                raw_reply = raw_reply[7:]
            if raw_reply.startswith("```"):
                raw_reply = raw_reply[3:]
            if raw_reply.endswith("```"):
                raw_reply = raw_reply[:-3]
                
            data = json.loads(raw_reply.strip())
            cat = data.get('category', 'Unknown_Unsorted')
            
            if cat not in categories:
                cat = "Unknown_Unsorted"
                
            return cat, int(data.get('confidence', 0)), data.get('reason', 'AI classification')
            
        except json.JSONDecodeError:
            return "Unknown_Unsorted", 0, "AI returned invalid JSON"
        except requests.exceptions.ReadTimeout:
            return "Unknown_Unsorted", 0, "Ollama timed out (>60s)"
        except Exception as e:
            return "Unknown_Unsorted", 0, f"Error: {type(e).__name__}"