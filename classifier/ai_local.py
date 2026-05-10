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
        You are a document classification AI. Classify the file into EXACTLY ONE category:
        {', '.join(categories)}

        Filename: {filename}
        Extracted Text Snippet: {snippet}

        DISAMBIGUATION RULES (apply in order):
        1. Canadian_PR_Docs is the bucket for everything the user is collecting for a Canadian Permanent Residence application. This INCLUDES: employment verification letters (any employer, any country, including Guidewire), reference letters used as PR proof, T4 slips and pay slips when used as proof-of-employment, IMM forms, IRCC paperwork, IELTS / WES / ECA results, NOC references, LMIA, PCC (Police Clearance Certificates), ITA, retainer agreements with immigration consultants, and Express Entry profiles.
        2. Resumes_Career_Tech is ONLY for actual resumes / CVs (career-summary documents listing the person's own skills and experience), cover letters, certifications, interview prep, and portfolios. It is NOT for employment verification letters, even if they describe a role.
        3. Guidewire_PSE_Work is strictly for internal Guidewire operational artifacts: JIRA tickets, support cases, stack traces, logs, platform/system specs, customer SAML metadata, internal policies, and EA / provisioning emails. It is NOT for the user's own employment verification letters or T4s, even when the employer is Guidewire — those go to Canadian_PR_Docs.
        4. Financial_Taxes covers personal banking, credit-card statements, EMI, receipts, invoices, vouchers, and utility bills. T4s and pay slips submitted as PR proof go to Canadian_PR_Docs (rule 1) instead.
        5. Travel_Transit is for itineraries, boarding passes, e-tickets, and hotel reservations only. Business / market-research reports about a location are NOT travel.
        6. Use the filename as a strong tie-breaker, especially prefixes like 'PR_', 'Canada_', or form codes like IMM####, T4, PCC, NOC.
        7. If the snippet is mostly redacted or generic and you cannot confidently classify, return Unknown_Unsorted with a confidence below 80.

        Return ONLY valid JSON in this exact format, with no markdown formatting or backticks:
        {{"category": "Category_Name", "confidence": 95, "reason": "brief explanation grounded in the text content"}}
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