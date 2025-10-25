
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), "GeminiAssistant"))
# Load .env
load_dotenv()

# Validate
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env")

# Set env var
os.environ["GEMINI_API_KEY"] = gemini_key

print("🚀 Starting LiteLLM proxy server...")
print("   Config: config.yaml")
print(f"Model from .env: {os.getenv('MODEL_NAME')}")
print("   Endpoint: http://localhost:4000/v1")
print("   Models: http://localhost:4000/v1/models")

# Start LiteLLM proxy using config.yaml
os.system("litellm --config GeminiAssistant\config.yaml --port 4000")