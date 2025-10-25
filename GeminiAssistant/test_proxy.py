# test_proxy.py
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the model name from .env
model = os.getenv("MODEL_NAME")
if not model:
    raise ValueError("❌ MODEL not found in .env file")
print(f"   Model: {model}")

# Send a test request to the LiteLLM proxy
try:
    response = requests.post(
        "http://localhost:4000/v1/chat/completions",
        json={
            "model": model,  # Explicitly use the model defined in .env
            "messages": [
                {"role": "user", "content": "Write 'Proxy works!' if everything is connected correctly."}
            ]
        },
        timeout=60  # Wait up to 30 seconds for a response
    )

    print("Status code:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print("Response from model:")
        print(content)
    else:
        print("Error from proxy:")
        print(response.text)

except Exception as e:
    print("❌ Request failed:", e)