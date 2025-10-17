import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SEARX_HOST = os.getenv("SEARX_HOST", "http://127.0.0.1:8089")

# Model settings
MODEL_NAME = "mistral-medium"
MODEL_TEMPERATURE = 0.2
MAX_TOKENS = 500

# Fact-checking settings
MAX_CLAIMS = 10
SEARCH_TIMEOUT = 10
RATE_LIMIT = 2