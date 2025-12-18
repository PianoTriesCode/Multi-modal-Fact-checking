import os
from dotenv import load_dotenv

load_dotenv()

# CHANGED: Replaced MISTRAL_API_KEY and SEARX_HOST with OPENAI and TAVILY keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model settings
# CHANGED: Updated model name to a GPT model
MODEL_NAME = "gpt-4o" 
MODEL_TEMPERATURE = 0.2
MAX_TOKENS = 500

# Fact-checking settings
MAX_CLAIMS = 10
SEARCH_TIMEOUT = 10
RATE_LIMIT = 2