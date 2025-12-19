"""
Tools for the Fact-Checking System
"""

import os
import re
from typing import List
from langchain_core.tools import tool
# CHANGED: Import Tavily and OpenAI instead of Searx and Mistral
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import logging
from dotenv import load_dotenv

# Initialize components
load_dotenv()
# CHANGED: Check for OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# CHANGED: Initialize ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.2,
    max_tokens=500,
    api_key=OPENAI_API_KEY
)

# CHANGED: Initialize TavilySearchResults instead of SearxSearchWrapper
tavily = TavilySearchResults(
    max_results=5
)

@tool
def search_fact_tool(claim: str) -> str:
    """
    Search the web for factual information about a specific claim.
    Use this tool to gather evidence and context for fact-checking.
    
    Args:
        claim: The factual claim to search for evidence about
    """
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _search_with_retry(claim: str) -> str:
        try:
            # CHANGED: Use tavily.invoke
            claim = f"fact check: {claim}"
            results = tavily.invoke({"query": claim})
            print(results)
            return str(results)  # Limit context length
        except Exception as e:
            logging.error(f"Search failed for '{claim}': {e}")
            raise
    
    try:
        search_text = _search_with_retry(claim)
        if not search_text or len(search_text) < 50:
            return "Insufficient search results to verify this claim."
        return search_text
    except RetryError:
        return "Search failed after multiple attempts. Please try again later."
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def extract_claims_tool(text: str) -> List[str]:
    # ... (rest of the file remains the same) ...
    """
    Extract factual claims from a given text. Identifies verifiable statements
    that can be checked for accuracy.
    
    Args:
        text: The input text to analyze and extract claims from
    """
    # Sanitize input
    safe_text = re.sub(r'<.*?>', '', text)
    safe_text = re.sub(r'(https?://\S+)', '', safe_text).strip()
    
    extraction_prompt = """Extract concise, factual, and verifiable statements from the following text.
Each claim should be clear, objective, and capable of being verified.

Text: {text}

Return ONLY the claims as a numbered list, one per line. No additional commentary."""
    
    try:
        # Use the LLM to extract claims
        response = llm.invoke(extraction_prompt.format(text=safe_text))
        
        # Parse claims from response
        lines = [line.strip(" .") for line in response.content.split("\n") if line.strip()]
        claims = []
        for line in lines:
            clean_line = re.sub(r'^[\d\.\-\*]+', '', line).strip()
            if clean_line and len(clean_line) > 10:
                claims.append(clean_line)
        return claims[:10]  # Max 10 claims
        
    except Exception as e:
        logging.error(f"Error extracting claims: {e}")
        return []

# Export all tools
ALL_TOOLS = [search_fact_tool, extract_claims_tool]