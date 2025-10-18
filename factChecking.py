# factchecking.py
"""
Multi-Agent Fact-Checking System (Mistral + SearxNG)
Author: Shaheer
Version: Phase 3 - Modern LangChain Integration
"""

import os
import asyncio
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_core.tools import tool
from langchain_community.utilities import SearxSearchWrapper
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
import logging
import time

# ====================================
# 1. Setup & Configuration
# ====================================

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY not found in .env file.")

MAX_CLAIMS = 10
SEARCH_TIMEOUT = 10
MAX_TOKENS = 500
RATE_LIMIT = 2

# Initialize components
llm = ChatMistralAI(
    model="mistral-medium", 
    temperature=0.2,
    max_tokens=MAX_TOKENS
)

searx = SearxSearchWrapper(
    searx_host="http://127.0.0.1:8089",
    unsecure=True,
    headers={"User-Agent": "Mozilla/5.0"},
)

# ====================================
# 2. Tool Definitions
# ====================================

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
            results = searx.run(claim)
            print(results)
            return str(results)[:1000]  # Limit context length
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
    """
    Extract factual claims from a given text. Identifies verifiable statements
    that can be checked for accuracy.
    
    Args:
        text: The input text to analyze and extract claims from
    """
    class ClaimOutputParser(BaseOutputParser):
        def parse(self, text: str) -> List[str]:
            lines = [line.strip(" .") for line in text.split("\n") if line.strip()]
            claims = []
            for line in lines:
                clean_line = re.sub(r'^[\d\.\-\*]+', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    claims.append(clean_line)
            return claims[:MAX_CLAIMS]
    
    claim_parser = ClaimOutputParser()
    
    # Sanitize input
    safe_text = re.sub(r'<.*?>', '', text)
    safe_text = re.sub(r'(https?://\S+)', '', safe_text).strip()
    
    extraction_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Extract concise, factual, and verifiable statements from the following text.
Each claim should be clear, objective, and capable of being verified.

Text: {text}

Return ONLY the claims as a numbered list, one per line. No additional commentary."""
    )
    
    try:
        # Use the LLM to extract claims
        response = llm.invoke(extraction_prompt.format(text=safe_text))
        claims = claim_parser.parse(response.content)
        return claims
    except Exception as e:
        logging.error(f"Error extracting claims: {e}")
        return []

# ====================================
# 3. Chain Definitions
# ====================================

def create_verification_chain():
    """Create a chain for verifying claims using search evidence"""
    verification_prompt = PromptTemplate(
        input_variables=["claim", "evidence"],
        template="""Based on the following search evidence, determine if the claim below is true, false, or uncertain.

    Claim: {claim}

    Search Evidence:
    {evidence}

    Analyze the evidence carefully. Consider:
    - Does the evidence directly support or contradict the claim?
    - Is the evidence from reliable sources?
    - Is there consensus or disagreement in the evidence?

    Respond with ONLY one word: True, False, or Uncertain. Do not include any explanation."""
        )
    
    return verification_prompt | llm

def create_claim_extraction_chain():
    """Create a chain for extracting claims from text"""
    extraction_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Extract factual claims from the following text. Focus on statements that can be objectively verified.

Text: {text}

Return each claim on a separate line with a number. Be concise and accurate."""
    )
    
    class ClaimParser(BaseOutputParser):
        def parse(self, text: str) -> List[str]:
            claims = []
            for line in text.split("\n"):
                line = re.sub(r'^\d+\.\s*', '', line.strip())
                if line and len(line) > 8:  # Minimum claim length
                    claims.append(line)
            return claims[:MAX_CLAIMS]
    
    return extraction_prompt | llm | ClaimParser()

# ====================================
# 4. Fact-Checking Pipeline
# ====================================

class FactCheckingPipeline:
    """Main fact-checking pipeline using modern LangChain patterns"""
    
    def __init__(self):
        self.claim_extractor = create_claim_extraction_chain()
        self.verifier = create_verification_chain()
        self.tools = [search_fact_tool, extract_claims_tool]
    
    async def extract_claims(self, text: str) -> List[str]:
        """Extract claims using the extraction chain"""
        try:
            claims = await self.claim_extractor.ainvoke({"text": text})
            logging.info(f"Extracted {len(claims)} claims")
            return claims
        except Exception as e:
            logging.error(f"Claim extraction failed: {e}")
            return []
    
    async def verify_single_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a single claim using search tool and verification chain"""
        logging.info(f"Verifying: {claim}")
        
        try:
            # Step 1: Gather evidence using search tool
            evidence = await search_fact_tool.ainvoke({"claim": claim})
            await asyncio.sleep(RATE_LIMIT)
            
            # Step 2: Analyze with verification chain
            verdict_response = await self.verifier.ainvoke({
                "claim": claim, 
                "evidence": evidence
            })
            await asyncio.sleep(RATE_LIMIT)
            
            verdict = verdict_response.content.strip()
            
            # Determine confidence
            confidence = "high" if verdict in ["True", "False"] else "medium"
            
            # Rate limiting
            
            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_snippet": evidence[:200] + "..." if len(evidence) > 200 else evidence
            }
            
        except asyncio.TimeoutError:
            return {"claim": claim, "verdict": "Search timeout", "confidence": "low"}
        except Exception as e:
            logging.error(f"Verification failed for '{claim}': {e}")
            return {"claim": claim, "verdict": f"Error: {str(e)}", "confidence": "low"}
    
    async def run_pipeline(self, input_text: str) -> Dict[str, Any]:
        """Execute the complete fact-checking pipeline"""
        print("Starting modern fact-checking pipeline...")
        start_time = time.time()
        
        # Step 1: Extract claims
        print("Step 1: Extracting claims...")
        claims = await self.extract_claims(input_text)
        
        if not claims:
            return {
                "status": "error",
                "message": "No claims could be extracted",
                "results": [],
                "processing_time": time.time() - start_time
            }
        
        print(f"Extracted {len(claims)} claims:")
        for i, claim in enumerate(claims, 1):
            print(f"  {i}. {claim}")
        
        # Step 2: Verify claims in parallel
        print("\nStep 2: Verifying claims...")
        print("-" * 70)
        
        verification_tasks = [self.verify_single_claim(claim) for claim in claims]
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Verification task failed: {result}")
                continue
            final_results.append(result)
            
            # Print individual result
            print(f"Claim: {result['claim']}")
            print(f"Verdict: {result['verdict']} (Confidence: {result['confidence']})")
            print("-" * 70)
        
        # Generate summary
        summary = self._generate_summary(final_results)
        
        return {
            "status": "success",
            "summary": summary,
            "results": final_results,
            "processing_time": time.time() - start_time,
            "claims_analyzed": len(final_results)
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from verification results"""
        verdicts = [r["verdict"] for r in results]
        
        true_count = sum(1 for v in verdicts if "True" in v)
        false_count = sum(1 for v in verdicts if "False" in v)
        uncertain_count = sum(1 for v in verdicts if "Uncertain" in v)
        error_count = len(verdicts) - true_count - false_count - uncertain_count
        
        confidence_scores = [r.get("confidence", "unknown") for r in results]
        high_confidence = sum(1 for c in confidence_scores if c == "high")
        
        return {
            "total_claims": len(results),
            "true_claims": true_count,
            "false_claims": false_count,
            "uncertain_claims": uncertain_count,
            "errors": error_count,
            "high_confidence_verdicts": high_confidence,
            "accuracy_estimate": (true_count + false_count) / len(results) if results else 0
        }

# ====================================
# 5. Example Usage & Testing
# ====================================

async def main():
    """Demonstrate the modern fact-checking pipeline"""
    
    # Initialize pipeline
    pipeline = FactCheckingPipeline()
    
    # Test cases
    sample_texts = [
        """
        The Eiffel Tower is located in Paris, France. It was built in 1889 and is 330 meters tall. 
        Napoleon Bonaparte ordered its construction, which took exactly 2 years to complete.
        The tower has 1665 steps to the top and was the tallest structure in the world until 1930.
        """,
        
        """
        Climate change is primarily caused by human activities like burning fossil fuels. 
        The Great Wall of China is visible from space with the naked eye. 
        Humans only use 10% of their brain capacity. 
        Vitamin C can prevent the common cold according to most medical studies.
        """
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}")
        
        result = await pipeline.run_pipeline(text)
        
        # Print summary
        summary = result["summary"]
        print(f"\nSUMMARY (Processing time: {result['processing_time']:.2f}s)")
        print(f"Total claims analyzed: {summary['total_claims']}")
        print(f"True: {summary['true_claims']} | False: {summary['false_claims']} |  Uncertain: {summary['uncertain_claims']}")
        print(f"Errors: {summary['errors']} | High confidence: {summary['high_confidence_verdicts']}")
        print(f"Accuracy estimate: {summary['accuracy_estimate']:.1%}")

# Tool usage examples
async def demonstrate_tools():
    """Show how to use the tools directly"""
    print("\nDirect Tool Usage Examples:")
    
    # Use extraction tool
    sample_text = "The moon landing happened in 1969. Water boils at 90 degrees Celsius."
    claims = await extract_claims_tool.ainvoke({"text": sample_text})
    print(f"Extracted claims: {claims}")
    
    # Use search tool
    evidence = await search_fact_tool.ainvoke({"claim": "Moon landing year 1969"})
    print(f"Search result snippet: {evidence[:100]}...")

if __name__ == "__main__":
    # Run the main pipeline
    asyncio.run(main())
    
    # Demonstrate tool usage
    asyncio.run(demonstrate_tools())