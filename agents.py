"""
Agents for the Fact-Checking System
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from langchain.schema import BaseOutputParser
from langchain_core.runnables import RunnableSequence
import re

from tools import search_fact_tool, extract_claims_tool, llm
from prompts import CLAIM_EXTRACTION_PROMPT, VERIFICATION_PROMPT

class ClaimParser(BaseOutputParser):
    """Parse claims from LLM response"""
    def parse(self, text: str) -> List[str]:
        claims = []
        for line in text.split("\n"):
            line = re.sub(r'^\d+\.\s*', '', line.strip())
            if line and len(line) > 8:  # Minimum claim length
                claims.append(line)
        return claims[:10]  # Max 10 claims

class ClaimExtractorAgent:
    """Agent for extracting factual claims from text"""
    
    def __init__(self):
        self.chain = CLAIM_EXTRACTION_PROMPT | llm | ClaimParser()
    
    async def extract_claims(self, text: str) -> List[str]:
        """Extract claims from input text"""
        try:
            time.sleep(3)
            claims = await self.chain.ainvoke({"text": text})
            logging.info(f"Extracted {len(claims)} claims")
            return claims
        except Exception as e:
            logging.error(f"Claim extraction failed: {e}")
            return []

class ClaimVerifierAgent:
    """Agent for verifying claims using search evidence"""
    
    def __init__(self):
        self.chain = VERIFICATION_PROMPT | llm
    
    async def verify_claim(self, claim: str) -> Dict[str, Any]:
        """Verify a single claim"""
        logging.info(f"Verifying: {claim}")
        
        try:
            # Get evidence using search tool
            evidence = await search_fact_tool.ainvoke({"claim": claim})
            
            # Analyze with verification chain
            verdict_response = await self.chain.ainvoke({
                "claim": claim, 
                "evidence": evidence
            })
            
            verdict = verdict_response.content.strip()
            confidence = "high" if verdict in ["True", "False"] else "medium"
            
            # Rate limiting
            await asyncio.sleep(2)
            
            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_snippet": evidence[:200] + "..." if len(evidence) > 200 else evidence
            }
            
        except Exception as e:
            logging.error(f"Verification failed for '{claim}': {e}")
            return {"claim": claim, "verdict": f"Error: {str(e)}", "confidence": "low"}

class FactCheckingAgentol:
    """Main fact-checking agent that orchestrates the pipeline"""
    
    def __init__(self):
        self.claim_extractor = ClaimExtractorAgent()
        self.claim_verifier = ClaimVerifierAgent()
    
    async def fact_check(self, text: str) -> Dict[str, Any]:
        """Run the complete fact-checking pipeline"""
        start_time = time.time()
        
        # Step 1: Extract claims
        print("Extracting claims...")
        claims = await self.claim_extractor.extract_claims(text)
        
        if not claims:
            return {
                "status": "error",
                "message": "No claims could be extracted",
                "results": [],
                "processing_time": time.time() - start_time
            }
        
        print(f"Found {len(claims)} claims to verify")
        
        # Step 2: Verify claims in parallel
        print("Verifying claims...")
        verification_tasks = [self.claim_verifier.verify_claim(claim) for claim in claims]
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        
        # Process results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Verification task failed: {result}")
                continue
            final_results.append(result)
        
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
        """Generate summary statistics"""
        verdicts = [r["verdict"] for r in results]
        
        true_count = sum(1 for v in verdicts if "True" in v)
        false_count = sum(1 for v in verdicts if "False" in v)
        uncertain_count = sum(1 for v in verdicts if "Uncertain" in v)
        error_count = len(verdicts) - true_count - false_count - uncertain_count
        
        return {
            "total_claims": len(results),
            "true_claims": true_count,
            "false_claims": false_count,
            "uncertain_claims": uncertain_count,
            "errors": error_count,
            "accuracy_score": (true_count / len(results)) if results else 0
        }