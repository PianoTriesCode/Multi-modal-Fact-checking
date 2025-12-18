"""
Agents for the Fact-Checking System
"""

import asyncio
import time
import logging
from typing import List, Dict, Union, Any
from langchain_core.schema import BaseOutputParser
from langchain_core.runnables import RunnableSequence
import re
from transformers import pipeline
from PIL import Image

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


class AudioAgent:
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initialize a Hugging Face Whisper transcriber pipeline.
        """
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
        )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes audio to text.
        """
        print(f"Transcribing: {audio_path} ...")
        result = self.transcriber(audio_path)
        return result["text"]

class ImageAgent:
    """Agent to handle image inputs and generate text descriptions."""

    def __init__(self):
        print("Loading image captioning model...")
        self.captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base"
        )
        print("ImageAgent ready.")

    def describe(self, image_path: str) -> str:
        """Generate a description of the image."""
        try:
            image = Image.open(image_path)
            result = self.captioner(image)
            caption = result[0]['generated_text']
            print(f"[ImageAgent] Caption generated: {caption}")
            return caption
        except Exception as e:
            print(f"[ImageAgent] Error describing image: {e}")
            return "Error: Could not process image."


class ClaimExtractorAgent:
    """Agent for extracting factual claims from text"""
    
    def __init__(self):
        self.chain = CLAIM_EXTRACTION_PROMPT | llm | ClaimParser()
    
    async def extract_claims(self, text: str) -> List[str]:
        """Extract claims from input text"""
        try:
            await asyncio.sleep(3)
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
            await asyncio.sleep(2)
            # Analyze with verification chain
            verdict_response = await self.chain.ainvoke({
                "claim": claim, 
                "evidence": evidence
            })
            await asyncio.sleep(2)
            
            verdict = verdict_response.content.strip()
            confidence = "high" if verdict in ["True", "False"] else "medium"
            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_snippet": evidence[:200] + "..." if len(evidence) > 200 else evidence
            }
        except Exception as e:
            logging.error(f"Verification failed for '{claim}': {e}")
            return {"claim": claim, "verdict": f"Error: {str(e)}", "confidence": "low"}
        
class RouterAgent:
    def __init__(self):
        self.audio_agent = AudioAgent()
        self.image_agent = ImageAgent()

    def route_input(self, input_data):
        if input_data.endswith((".wav", ".mp3", ".flac", ".m4a")):
            return "audio"
        elif isinstance(input_data, str) and input_data.endswith((".png",".jpg",".jpeg")):
            return "image"
        return "text"

    def handle(self, input_data):
        route = self.route_input(input_data)
        if route == "audio":
            text = self.audio_agent.transcribe(input_data)
            return {"type": "audio", "content": text}
        elif route == "image":
            caption = self.image_agent.describe(input_data)
            return {"type" : "image", "content" : caption}

        else:
            return {"type": "text", "content": input_data}
        
class FactCheckingAgentol:
    """Main fact-checking agent that orchestrates the pipeline"""
    
    def __init__(self):
        self.router = RouterAgent()
        self.claim_extractor = ClaimExtractorAgent()
        self.claim_verifier = ClaimVerifierAgent()
    
    async def fact_check(self, input_data: Union[str,bytes]) -> Dict[str, Any]:
        """Run the complete fact-checking pipeline"""
        start_time = time.time()
        #Step 0 : Route input
        route_result = self.router.handle(input_data)
        # Step 1: Extract claims
        print("Extracting claims...")
        claims = await self.claim_extractor.extract_claims(route_result["content"])
        
        if not claims:
            return {
                "status": "error",
                "message": "No claims could be extracted",
                "results": [],
                "processing_time": time.time() - start_time
            }
        
        print(f"Found {len(claims)} claims to verify")
        print("Verifying claims...")
        verification_tasks = [self.claim_verifier.verify_claim(claim) for claim in claims]
        results = await asyncio.gather(*verification_tasks, return_exceptions=True) #waits on all claims
        
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Verification task failed: {result}")
                continue
            final_results.append(result)
        
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