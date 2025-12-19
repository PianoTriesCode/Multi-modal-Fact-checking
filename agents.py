"""
Agents for the Fact-Checking System
"""

import asyncio
import time
import logging
from typing import List, Dict, Union, Any
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableSequence
import re
from transformers import pipeline
from PIL import Image

from tools import search_fact_tool, extract_claims_tool, llm
from prompts import CLAIM_EXTRACTION_PROMPT, VERIFICATION_PROMPT
from langchain_openai import ChatOpenAI

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

# class ImageAgent:
#     """Agent to handle image inputs and generate text descriptions."""

#     def __init__(self):
#         print("Loading image captioning model...")
#         self.captioner = pipeline(
#             "image-to-text",
#             model="Salesforce/blip-image-captioning-base"
#         )
#         print("ImageAgent ready.")

#     def describe(self, image_path: str) -> str:
#         """Generate a description of the image."""
#         try:
#             image = Image.open(image_path)
#             result = self.captioner(image)
#             caption = result[0]['generated_text']
#             print(f"[ImageAgent] Caption generated: {caption}")
#             return caption
#         except Exception as e:
#             print(f"[ImageAgent] Error describing image: {e}")
#             return "Error: Could not process image."


# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image

# class ImageAgent:
#     """Agent to handle image inputs and extract text using Hugging Face TrOCR."""

#     def __init__(self):
#         print("Loading Hugging Face OCR model (TrOCR)...")
#         # We use the 'printed' version. Use 'microsoft/trocr-base-handwritten' for handwriting.
#         self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
#         self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
#         print("ImageAgent (TrOCR) ready.")

#     def describe(self, image_path: str) -> str:
#         """Extract text from the image."""
#         try:
#             print(f"Extracting text from: {image_path} ...")
#             image = Image.open(image_path).convert("RGB")
            
#             # Preprocess the image
#             pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            
#             # Generate text
#             generated_ids = self.model.generate(pixel_values)
#             generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
#             if not generated_text.strip():
#                 print("[ImageAgent] No text found.")
#                 return ""
                
#             print(f"[ImageAgent] Text found: {generated_text}")
#             return generated_text
            
#         except Exception as e:
#             print(f"[ImageAgent] Error extracting text: {e}")
#             return "Error: Could not process image."
        
    
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import logging

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import logging

class ImageAgent:
    """
    Agent to handle image inputs using Microsoft Florence-2.
    Extracts BOTH text (OCR) and visual descriptions.
    """

    def __init__(self):
        print("Loading Florence-2 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = 'microsoft/Florence-2-base'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        print(f"ImageAgent (Florence-2) ready on {self.device}.")

    def _run_task(self, image, task_prompt):
        """Helper to run a specific task on Florence-2"""
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer

    def describe(self, image_path: str) -> str:
        """Extracts both visual description and text from the image."""
        try:
            print(f"Analyzing image: {image_path} ...")
            image = Image.open(image_path).convert("RGB")
            
            # 1. Get Visual Description
            description_result = self._run_task(image, "<MORE_DETAILED_CAPTION>")
            visual_desc = description_result.get('<MORE_DETAILED_CAPTION>', '')
            
            # 2. Get Text (OCR)
            ocr_result = self._run_task(image, "<OCR>")
            raw_text = ocr_result.get('<OCR>', '')
            
            # Combine them into a context-rich string
            final_output = f"{visual_desc}"
            
            print(f"[ImageAgent] Analysis complete.")
            return final_output
            
        except Exception as e:
            logging.error(f"[ImageAgent] Error: {e}")
            return 
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
    
    def __init__(self,model_name):
        # Make sure you are using the new VERIFICATION_PROMPT from prompts.py
        self.llm = ChatOpenAI(model_name = model_name)
        self.chain = VERIFICATION_PROMPT | self.llm
    
    async def verify_claim(self,context:str ,claim: str) -> Dict[str, Any]:
        """Verify a single claim"""
        logging.info(f"Verifying: {claim}")
        
        try:
            # Get evidence
            evidence = await search_fact_tool.ainvoke({"claim": claim})
            await asyncio.sleep(2)
            
            # Analyze
            # Note: You need to pass 'context' here if your prompt uses it. 
            # If you don't have visual context for text-only claims, pass "N/A".
            verdict_response = await self.chain.ainvoke({
                "claim": claim, 
                "evidence": evidence,
                "context": context # Or pass actual context if available
            })
            
            # --- NEW PARSING LOGIC ---
            raw_content = verdict_response.content.strip()
            
            # Default values
            verdict = "Uncertain"
            explanation = raw_content
            
            # safe extraction using keyword splitting
            if "VERDICT:" in raw_content:
                parts = raw_content.split("VERDICT:")
                # Part 1 is usually empty or garbage, Part 2 has the verdict
                after_verdict = parts[1]
                
                if "EXPLANATION:" in after_verdict:
                    v_part, e_part = after_verdict.split("EXPLANATION:")
                    verdict = v_part.strip().title() # "True" or "False"
                    explanation = e_part.strip()
                else:
                    verdict = after_verdict.strip().title()
            
            # Fallback regex if the LLM forgot the prefix but outputted "True" or "False"
            elif raw_content.lower().startswith("true"):
                verdict = "True"
            elif raw_content.lower().startswith("false"):
                verdict = "False"
            # -------------------------

            confidence = "high" if verdict in ["True", "False"] else "medium"
            
            return {
                "claim": claim,
                "verdict": verdict,   # Now cleanly "True" or "False"
                "confidence": confidence,
                "evidence_snippet": explanation[:500] # Store explanation as evidence/reasoning
            }
            
        except Exception as e:
            logging.error(f"Verification failed for '{claim}': {e}")
            return {"claim": claim, "verdict": "Error", "confidence": "low"}
        
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
    
    def __init__(self, model_name):
        self.router = RouterAgent()
        self.claim_extractor = ClaimExtractorAgent()
        self.claim_verifier = ClaimVerifierAgent(model_name)
    
# In agents.py

    async def fact_check(self, input_data: Union[str,bytes], specific_claim: str = None) -> Dict[str, Any]:
        """
        Run the pipeline.
        If specific_claim is provided (from Excel), we verify THAT claim directly using the input's context.
        """
        start_time = time.time()
        
        # Step 0: Route input (Get Description/OCR or just text)
        route_result = self.router.handle(input_data)
        context_content = route_result["content"] # This is the Image Description or Raw Text
        
        # --- CHANGED LOGIC HERE ---
        if specific_claim:
            print(f"Using specific claim from dataset: '{specific_claim[:50]}...'")
            # We treat the input content as "Context" for the verification
            claims = [specific_claim]
        else:
            # Original behavior: Extract claims automatically
            print("Extracting claims...")
            claims = await self.claim_extractor.extract_claims(context_content)
        # --------------------------
        
        if not claims:
             return {
                "status": "error",
                "message": "No claims found.",
                "results": [],
                "processing_time": time.time() - start_time
            }
        
        # Determine Context for the Prompt
        # If it was an image, the 'context_content' is the Description+OCR.
        # If it was text, the 'context_content' is the text itself (or you can set it to N/A).
        if route_result["type"] == "image":
            verification_context = context_content
        else:
            # For text-only inputs, the claim usually IS the context, 
            # but we can pass the raw input just in case.
            verification_context = "Source Text: " + str(context_content)[:500]

        print(f"Verifying {len(claims)} claim(s)...")
        
        # Pass the context to the verifier
        verification_tasks = [self.claim_verifier.verify_claim(claim, verification_context) for claim in claims]
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Verification task failed: {result}")
                continue
            final_results.append(result)
        

        
        return {
            "status": "success",
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
    
class FactCheckingAgentolOLD:
    """Main fact-checking agent that orchestrates the pipeline"""
    
    def __init__(self):
        self.router = RouterAgent()
        self.claim_verifier = ClaimVerifierAgent()
    
    async def fact_check(self, input_data: Union[str,bytes]) -> Dict[str, Any]:
        """Run the complete fact-checking pipeline"""
        start_time = time.time()
        
        # Step 0: Route input (Handles Text directly, or extracts text from Audio/Image)
        route_result = self.router.handle(input_data)
        
        # CHANGED: Instead of extracting claims, we treat the content as the claim itself.
        # We wrap it in a list to match the structure expected by the verifier loop.
        claim_text = route_result["content"]
        if len(claim_text.strip()) < 10:
             return {
                "status": "error",
                "message": f"Extracted text is too short to be a claim: '{claim_text}'",
                "results": [],
                "processing_time": time.time() - start_time
            }
        claims = [claim_text] if claim_text.strip() else []
        
        
        if not claims:
            return {
                "status": "error",
                "message": "No input provided",
                "results": [],
                "processing_time": time.time() - start_time
            }
        
        print(f"Verifying claim: {claims[0]}")
        
        verification_tasks = [self.claim_verifier.verify_claim(claim) for claim in claims]
        results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        
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
