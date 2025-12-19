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
from prompts import SEARCH_CRITIC_PROMPT, CLAIM_EXTRACTION_PROMPT, VERIFICATION_PROMPT

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

class ClaimVerifierAgentOLD:
    """Agent for verifying claims using search evidence"""
    
    def __init__(self):
        # Make sure you are using the new VERIFICATION_PROMPT from prompts.py
        self.chain = VERIFICATION_PROMPT | llm
    
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
     
     # ... (Imports remain the same, ensure 'import re' is at the top)
import re 
import json
import logging
# ...

# class SearchCriticAgent:
#     """
#     Agent that critiques individual search results for relevance and bias.
#     """
#     def __init__(self):
#         self.chain = SEARCH_CRITIC_PROMPT | llm 

#     async def critique_result(self, result: Dict[str, Any], context: str, rank: int = 0) -> Dict[str, Any]:
#         url = result.get('url', 'No URL')
#         try:
#             title = result.get('title', 'No Title')
#             snippet = result.get('content', result.get('snippet', ''))
#             full_text = result.get('raw_content', '') 
            
#             # 1. Truncate inputs to prevent Token Limit Errors
#             # Context can be huge (OCR), so we limit it to 2000 chars
#             safe_context = context[:2000] if context else "N/A"
#             safe_full_text = full_text[:2000]
            
#             # 2. Invoke LLM
#             response = await self.chain.ainvoke({
#                 "title": title,
#                 "snippet": snippet,
#                 "url": url,
#                 "full_text": safe_full_text,
#                 "rank": rank,
#                 "context": safe_context 
#             })
            
#             content = response.content.strip()
            
#             # 3. ROBUST JSON PARSING (Fixes the "Failed" issue)
#             # This regex finds the largest block starting with { and ending with }
#             # It ignores any text before or after the JSON.
#             json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
#             if json_match:
#                 clean_json = json_match.group(0)
#                 try:
#                     analysis = json.loads(clean_json)
#                 except json.JSONDecodeError:
#                     # Last resort: sometimes LLMs use single quotes instead of double
#                     # This is a hacky fix but saves many crashes
#                     import ast
#                     analysis = ast.literal_eval(clean_json)
#             else:
#                 raise ValueError(f"No JSON object found in response. Raw output: {content[:100]}...")
            
#             return {
#                 "url": url,
#                 "should_use": analysis.get("recommendation", {}).get("should_use", False),
#                 "reliability_score": analysis.get("sourcing_quality", {}).get("confidence", 0),
#                 "bias_label": analysis.get("polarity_tone", {}).get("label", "Unknown"),
#                 "key_evidence": analysis.get("relevance_assessment", {}).get("justification", "") + " " + 
#                                 str([c.get('evidence_summary') for c in analysis.get("claims_analysis", [])[:2]]),
#             }
            
#         except Exception as e:
#             # PRINT THE ERROR so you can see it in your terminal
#             print(f"⚠️ Critic Error on {url}: {str(e)}") 
#             return {"should_use": False, "error": str(e)}

class ClaimVerifierAgent:
    def __init__(self):
        self.critic = SearchCriticAgent()
        self.chain = VERIFICATION_PROMPT | llm
    
    async def verify_claim(self, claim: str, context: str = "N/A") -> Dict[str, Any]:
        logging.info(f"Verifying: {claim}")
        
        try:
            # 1. Search
            raw_results = await search_fact_tool.ainvoke({"claim": claim})
            print(f"DEBUG: Type of raw_results: {type(raw_results)}")
            print(f"DEBUG: Content of raw_results: {raw_results}")
            
            if isinstance(raw_results, str):
                try:
                    search_data = json.loads(raw_results)
                except json.JSONDecodeError:
                    try:
                        # Fallback: Handle python-style stringified lists
                        import ast
                        search_data = ast.literal_eval(raw_results)
                    except:
                        print("DEBUG: Parsing completely failed.")
                        search_data = []
            else:
                search_data = raw_results

            # 2. Critique
            print("HERE")
            critic_tasks = [
                self.critic.critique_result(res, context=context, rank=i) 
                for i, res in enumerate(search_data[:3])
            ]
            print(critic_tasks)
            critiques = await asyncio.gather(*critic_tasks)
            print(critiques)
            
            # 3. Synthesize Evidence
            valid_evidence = []
            for c in critiques:
                if "error" in c:
                    # We skip errors, but now we know WHY because of the print above
                    print("oops")
                    continue
                rec_str = "Recommended" if c.get("should_use") else "CAUTION: Low Quality/Irrelevant"
                
                evidence_block = (
                    f"SOURCE: {c.get('url')}\n"
                    f"CRITIQUE: Reliability: {c.get('reliability_score')}/100 | Bias: {c.get('bias_label')} | Status: {rec_str}\n"
                    f"EVIDENCE: {c.get('key_evidence', '')}\n"
                    f"-----------------------------------"
                )
                valid_evidence.append(evidence_block)
            
            print(valid_evidence, 'is the evidence')
            evidence_text = "\n".join(valid_evidence)
            
            if not evidence_text:
                print("Critic failed on all sources (Check '⚠️ Critic Error' logs above). Using raw snippets as fallback.")
                evidence_text = str(raw_results)

            # 4. Verify
            verdict_response = await self.chain.ainvoke({
                "claim": claim, 
                "evidence": evidence_text, 
                "context": context
            })
            
            # Parsing Logic
            raw_content = verdict_response.content.strip()
            verdict = "Uncertain"
            explanation = raw_content
            
            if "VERDICT:" in raw_content:
                parts = raw_content.split("VERDICT:")
                after_verdict = parts[1]
                if "EXPLANATION:" in after_verdict:
                    v_part, e_part = after_verdict.split("EXPLANATION:")
                    verdict = v_part.strip().title()
                    explanation = e_part.strip()
                else:
                    verdict = after_verdict.strip().title()
            elif raw_content.lower().startswith("true"):
                verdict = "True"
            elif raw_content.lower().startswith("false"):
                verdict = "False"
            
            verdict = verdict.strip("., ")
            confidence = "high" if verdict in ["True", "False"] else "medium"
            
            return {
                "claim": claim,
                "verdict": verdict,
                "confidence": confidence,
                "evidence_snippet": explanation[:500]
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
        
# agents.py (Add this new class)

import json
from prompts import SEARCH_CRITIC_PROMPT

class SearchCriticAgent:
    """
    Agent that critiques individual search results for relevance and bias.
    DEBUG VERSION with verbose logging.
    """
    def __init__(self):
        self.chain = SEARCH_CRITIC_PROMPT | llm 

    async def critique_result(self, result: Dict[str, Any], context: str, rank: int = 0) -> Dict[str, Any]:
        url = result.get('url', 'No URL')
        print(f"\n[Critic] STARTING critique for: {url}")
        
        try:
            # 1. Inspect Inputs
            title = result.get('title', 'No Title')
            snippet = result.get('content', result.get('snippet', ''))
            full_text = result.get('raw_content', '') 
            
            print(f"[Critic] Title: {title[:50]}...")
            print(f"[Critic] Snippet Length: {len(snippet)} chars")
            print(f"[Critic] Full Text Length: {len(full_text)} chars")
            print(f"[Critic] Context Length: {len(context)} chars")

            # 2. Invoke LLM
            print(f"[Critic] Invoking LLM for {url}...")
            start_t = time.time()
            
            # Note: We keep the timeout large for debugging to see if it finishes eventually
            response = await self.chain.ainvoke({
                "title": title,
                "snippet": snippet,
                "url": url,
                "full_text": full_text[:2000],
                "rank": rank,
                "context": context[:2500] # Log truncated context
            })
            
            print(f"[Critic] LLM Response received in {time.time() - start_t:.2f}s")
            
            # 3. Inspect Raw Output
            content = response.content.strip()
            print(f"[Critic] Raw LLM Output (first 100 chars): {content[:100]}...")
            
            # 4. Parsing Logic
            print("[Critic] Attempting to parse JSON...")
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
                print("[Critic] Found ```json block. extracted.")
            elif "```" in content:
                 content = content.split("```")[1].strip()
                 print("[Critic] Found ``` block. extracted.")
            
            # 5. JSON Loading
            analysis = json.loads(content)
            print("[Critic] JSON parsed successfully.")
            
            # 6. Extract fields
            should_use = analysis.get("recommendation", {}).get("should_use", False)
            rel_score = analysis.get("sourcing_quality", {}).get("confidence", 0)
            print(f"[Critic] Extraction Result -> Should Use: {should_use}, Score: {rel_score}")

            return {
                "url": url,
                "should_use": should_use,
                "reliability_score": rel_score,
                "bias_label": analysis.get("polarity_tone", {}).get("label", "Unknown"),
                "key_evidence": analysis.get("relevance_assessment", {}).get("justification", "") + " " + 
                                str([c.get('evidence_summary') for c in analysis.get("claims_analysis", [])[:2]]),
            }
            
        except Exception as e:
            print(f"\n[Critic] ❌ ERROR on {url}")
            print(f"[Critic] Error Type: {type(e).__name__}")
            print(f"[Critic] Error Message: {str(e)}")
            # If it was a JSON error, print what failed to parse
            if 'content' in locals():
                print(f"[Critic] Failed Content Snippet: {content[:200]}...")
                
            return {"should_use": False, "error": str(e)}    
class FactCheckingAgentol:
    """Main fact-checking agent that orchestrates the pipeline"""
    
    def __init__(self):
        self.router = RouterAgent()
        self.claim_extractor = ClaimExtractorAgent()
        self.claim_verifier = ClaimVerifierAgent()
    
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
