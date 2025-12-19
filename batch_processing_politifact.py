import asyncio
import os
import json
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_JSON = "politifact_factcheck_data.json" 
OUTPUT_FILE = "fact_check_results_politifact_noSer.csv"

async def process_batch():
    # 1. Load the Data from JSON
    print("Loading data from politifact JSON...")
    try:
        data = []
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(data)} total records from {INPUT_JSON}.")
        
        # 2. Filter the Data
        # Filter for only "true" and "pants-fire" verdicts
        filtered_data = [
            record for record in data 
            if record.get('verdict', '').lower() in ['true', 'pants-fire']
        ]
        
        print(f"Filtered to {len(filtered_data)} 'true' or 'pants-fire' claims.")
        
        # Limit to first 100 claims for the test
        filtered_data = filtered_data[:100]
        print(f"Using first {len(filtered_data)} claims for evaluation.")
        
        if len(filtered_data) == 0:
            print("No records found matching the filter criteria.")
            return

    except Exception as e:
        print(f"Could not read JSON file: {e}")
        return

    # 3. Initialize the Agent
    agent = FactCheckingAgentol()

    results_data = []
    correct_predictions = 0
    total_processed = 0

    print(f"\nStarting Batch Processing for {len(filtered_data)} claims...")

    for index, record in tqdm(enumerate(filtered_data), total=len(filtered_data), desc="Fact Checking Politifact"):
        statement = str(record.get('statement', '')).strip()
        statement_originator = str(record.get('statement_originator', '')).strip()
        statement_date = str(record.get('statement_date', '')).strip()
        
        # Normalize Ground Truth
        # 'true' -> 'true'
        # 'pants-fire' -> 'false' (because pants-fire is a lie)
        raw_ground = str(record.get('verdict', '')).strip().lower()
        if "true" in raw_ground:
            ground_truth = "true"
        elif "pants-fire" in raw_ground:
            ground_truth = "false"
        else:
            ground_truth = "false" # Default fallback
        
        # Create rich context for the verifier
        # This tells the model WHO said it and WHEN (input_data)
        full_context = f"Statement by {statement_originator} on {statement_date}: {statement}"
        
        try:
            # --- CRITICAL CHANGE ---
            # input_data = full_context (The background info/context)
            # specific_claim = statement (The EXACT sentence to verify)
            # This bypasses the extractor agent completely.
            response = await agent.fact_check(input_data=full_context, specific_claim=statement)
            
            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                # --- NORMALIZE VERDICT ---
                raw_ai_verdict = str(verification.get("verdict", "")).strip().lower()
                
                # Map AI verdicts to binary labels
                if "true" in raw_ai_verdict:
                    ai_verdict = "true"
                else:
                    ai_verdict = "false"
                
                # Check accuracy
                is_correct = (ground_truth == ai_verdict)
                if is_correct:
                    correct_predictions += 1
                
                result_row = {
                    "statement_originator": statement_originator,
                    "statement_date": statement_date,
                    "statement": statement[:100] + "...", 
                    "ground_truth_verdict": ground_truth,
                    "AI_Verdict": ai_verdict,
                    "Match": "YES" if is_correct else "NO",
                    "Reasoning": verification.get("evidence_snippet", ""),
                    "Status": "Success"
                }
            else:
                result_row = {
                    "statement_originator": statement_originator,
                    "statement": statement[:100] + "...",
                    "Status": "Failed", 
                    "Error": response.get("message", "No results")
                }
                
        except Exception as e:
            result_row = {
                "statement_originator": statement_originator,
                "statement": statement[:100] + "...",
                "Status": "Crash", 
                "Error": str(e)
            }

        results_data.append(result_row)
        total_processed += 1

    # 4. Final Save
    print(f"\nProcessing complete. Saving results to {OUTPUT_FILE}...")
    pd.DataFrame(results_data).to_csv(OUTPUT_FILE, index=False)

    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        print("-" * 30)
        print(f"POLITIFACT FACT-CHECK ACCURACY REPORT")
        print("-" * 30)
        print(f"Total Claims:        {total_processed}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Final Accuracy:      {accuracy:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(process_batch())