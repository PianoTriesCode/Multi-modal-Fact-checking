import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm
# Ensure you have your specific agent class in an accessible file
from agents import FactCheckingAgentol

# Configuration
INPUT_EXCEL = "claims.xlsx"       # Your excel file
OUTPUT_FILE = "fact_check_results.csv"

async def process_batch():
    # 1. Load the Data
    print("Loading data...")
    try:
        df = pd.read_excel(INPUT_EXCEL)
        # Filter for rows that actually have text to check
        df = df.dropna(subset=['claimText'])
    except Exception as e:
        print(f"Could not read Excel file: {e}")
        return

    # 2. Initialize the Agent
    agent = FactCheckingAgentol()

    # 3. Processing Loop
    results_data = []
    correct_predictions = 0
    total_processed = 0
    
    print(f"\nStarting Batch Processing for {len(df)} records...")

    for index, row_data in tqdm(df.iterrows(), total=len(df), desc="Fact Checking"):
        claim_to_check = str(row_data['claimText'])
        
        # Standardize Ground Truth for comparison
        ground_truth_raw = str(row_data.get('Ground Truth', '')).strip().lower()
        ground_truth = "true" if "true" in ground_truth_raw else "false"
        
        try:
            # Run the agent pipeline
            response = await agent.fact_check(claim_to_check)
            
            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                # --- STRATEGY FOR BINARY LOWERCASE OUTPUT ---
                # 1. Convert any verdict to a standard lowercase string
                raw_ai_verdict = str(verification.get("verdict", "false")).strip().lower()
                
                # 2. Extract strictly 'true' or 'false'
                # This ensures "True Statement" becomes "true" and "The claim is false" becomes "false"
                ai_verdict = "true" if "true" in raw_ai_verdict else "false"
                
                # Check accuracy against standardized ground truth
                is_correct = (ground_truth == ai_verdict)
                if is_correct:
                    correct_predictions += 1
                
                result_row = {
                    "customId": row_data.get("customId", ""),
                    "adId": row_data.get("adId", ""),
                    "isText": 1,
                    "isImage": 0,
                    "claimText": claim_to_check,
                    "extractorName": "FactCheckingAgentol",
                    "AI_Verdict": ai_verdict,  # This will strictly be 'true' or 'false'
                    "Ground Truth": ground_truth,
                    "Match": "Yes" if is_correct else "No",
                    "Reasoning": verification.get("evidence_snippet", ""),
                    "Confidence": verification.get("confidence", ""),
                    "Status": "Success"
                }
            else:
                result_row = {"adId": row_data.get("adId"), "Status": "Failed", "Error": "No results"}
                
        except Exception as e:
            result_row = {"adId": row_data.get("adId"), "Status": "Crash", "Error": str(e)}

        results_data.append(result_row)
        total_processed += 1

    # 4. Final Save and Accuracy Calculation
    print(f"\nProcessing complete. Saving results to {OUTPUT_FILE}...")
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(OUTPUT_FILE, index=False)

    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        print("-" * 30)
        print(f"TOTAL ACCURACY REPORT")
        print("-" * 30)
        print(f"Total Claims Processed: {total_processed}")
        print(f"Correct Predictions:    {correct_predictions}")
        print(f"Final Accuracy Score:   {accuracy:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(process_batch())