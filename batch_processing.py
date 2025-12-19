import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_EXCEL = "Multi Modal Fact Checking Dataset.xlsx" 
OUTPUT_FILE = "fact_check_results_text.csv"

# The names you want to filter for
TARGET_EXTRACTORS = ["Sheraz", "Shaheer", "Yahya"]

async def process_batch():
    # 1. Load the Data
    print("Loading data...")
    try:
        # Load with header=1 to skip the title row
        df = pd.read_excel(INPUT_EXCEL, header=1)
        df.columns = df.columns.str.strip() # Fix whitespace in headers
        
        # 2. Filter the Data
        # Step A: Ensure it is a text row (claimText exists)
        df = df[df['claimText'].notna()]
        
        # Step B: Filter by Extractor Name (Sheraz, Shaheer, Yahya)
        # We normalize to string and strip whitespace to be safe
        df['extractorName'] = df['extractorName'].astype(str).str.strip()
        
        # Filter: Keep row ONLY if extractorName is in our target list
        df = df[df['extractorName'].isin(TARGET_EXTRACTORS)]
        
        print(f"Loaded {len(df)} text claims from {TARGET_EXTRACTORS}.")
        
        if len(df) == 0:
            print("No records found for these names. Check spelling in Excel.")
            return

    except Exception as e:
        print(f"Could not read Excel file: {e}")
        return

    # 3. Initialize the Agent
    agent = FactCheckingAgentol()

    results_data = []
    correct_predictions = 0
    total_processed = 0
    
    # Optional: Uncomment .head(3) if you want a quick test first
    # df = df.head(3)
    
    print(f"\nStarting Batch Processing for {len(df)} text records...")

    for index, row_data in tqdm(df.iterrows(), total=len(df), desc="Fact Checking Text"):
        claim_to_check = str(row_data['claimText']).strip()
        ad_id = str(row_data.get('adId', ''))
        
        # Standardize Ground Truth
        ground_truth_raw = str(row_data.get('Ground Truth', '')).strip().lower()
        ground_truth = "true" if "true" in ground_truth_raw else "false"
        
        try:
            # Run the agent pipeline (Passing text string directly)
            response = await agent.fact_check(claim_to_check)
            
            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                # --- NORMALIZE VERDICT ---
                raw_ai_verdict = str(verification.get("verdict", "false")).strip().lower()
                ai_verdict = "true" if "true" in raw_ai_verdict else "false"
                
                # Check accuracy
                is_correct = (ground_truth == ai_verdict)
                if is_correct:
                    correct_predictions += 1
                
                result_row = {
                    "customId": row_data.get("customId", ""),
                    "adId": ad_id,
                    "isText": 1,
                    "isImage": 0,
                    "claimText": claim_to_check[:100] + "...", # Truncate for cleaner CSV
                    "extractorName": row_data.get("extractorName", ""),
                    "AI_Verdict": ai_verdict,
                    "Ground Truth": ground_truth,
                    "Match": "YES" if is_correct else "NO",
                    "Reasoning": verification.get("evidence_snippet", ""),
                    "Status": "Success"
                }
            else:
                result_row = {
                    "adId": ad_id, 
                    "Status": "Failed", 
                    "Error": response.get("message", "No results")
                }
                
        except Exception as e:
            result_row = {
                "adId": ad_id, 
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
        print(f"TOTAL TEXT ACCURACY REPORT")
        print("-" * 30)
        print(f"Total Claims:       {total_processed}")
        print(f"Correct Predictions:{correct_predictions}")
        print(f"Final Accuracy:     {accuracy:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(process_batch())