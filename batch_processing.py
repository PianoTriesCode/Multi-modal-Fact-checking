import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_EXCEL = "Multi Modal Fact Checking Dataset.xlsx" 
OUTPUT_FILE = "fact_check_results_text_critic.csv"

# The names you want to filter for
TARGET_EXTRACTORS = ["Sheraz", "Shaheer", "Yahya", "Abubakar"]

async def process_batch():
    # 1. Load the Data
    print("Loading data...")
    try:
        df = pd.read_excel(INPUT_EXCEL, header=1)
        df.columns = df.columns.str.strip()
        
        # Filter for text rows (claimText exists)
        df = df[df['claimText'].notna()]
        
        # Filter by Extractor Name
        df['extractorName'] = df['extractorName'].astype(str).str.strip()
        df = df[df['extractorName'].isin(TARGET_EXTRACTORS)]
        
        # Ensure adId is string for consistent matching
        df['adId'] = df['adId'].astype(str).str.strip()
        
        print(f"Loaded {len(df)} text claims from {TARGET_EXTRACTORS}.")
        
        if len(df) == 0:
            print("No records found for these names. Check spelling in Excel.")
            return

    except Exception as e:
        print(f"Could not read Excel file: {e}")
        return

    # --- RESUME LOGIC ---
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            print(f"Found existing output file: {OUTPUT_FILE}")
            existing_df = pd.read_csv(OUTPUT_FILE)
            if 'adId' in existing_df.columns:
                processed_ids = set(existing_df['adId'].astype(str).str.strip())
                print(f"Skipping {len(processed_ids)} claims that are already done.")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    # Filter out already processed rows
    df_to_process = df[~df['adId'].isin(processed_ids)]

    if len(df_to_process) == 0:
        print("All claims have already been processed! ✅")
        return

    print(f"\nProcessing remaining {len(df_to_process)} records...")
    # --------------------

    # 3. Initialize the Agent
    agent = FactCheckingAgentol()

    results_data = []
    correct_predictions = 0
    total_processed = 0

    # 4. Processing Loop
    for index, row_data in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Fact Checking Text"):
        claim_to_check = str(row_data['claimText']).strip()
        ad_id = str(row_data.get('adId', ''))
        
        # Standardize Ground Truth
        ground_truth_raw = str(row_data.get('Ground Truth', '')).strip().lower()
        ground_truth = "true" if "true" in ground_truth_raw else "false"
        
        try:
            # --- CRITICAL UPDATE ---
            # We pass the claim as BOTH input_data (context) and specific_claim (target).
            # This tells the agent: "Here is the text, and I want you to verify THIS exact sentence."
            # It skips the 'Extraction' step entirely.
            response = await agent.fact_check(input_data=claim_to_check, specific_claim=claim_to_check)
            
            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                # Normalize Verdict
                raw_ai_verdict = str(verification.get("verdict", "false")).strip().lower()
                ai_verdict = "true" if "true" in raw_ai_verdict else "false"
                
                is_correct = (ground_truth == ai_verdict)
                if is_correct:
                    correct_predictions += 1
                
                result_row = {
                    "customId": row_data.get("customId", ""),
                    "adId": ad_id,
                    "isText": 1,
                    "isImage": 0,
                    "claimText": claim_to_check[:150], # Keep a bit more context
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
                    "claimText": claim_to_check[:100],
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

    # 5. Save Results (APPEND MODE)
    if results_data:
        print(f"\nProcessing complete. Appending {len(results_data)} new results to {OUTPUT_FILE}...")
        new_df = pd.DataFrame(results_data)
        
        # Check if file exists to determine if we need a header
        file_exists = os.path.isfile(OUTPUT_FILE)
        
        # Append to CSV
        new_df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)

        if total_processed > 0:
            accuracy = (correct_predictions / total_processed) * 100
            print("-" * 30)
            print(f"BATCH REPORT")
            print("-" * 30)
            print(f"Processed:          {total_processed}")
            print(f"Correct Predictions:{correct_predictions}")
            print(f"Accuracy:           {accuracy:.2f}%")
            print("-" * 30)
    else:
        print("\nNo new results to save.")

if __name__ == "__main__":
    asyncio.run(process_batch())