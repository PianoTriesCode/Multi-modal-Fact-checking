import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_EXCEL = "Multi Modal Fact Checking Dataset.xlsx" 
IMAGE_FOLDER = "./images" 
OUTPUT_FILE = "fact_check_results_one_more.csv"

# Supported image extensions
VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

async def process_batch():
    # 1. Load the Excel Sheet (lookup table)
    print("Loading Excel lookup table...")
    try:
        df = pd.read_excel(INPUT_EXCEL, header=1)
        df.columns = df.columns.str.strip()
        df['adId'] = df['adId'].astype(str).str.strip()
        
        # Clean duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['adId'], keep='first')
        if initial_count - len(df) > 0:
            print(f"Warning: Dropped {initial_count - len(df)} duplicate rows.")

        lookup_db = df.set_index('adId').to_dict('index')
        print(f"Loaded lookup table with {len(lookup_db)} unique records.")
    except Exception as e:
        print(f"Could not read Excel file: {e}")
        return

    # 2. Get Image Files
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' does not exist.")
        return

    all_files = os.listdir(IMAGE_FOLDER)
    image_files = [f for f in all_files if f.lower().endswith(VALID_IMAGE_EXTS)]
    
    # --- RESUME LOGIC STARTS HERE ---
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            print(f"Found existing output file: {OUTPUT_FILE}")
            existing_df = pd.read_csv(OUTPUT_FILE)
            # Make sure to treat IDs as strings for comparison
            if 'Ad ID' in existing_df.columns:
                processed_ids = set(existing_df['Ad ID'].astype(str).str.strip())
                print(f"Skipping {len(processed_ids)} images that are already done.")
        except Exception as e:
            print(f"Warning: Could not read existing output file (might be empty): {e}")

    # Filter the batch to only include images NOT in processed_ids
    files_to_process = []
    for f in image_files:
        # Extract ID just like we do in the loop
        f_id = os.path.splitext(f)[0].strip()
        if f_id not in processed_ids:
            files_to_process.append(f)

    if not files_to_process:
        print("All images in this folder have already been processed! ✅")
        return

    print(f"\nFound {len(image_files)} total images. {len(processed_ids)} skipped. processing remaining {len(files_to_process)}...")
    # --------------------------------

    # 3. Initialize Agent
    agent = FactCheckingAgentol()
    results_data = []
    correct_predictions = 0
    total_processed = 0

    # 4. Processing Loop
    for filename in tqdm(files_to_process, desc="Processing Images"):
        ad_id = os.path.splitext(filename)[0].strip()
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        row_data = lookup_db.get(ad_id)
        
        if not row_data:
            print(f"\nSkipping {filename}: ID '{ad_id}' not found in Excel sheet.")
            continue

        ground_truth_raw = str(row_data.get('Ground Truth', '')).strip().lower()
        ground_truth = "true" if "true" in ground_truth_raw else "false"

        target_claim_text = str(row_data.get('claimText', '')).strip()

        try:
            # Check if specific claim text exists
            if target_claim_text and target_claim_text.lower() != 'nan':
                 response = await agent.fact_check(input_data=file_path, specific_claim=target_claim_text)
            else:
                 response = await agent.fact_check(input_data=file_path)

            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                raw_ai_verdict = str(verification.get("verdict", "false")).strip().lower()
                ai_verdict = "true" if "true" in raw_ai_verdict else "false"
                
                is_correct = (ground_truth == ai_verdict)
                if is_correct:
                    correct_predictions += 1
                
                result_row = {
                    "File Name": filename,
                    "Ad ID": ad_id,
                    "AI Verdict": ai_verdict,
                    "Ground Truth": ground_truth,
                    "Match": "YES" if is_correct else "NO",
                    "Reasoning": verification.get("evidence_snippet", ""),
                    "Status": "Success"
                }
            else:
                result_row = {
                    "File Name": filename, 
                    "Ad ID": ad_id, 
                    "Status": "Failed", 
                    "Error": response.get("message", "No results")
                }

        except Exception as e:
            result_row = {
                "File Name": filename, 
                "Ad ID": ad_id, 
                "Status": "Crash", 
                "Error": str(e)
            }
        
        results_data.append(result_row)
        total_processed += 1

    # 5. Save Results (APPEND MODE)
    if results_data:
        print(f"\nProcessing complete. Appending {len(results_data)} new results to {OUTPUT_FILE}...")
        new_df = pd.DataFrame(results_data)
        
        # Check if we need to write the header (only if file didn't exist before)
        file_exists = os.path.isfile(OUTPUT_FILE)
        
        # mode='a' appends to the file
        new_df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)

        if total_processed > 0:
            accuracy = (correct_predictions / total_processed) * 100
            print(f"\n--- ACCURACY ON THIS BATCH ({total_processed} IMAGES): {accuracy:.2f}% ---")
    else:
        print("\nNo new results to save.")

if __name__ == "__main__":
    asyncio.run(process_batch())