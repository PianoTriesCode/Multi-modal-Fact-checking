import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_EXCEL = "Multi Modal Fact Checking Dataset.xlsx" 
IMAGE_FOLDER = "./images" 
OUTPUT_FILE = "fact_check_results.csv"

# Supported image extensions
VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')

async def process_batch():
    # 1. Load the Excel Sheet (lookup table)
    print("Loading Excel lookup table...")
    try:
        # Load with header=1 to skip the title row
        df = pd.read_excel(INPUT_EXCEL, header=1)
        df.columns = df.columns.str.strip() # Fix whitespace in headers
        
        # Clean IDs
        df['adId'] = df['adId'].astype(str).str.strip()
        
        # --- FIX: DROP DUPLICATE AD IDs ---
        # This keeps the first occurrence and removes subsequent duplicates
        # to prevent the "Index must be unique" error.
        initial_count = len(df)
        df = df.drop_duplicates(subset=['adId'], keep='first')
        dropped_count = initial_count - len(df)
        
        if dropped_count > 0:
            print(f"Warning: Dropped {dropped_count} duplicate rows based on 'adId'.")
        # ----------------------------------

        # Create the lookup dictionary
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
    
    # --- TEST LIMIT: ONLY FIRST 3 IMAGES ---
    test_batch = image_files
    # ---------------------------------------

    if not test_batch:
        print("No images found in the folder.")
        return

    print(f"\nFound {len(image_files)} images. Testing on the first {len(test_batch)}...")

    # 3. Initialize Agent
    agent = FactCheckingAgentol()
    results_data = []
    correct_predictions = 0
    total_processed = 0

    # 4. Processing Loop
    for filename in tqdm(test_batch, desc="Processing Images"):
        # Extract ID from filename (e.g., "CR123.jpg" -> "CR123")
        ad_id = os.path.splitext(filename)[0].strip()
        file_path = os.path.join(IMAGE_FOLDER, filename)
        
        # Look up Ground Truth
        row_data = lookup_db.get(ad_id)
        
        if not row_data:
            print(f"\nSkipping {filename}: ID '{ad_id}' not found in Excel sheet.")
            continue

        ground_truth_raw = str(row_data.get('Ground Truth', '')).strip().lower()
        ground_truth = "true" if "true" in ground_truth_raw else "false"

        try:
            # Run Agent
            response = await agent.fact_check(file_path)

            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                # Normalize Verdict
                raw_ai_verdict = str(verification.get("verdict", "false")).strip().lower()
                ai_verdict = "true" if "true" in raw_ai_verdict else "false"
                
                # Compare
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

    # 5. Save Results
    print(f"\nProcessing complete. Saving to {OUTPUT_FILE}...")
    pd.DataFrame(results_data).to_csv(OUTPUT_FILE, index=False)

    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        print(f"\n--- ACCURACY ON {total_processed} IMAGES: {accuracy:.2f}% ---")

if __name__ == "__main__":
    asyncio.run(process_batch())