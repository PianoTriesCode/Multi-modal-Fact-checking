import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_EXCEL = "claims.xlsx"       # Your excel file with text
IMAGE_FOLDER = "./images"         # Your folder with images
OUTPUT_FILE = "fact_check_results.csv"

async def process_batch():
    # 1. Load the Data
    print("Loading data...")
    # Read text claims from Excel
    try:
        df = pd.read_excel(INPUT_EXCEL)
        # Assuming the column is named 'text' or 'claim'. Let's convert to a list.
        text_inputs = df['text'].dropna().tolist() if 'text' in df.columns else []
    except Exception as e:
        print(f"Could not read Excel file: {e}")
        text_inputs = []

    # Get all image paths from the folder
    image_inputs = []
    if os.path.exists(IMAGE_FOLDER):
        image_inputs = [
            os.path.join(IMAGE_FOLDER, f) 
            for f in os.listdir(IMAGE_FOLDER) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    # Combine them into one list to process
    all_inputs = text_inputs + image_inputs
    print(f"Found {len(text_inputs)} texts and {len(image_inputs)} images. Total: {len(all_inputs)}")

    # 2. Initialize the Agent (Do this ONCE, not in the loop)
    agent = FactCheckingAgentol()

    # 3. The Processing Loop with Progress Bar (tqdm)
    results_data = []
    
    # We use tqdm to wrap the list 'all_inputs' to create a progress bar
    print("\nStarting Batch Processing...")
    for item in tqdm(all_inputs, desc="Fact Checking"):
        try:
            # Run the existing pipeline
            # Note: We await one by one to avoid hitting Rate Limits (429 Errors)
            response = await agent.fact_check(item)
            
            # 4. Parse and Store Results
            if response["status"] == "success" and response["results"]:
                # We assume 1 claim per input based on your recent changes
                verification = response["results"][0]
                
                row = {
                    "Input": os.path.basename(item) if os.path.exists(str(item)) else item[:50],
                    "Type": "Image" if isinstance(item, str) and os.path.exists(item) else "Text",
                    "Verdict": verification.get("verdict", "Unknown"),
                    "Confidence": verification.get("confidence", "Unknown"),
                    "Evidence": verification.get("evidence_snippet", "No evidence"),
                    "Status": "Success",
                    "Error": ""
                }
            else:
                # Handle cases where no claims were found or other logical errors
                row = {
                    "Input": str(item)[:50],
                    "Type": "Unknown",
                    "Verdict": "Error",
                    "Confidence": "",
                    "Evidence": "",
                    "Status": "Failed",
                    "Error": response.get("message", "Unknown error")
                }
                
        except Exception as e:
            # Handle crashes so the loop doesn't stop
            row = {
                "Input": str(item)[:50],
                "Type": "Unknown",
                "Verdict": "Error",
                "Confidence": "",
                "Evidence": "",
                "Status": "Crash",
                "Error": str(e)
            }

        # Add to our results list
        results_data.append(row)
        
        # Optional: Save continuously (append mode) to avoid data loss
        # pd.DataFrame([row]).to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)

    # 5. Final Save
    print(f"\nProcessing complete. Saving results to {OUTPUT_FILE}...")
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(process_batch())