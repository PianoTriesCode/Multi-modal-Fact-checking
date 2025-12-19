import asyncio
import os
import json
import pandas as pd
from tqdm.asyncio import tqdm
from agents import FactCheckingAgentol

# Configuration
INPUT_JSON = "politifact_factcheck_data.json" 

# --- DEFINE YOUR MODELS HERE ---
MODELS_TO_TEST = ["gpt-5.1", "gpt-4o", "o3"] 
# You can add others like "claude-3-5-sonnet-20240620" if you have LangChain configured for them.

async def process_batch_for_model(model_name):
    output_file = f"fact_check_results_{model_name}.csv"
    print(f"\n" + "="*50)
    print(f"STARTING BENCHMARK FOR MODEL: {model_name}")
    print(f"Output file: {output_file}")
    print("="*50)

    # 1. Load the Data
    print("Loading data from politifact JSON...")
    try:
        data = []
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Filter Data
        filtered_data = [
            record for record in data 
            if record.get('verdict', '').lower() in ['true', 'pants-fire']
        ]
        
        # Limit to 100 for testing
        filtered_data = filtered_data[:100]
        
        if not filtered_data:
            print("No records found.")
            return

    except Exception as e:
        print(f"Could not read JSON file: {e}")
        return

    # 2. Check for Resume (Skip already processed items for THIS model)
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            # Assuming 'statement' is unique enough, or use index if available.
            # Ideally, use a unique ID from JSON if available. For now, we use the statement text.
            if 'statement' in existing_df.columns:
                processed_ids = set(existing_df['statement'].str.strip())
                print(f"Skipping {len(processed_ids)} claims already done for {model_name}.")
        except Exception as e:
            print(f"Warning: Could not read existing output: {e}")

    # Filter
    # Note: Using statement as ID is risky if duplicates exist, but okay for this dataset.
    data_to_process = [d for d in filtered_data if str(d.get('statement', '')).strip() not in processed_ids]

    if not data_to_process:
        print(f"All claims already processed for {model_name}! ✅")
        calculate_accuracy(output_file)
        return

    # 3. Initialize Agent with SPECIFIC MODEL
    print(f"Initializing Agent with {model_name}...")
    agent = FactCheckingAgentol(model_name=model_name)

    results_data = []

    # 4. Processing Loop
    for index, record in tqdm(enumerate(data_to_process), total=len(data_to_process), desc=f"Testing {model_name}"):
        statement = str(record.get('statement', '')).strip()
        statement_originator = str(record.get('statement_originator', '')).strip()
        statement_date = str(record.get('statement_date', '')).strip()
        
        # Normalize Ground Truth
        raw_ground = str(record.get('verdict', '')).strip().lower()
        if "true" in raw_ground:
            ground_truth = "true"
        elif "pants-fire" in raw_ground:
            ground_truth = "false"
        else:
            ground_truth = "false"
        
        full_context = f"Statement by {statement_originator} on {statement_date}: {statement}"
        
        try:
            # Run Agent
            response = await agent.fact_check(input_data=full_context, specific_claim=statement)
            
            if response["status"] == "success" and response["results"]:
                verification = response["results"][0]
                
                raw_ai_verdict = str(verification.get("verdict", "")).strip().lower()
                ai_verdict = "true" if "true" in raw_ai_verdict else "false"
                is_correct = (ground_truth == ai_verdict)
                
                result_row = {
                    "Model": model_name,
                    "statement_originator": statement_originator,
                    "statement_date": statement_date,
                    "statement": statement, 
                    "ground_truth_verdict": ground_truth,
                    "AI_Verdict": ai_verdict,
                    "Match": "YES" if is_correct else "NO",
                    "Reasoning": verification.get("evidence_snippet", ""),
                    "Status": "Success"
                }
            else:
                result_row = {
                    "Model": model_name,
                    "statement": statement,
                    "Status": "Failed", 
                    "Error": response.get("message", "No results")
                }
                
        except Exception as e:
            result_row = {
                "Model": model_name,
                "statement": statement,
                "Status": "Crash", 
                "Error": str(e)
            }

        results_data.append(result_row)

    # 5. Save Results (Append)
    if results_data:
        print(f"Saving {len(results_data)} results to {output_file}...")
        new_df = pd.DataFrame(results_data)
        file_exists = os.path.isfile(output_file)
        new_df.to_csv(output_file, mode='a', header=not file_exists, index=False)
    
    # 6. Calculate Accuracy
    calculate_accuracy(output_file)

def calculate_accuracy(file_path):
    if not os.path.exists(file_path):
        return

    df = pd.read_csv(file_path)
    valid = df[df['Status'] == 'Success']
    total = len(valid)
    correct = len(valid[valid['Match'] == 'YES'])
    
    if total > 0:
        acc = (correct / total) * 100
        print(f"--- ACCURACY for {file_path}: {acc:.2f}% ({correct}/{total}) ---")

async def main():
    # Loop through all models
    for model in MODELS_TO_TEST:
        await process_batch_for_model(model)

if __name__ == "__main__":
    asyncio.run(main())