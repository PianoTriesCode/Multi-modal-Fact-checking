"""
Multi-Agent Fact-Checking System
Main Entry Point
"""

import asyncio
import os
from dotenv import load_dotenv
from agents import FactCheckingAgentol
from tools import extract_claims_tool,search_fact_tool

# Load environment variables
load_dotenv()

async def main():
    """Main function to run the fact-checking system"""
    
    # Initialize the fact-checking agent
    print("Initializing Fact-Checking System...")
    fact_checker = FactCheckingAgentol()
    
    # Sample texts for testing
    sample_texts = [
        """
        1.5 million Acres of Oklahoma farm, land sold to foreign investors public schools are being sold out to benefit wealthy and exclusive private schools in stairs 
        """
    ]
    # sample_texts = ["sample1.jpg"]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}")
        print(f"{'='*80}")
        
        # Run fact-checking
        result = await fact_checker.fact_check(text)
        
        # Display results
        if result["status"] == "success":
            summary = result["summary"]
            
            print(f"\nRESULTS (Processed in {result['processing_time']:.2f}s)")
            print(f"Claims analyzed: {summary['total_claims']}")
            print(f"True: {summary['true_claims']} | False: {summary['false_claims']} | Uncertain: {summary['uncertain_claims']}")
            print(f"Errors: {summary['errors']}")
            print(f"Accuracy Score: {summary['accuracy_score']:.1%}")
            
            # Show individual results
            print(f"\nDETAILED RESULTS:")
            for j, verification in enumerate(result["results"], 1):
                print(f"\n{j}. {verification['claim']}")
                print(f"   Verdict: {verification['verdict']} (Confidence: {verification['confidence']})")
                if verification.get('evidence_snippet'):
                    print(f"Evidence: {verification['evidence_snippet']}")
        else:
            print(f"Error: {result['message']}")

async def demo_tools():
    """Demonstrate using tools directly"""
    print("\nTOOL DEMONSTRATION")
    print("="*50)
    
    # Test claim extraction tool
    sample_text = "The moon landing happened in 1969. Water boils at 100 degrees Celsius."
    claims = await extract_claims_tool.ainvoke({"text": sample_text})
    print(f"Extracted claims: {claims}")
    
    # Test search tool
    evidence = await search_fact_tool.ainvoke({"claim": "Moon landing year"})
    print(f"Search result preview: {evidence[:150]}...")

# ... (imports and main function remain the same) ...

if __name__ == "__main__":
    # Check for API key
    # CHANGED: Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        exit(1)
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("Error: HUGGINGFACE_API_KEY not found in environment variables")
        print("Please create a .env file with your HuggingFace API key")
        exit(1)
    # Run the main system
    asyncio.run(main())
    # Optional: Demo tools
    # asyncio.run(demo_tools())