"""
Prompt Templates for the Fact-Checking System
"""

from langchain_core.prompts import PromptTemplate

CLAIM_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Extract concise, factual, and verifiable statements from the following text.
Each claim should be clear, objective, and capable of being verified. They should be specifically political claims.
IGNORE ANY DESCRIPTIONS OF IMAGES AS CLAIMS

Text: {text}

Return ONLY the claims as a numbered list, one per line. No additional commentary."""
)

# Verification Prompts
VERIFICATION_PROMPT = PromptTemplate(
    input_variables=["claim", "evidence","context"],
    template="""Based on the following search evidence and context of the image, determine if the claim below is true or false.

CONTEXT: {context}
Claim: {claim}

Search Evidence:
{evidence}

Analyze the evidence carefully. Consider:
- Does the evidence directly support or contradict the claim?
- Is the evidence from reliable sources?
- Is there consensus or disagreement in the evidence?
- Does your internal knowledge align with the claim? Is it logical or reasonable?

Respond with ONLY one word: True, False. Also Include an explanation."""
)

# Summary Prompt
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["results"],
    template="""Analyze these fact-checking results and provide a brief summary:

{results}

Provide a concise summary of the overall accuracy and key findings."""
)