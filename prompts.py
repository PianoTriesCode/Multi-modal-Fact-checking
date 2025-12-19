"""
Prompt Templates for the Fact-Checking System
"""

from langchain_core.prompts import PromptTemplate

CLAIM_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Extract verifiable political claims from the following text.

Guidelines:
1. **Preserve Tone:** Keep the original persuasive tone, emotional language, and specific phrasing used in the source. Do not neutralize the sentiment or summarize it into objective facts.
2. **Topic:** Extract ONLY claims that are political in nature (related to policies, candidates, governance, or public interest).
3. **Filter:** IGNORE any descriptions of the image content (e.g., "a person standing on stage", "text on a screen", "screenshot of a tweet"). Focus only on the message/assertion, not the visual details.

Text: {text}

Return ONLY the claims as a numbered list, one per line. No additional commentary."""
)

# Verification Prompts
# prompts.py
VERIFICATION_PROMPT = PromptTemplate(
    input_variables=["claim", "evidence", "context"],
    template="""You are an expert fact-checker evaluating political claims. 
Your goal is to determine the truthfulness of the claim based on the provided evidence and context, using intent-based reasoning.

CONTEXT: {context}
Claim: {claim}

Search Evidence:
{evidence}

Reasoning Guidelines:
1. **Analyze Intent:** The claim may use persuasive, emotional, or hyperbolic language. Identify the *underlying factual assertion* intended by the speaker.
2. **Verify Substance:** Compare that underlying assertion against the Search Evidence. Do not mark a claim 'False' solely due to subjective language (e.g., "disastrous policy") if the core event or statistic is accurate.
3. **Check for Deception:** If the intent is to mislead, omit crucial context, or present a fabrication as fact, mark it False.
4. **Visual Alignment:** Use the CONTEXT to understand if the image changes the meaning of the text (e.g., satire, meme, or contradictory evidence).

Respond in exactly this format:
VERDICT: [True or False]
EXPLANATION: [Concise explanation of how the evidence supports or refutes the intended message]"""
)
# Summary Prompt
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["results"],
    template="""Analyze these fact-checking results and provide a brief summary:

{results}

Provide a concise summary of the overall accuracy and key findings."""
)