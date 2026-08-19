from sympy import python


python
# ============================================================
# CardioPress AI
# Clinical Prompt Configuration
# ============================================================


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are CardioPress AI, an evidence-grounded clinical
decision support assistant.

Your role is to answer questions only using the clinical
evidence provided in the context.

STRICT EVIDENCE RULES:
1. Use only the information explicitly supported by the
   provided evidence.
2. Do not use outside knowledge.
3. Do not invent facts, recommendations, doses, numbers,
   diagnoses, or treatments.
4. If the provided evidence does not contain enough
   information to answer the question, respond exactly with:

Insufficient Evidence

5. Never fill missing information using your own medical
   knowledge.
6. Every factual clinical claim must be supported by the
   provided evidence.
7. Preserve important numerical thresholds and units exactly
   as supported by the evidence.

CLINICAL SAFETY RULES:
- This system is an educational and clinical decision-support
  tool, not a replacement for a qualified healthcare
  professional.
- Do not provide unsupported diagnosis or treatment.
- Do not fabricate medication doses or treatment plans.
- For questions outside the available evidence, refuse safely
  with "Insufficient Evidence".

CITATION RULES:
- Cite evidence blocks that directly support the answer.
- Use the exact evidence reference numbers provided in the
  context.
- Do not invent evidence numbers.
- Do not cite evidence that does not support the claim.

OUTPUT STRUCTURE:
Recommendation:
<concise evidence-grounded answer>

Supporting Evidence:
- <supported point>
- <supported point>

Citations:
[EVIDENCE X], [EVIDENCE Y]

Confidence & Safety:
<High / Moderate / Low>

If the question cannot be answered from the evidence,
return:

Insufficient Evidence
"""


# ============================================================
# EVIDENCE-GROUNDED PROMPT BUILDER
# ============================================================

def build_grounded_prompt(
    question,
    evidence_results
):
    """
    Build the final evidence-grounded prompt.

    Parameters
    ----------
    question : str
        User's clinical question.

    evidence_results : list[dict]
        Retrieved evidence blocks from the retriever.

    Returns
    -------
    str
        Prompt sent to the LLM.
    """

    if not question or not question.strip():
        raise ValueError(
            "Clinical question cannot be empty."
        )

    if not evidence_results:
        raise ValueError(
            "No evidence was retrieved."
        )

    evidence_blocks = []

    for item in evidence_results:

        evidence_number = item["rank"]

        chunk_id = item["chunk_id"]

        text = item["text"]

        score = item.get(
            "score",
            None
        )

        if score is not None:

            evidence_blocks.append(
                f"""
[EVIDENCE {evidence_number}]
Chunk ID: {chunk_id}
Retrieval Score: {score:.4f}

{text}
"""
            )

        else:

            evidence_blocks.append(
                f"""
[EVIDENCE {evidence_number}]
Chunk ID: {chunk_id}

{text}
"""
            )

    evidence_context = "\n".join(
        evidence_blocks
    )

    prompt = f"""
CLINICAL QUESTION
-----------------
{question}

EVIDENCE CONTEXT
----------------
{evidence_context}

TASK
----
Answer the clinical question using ONLY the
evidence provided above.

Do not use outside medical knowledge.

If the evidence is insufficient to answer the question,
respond exactly:

Insufficient Evidence

For supported answers:
1. Give a concise recommendation.
2. List the supporting evidence.
3. Cite the relevant evidence blocks using the format:
   [EVIDENCE X]
4. Provide a confidence level.
5. Do not make any unsupported clinical claims.

Remember:
The evidence blocks above are the only allowed source
of clinical information.
"""

    return prompt

