# ============================================================
# CardioPress AI
# LLM Generation Module
# ============================================================

import os
import re

from dotenv import load_dotenv
from google import genai

from .prompts import SYSTEM_PROMPT, build_grounded_prompt


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(ENV_PATH)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)


# ============================================================
# VALIDATE API KEY
# ============================================================

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# CITATION EXTRACTION
# ============================================================

def extract_evidence_references(
    answer
):
    """
    Extract [EVIDENCE X] references from
    the generated answer.

    Example:
        [EVIDENCE 1], [EVIDENCE 4]

    Returns:
        [1, 4]
    """

    if not answer:
        return []

    matches = re.findall(
        r"\[EVIDENCE\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE
    )

    references = []

    for value in matches:

        number = int(value)

        if number not in references:
            references.append(number)

    return references


# ============================================================
# MAP EVIDENCE REFERENCES → CHUNK IDS
# ============================================================

def map_evidence_to_chunks(
    evidence_references,
    evidence_results
):
    """
    Convert evidence reference numbers into
    the actual retrieved chunk IDs.
    """

    cited_chunk_ids = []

    invalid_references = []

    for reference in evidence_references:

        index = reference - 1

        if (
            index < 0
            or index >= len(evidence_results)
        ):

            invalid_references.append(
                reference
            )

            continue

        chunk_id = evidence_results[
            index
        ]["chunk_id"]

        cited_chunk_ids.append(
            chunk_id
        )

    return (
        cited_chunk_ids,
        invalid_references
    )


# ============================================================
# REFUSAL DETECTION
# ============================================================

def detect_refusal(answer):
    """
    Detect whether the model returned the
    expected insufficient-evidence response.
    """

    if not answer:
        return False

    normalized = answer.strip().lower()

    return (
        normalized == "insufficient evidence"
    )


# ============================================================
# GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(
    question,
    evidence_results
):
    """
    Generate an evidence-grounded clinical answer.

    Parameters
    ----------
    question : str
        User's question.

    evidence_results : list[dict]
        Top-K retrieved evidence.

    Returns
    -------
    dict
        Structured generation result.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not evidence_results:

        raise ValueError(
            "No evidence was retrieved."
        )

    grounded_prompt = build_grounded_prompt(
        question=question,
        evidence_results=evidence_results
    )

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                SYSTEM_PROMPT
                                + "\n\n"
                                + grounded_prompt
                            )
                        }
                    ]
                }
            ]
        )

        # ====================================================
        # DEBUG GEMINI RESPONSE
        # ====================================================

        print(
            "===== GEMINI RESPONSE ====="
        )

        print(
            response
        )

        print(
            "============================"
        )

    except Exception as e:

        print(
            "===== GEMINI GENERATION ERROR ====="
        )

        print(
            repr(e)
        )

        print(
            "===================================="
        )

        return {
            "answer": "",
            "cited_chunk_ids": [],
            "evidence_references": [],
            "invalid_references": [],
            "refusal": False,
            "generation_error": str(e)
        }

    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    answer = getattr(
        response,
        "text",
        ""
    )

    if answer is None:
        answer = ""

    answer = answer.strip()

    # ========================================================
    # DEBUG GENERATED ANSWER
    # ========================================================

    print(
        "===== GEMINI ANSWER ====="
    )

    print(
        repr(answer)
    )

    print(
        "=========================="
    )

    # ========================================================
    # DEBUG RESPONSE METADATA
    # ========================================================

    try:

        print(
            "===== GEMINI RESPONSE METADATA ====="
        )

        print(
            "Model:",
            GEMINI_MODEL
        )

        print(
            "Response type:",
            type(response)
        )

        if hasattr(response, "candidates"):

            print(
                "Candidates:",
                len(response.candidates)
                if response.candidates
                else 0
            )

            if response.candidates:

                candidate = response.candidates[0]

                print(
                    "Finish reason:",
                    getattr(
                        candidate,
                        "finish_reason",
                        None
                    )
                )

                print(
                    "Safety ratings:",
                    getattr(
                        candidate,
                        "safety_ratings",
                        None
                    )
                )

        print(
            "======================================"
        )

    except Exception as debug_error:

        print(
            "Could not inspect response metadata:",
            repr(debug_error)
        )

    # ========================================================
    # CITATION EXTRACTION
    # ========================================================

    evidence_references = (
        extract_evidence_references(
            answer
        )
    )

    (
        cited_chunk_ids,
        invalid_references
    ) = map_evidence_to_chunks(
        evidence_references,
        evidence_results
    )

    # ========================================================
    # REFUSAL
    # ========================================================

    refusal = detect_refusal(
        answer
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "answer": answer,
        "cited_chunk_ids": cited_chunk_ids,
        "evidence_references": evidence_references,
        "invalid_references": invalid_references,
        "refusal": refusal,
        "generation_error": None
    }


# ============================================================
# SIMPLE VALIDATION
# ============================================================

def validate_generation_result(
    result
):
    """
    Basic local validation before displaying
    an answer in the application.
    """

    if not isinstance(
        result,
        dict
    ):
        return False

    answer = result.get(
        "answer",
        ""
    )

    if not answer:
        return False

    if result.get(
        "generation_error"
    ):
        return False

    if result.get(
        "invalid_references"
    ):
        return False

    return True
