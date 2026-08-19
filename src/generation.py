# ============================================================
# CardioPress AI
# LLM Generation Module
# ============================================================

import os
import re

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD PROJECT MODULES
# ============================================================

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()


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

def extract_evidence_references(answer):
    """
    Extract [EVIDENCE X] references from
    the generated answer.

    Example:
        [EVIDENCE 1]
        [EVIDENCE 4]

    Returns:
        list[int]
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
# MAP REFERENCES TO CHUNK IDS
# ============================================================

def map_evidence_to_chunks(
    evidence_references,
    evidence_results
):
    """
    Convert evidence reference numbers
    into actual retrieved chunk IDs.
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

        chunk = evidence_results[index]

        chunk_id = chunk.get(
            "chunk_id",
            f"chunk_{index + 1}"
        )

        if chunk_id not in cited_chunk_ids:

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
    Detect the expected insufficient-evidence
    response.
    """

    if not answer:
        return False

    normalized = answer.strip().lower()

    refusal_phrases = [
        "insufficient evidence",
        "insufficient clinical evidence",
        "not enough evidence",
        "i don't have sufficient evidence",
    ]

    return any(
        phrase in normalized
        for phrase in refusal_phrases
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
        User's clinical question.

    evidence_results : list[dict]
        Retrieved evidence chunks.

    Returns
    -------
    dict
        Structured generation result.
    """

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    # --------------------------------------------------------
    # VALIDATE EVIDENCE
    # --------------------------------------------------------

    if not evidence_results:

        raise ValueError(
            "No evidence was retrieved."
        )


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    grounded_prompt = build_grounded_prompt(
        question=question,
        evidence_results=evidence_results
    )


    # --------------------------------------------------------
    # FINAL PROMPT
    # --------------------------------------------------------

    final_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + grounded_prompt
    )


    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=final_prompt,
        )

    except Exception as error:

        return {
            "answer": "",
            "cited_chunk_ids": [],
            "evidence_references": [],
            "invalid_references": [],
            "refusal": False,
            "generation_error": (
                f"{type(error).__name__}: {str(error)}"
            ),
        }


    # --------------------------------------------------------
    # EXTRACT RESPONSE TEXT
    # --------------------------------------------------------

    answer = ""

    if response is not None:

        try:

            answer = response.text or ""

        except Exception:

            answer = ""


    answer = answer.strip()


    # --------------------------------------------------------
    # EMPTY RESPONSE
    # --------------------------------------------------------

    if not answer:

        return {
            "answer": "",
            "cited_chunk_ids": [],
            "evidence_references": [],
            "invalid_references": [],
            "refusal": False,
            "generation_error": (
                "Gemini returned an empty response."
            ),
        }


    # --------------------------------------------------------
    # EXTRACT CITATIONS
    # --------------------------------------------------------

    evidence_references = (
        extract_evidence_references(
            answer
        )
    )


    # --------------------------------------------------------
    # MAP CITATIONS TO CHUNKS
    # --------------------------------------------------------

    (
        cited_chunk_ids,
        invalid_references
    ) = map_evidence_to_chunks(
        evidence_references,
        evidence_results
    )


    # --------------------------------------------------------
    # REFUSAL
    # --------------------------------------------------------

    refusal = detect_refusal(
        answer
    )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "answer": answer,
        "cited_chunk_ids": cited_chunk_ids,
        "evidence_references": evidence_references,
        "invalid_references": invalid_references,
        "refusal": refusal,
        "generation_error": None,
    }


# ============================================================
# VALIDATE GENERATION RESULT
# ============================================================

def validate_generation_result(result):
    """
    Validate generation result before
    displaying it in Streamlit.
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


    if not answer or not answer.strip():
        return False


    if result.get(
        "generation_error"
    ):
        return False


    return True
