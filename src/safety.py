from sympy import python


python
# ============================================================
# CardioPress AI
# Safety & Output Validation Module
# ============================================================


# ============================================================
# OUT-OF-SCOPE / REFUSAL
# ============================================================

REFUSAL_MESSAGE = "Insufficient Evidence"


# ============================================================
# BASIC ANSWER VALIDATION
# ============================================================

def validate_answer(answer):
    """
    Validate the generated answer before displaying it.

    Returns
    -------
    tuple
        (is_valid, reason)
    """

    if answer is None:
        return False, "Empty answer."

    if not isinstance(answer, str):
        return False, "Answer is not text."

    answer = answer.strip()

    if not answer:
        return False, "Empty answer."

    return True, "Valid answer."


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(
    cited_chunk_ids,
    retrieved_chunk_ids
):
    """
    Ensure every cited chunk was actually retrieved.
    """

    if cited_chunk_ids is None:
        cited_chunk_ids = []

    if retrieved_chunk_ids is None:
        retrieved_chunk_ids = []

    invalid = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id not in retrieved_chunk_ids
    ]

    return (
        len(invalid) == 0,
        invalid
    )


# ============================================================
# REFUSAL VALIDATION
# ============================================================

def validate_refusal(
    answer
):
    """
    Validate the expected refusal format.
    """

    if not answer:
        return False

    normalized = answer.strip().lower()

    return (
        normalized
        == REFUSAL_MESSAGE.lower()
    )


# ============================================================
# FINAL RESPONSE VALIDATION
# ============================================================

def validate_final_response(
    answer,
    cited_chunk_ids,
    retrieved_chunk_ids,
    refusal=False
):
    """
    Final safety gate before displaying
    the generated response.
    """

    # --------------------------------------------------------
    # Refusal
    # --------------------------------------------------------

    if refusal:

        if validate_refusal(answer):
            return {
                "valid": True,
                "reason": "Valid evidence refusal.",
                "invalid_citations": []
            }

        return {
            "valid": False,
            "reason": "Invalid refusal format.",
            "invalid_citations": []
        }

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer_valid, reason = validate_answer(
        answer
    )

    if not answer_valid:

        return {
            "valid": False,
            "reason": reason,
            "invalid_citations": []
        }

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    citations_valid, invalid = (
        validate_citations(
            cited_chunk_ids,
            retrieved_chunk_ids
        )
    )

    if not citations_valid:

        return {
            "valid": False,
            "reason": "Invalid citation detected.",
            "invalid_citations": invalid
        }

    # --------------------------------------------------------
    # Everything passed
    # --------------------------------------------------------

    return {
        "valid": True,
        "reason": "Answer passed safety validation.",
        "invalid_citations": []
    }


# ============================================================
# USER-FACING FALLBACK
# ============================================================

def safe_fallback_message():
    """
    Message shown when the generated answer
    fails validation.
    """

    return (
        "I’m sorry, but I couldn’t generate a "
        "sufficiently evidence-grounded answer "
        "from the available clinical sources."
    )

