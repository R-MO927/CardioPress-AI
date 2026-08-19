from pathlib import Path
import base64
import html
import os

import streamlit as st
from dotenv import load_dotenv

from src.retrieval import CardioPressRetriever
from src.generation import generate_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CardioPress AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
HEART_PATH = ROOT / "assets" / "heart_background.png"


# ============================================================
# ENVIRONMENT / SECRETS
# ============================================================

# Local development:
# load variables from .env if it exists.
ENV_PATH = ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def get_secret(name: str, default=None):
    """
    Safely get a configuration value.

    Priority:
    1. Streamlit Secrets
    2. Environment variable
    3. Default value
    """

    # --------------------------------------------------------
    # Streamlit Cloud / Streamlit Secrets
    # --------------------------------------------------------

    try:

        value = st.secrets.get(name)

        if value is not None and str(value).strip():

            return str(value).strip()

    except Exception:
        pass

    # --------------------------------------------------------
    # Local environment / .env
    # --------------------------------------------------------

    value = os.getenv(name)

    if value is not None and value.strip():

        return value.strip()

    return default


GEMINI_API_KEY = get_secret(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = get_secret(
    "GEMINI_MODEL",
    "gemini-3-flash-preview"
)


# ============================================================
# BACKGROUND IMAGE
# ============================================================

heart_base64 = ""

if HEART_PATH.exists():

    try:

        heart_bytes = HEART_PATH.read_bytes()

        heart_base64 = base64.b64encode(
            heart_bytes
        ).decode("utf-8")

    except Exception:

        heart_base64 = ""


# ============================================================
# RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    return CardioPressRetriever()


try:

    retriever = load_retriever()

    retriever_ready = True

    retriever_error = None

except Exception as error:

    retriever = None

    retriever_ready = False

    retriever_error = error


# ============================================================
# BACKGROUND CSS
# ============================================================

if heart_base64:

    background_css = f"""
        background-image:
            linear-gradient(
                rgba(3, 5, 11, 0.58),
                rgba(3, 5, 11, 0.84)
            ),
            url("data:image/png;base64,{heart_base64}");

        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    """

else:

    background_css = """
        background:
            radial-gradient(
                circle at 50% 25%,
                rgba(180, 25, 75, 0.28),
                rgba(3, 5, 11, 1) 72%
            );
    """


# ============================================================
# GLOBAL STREAMLIT CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       APP
       ======================================================== */

    .stApp {{
        {background_css}
        color: #ffffff;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    header {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 28px !important;
        padding-bottom: 50px !important;
    }}


    /* ========================================================
       INPUT
       ======================================================== */

    div[data-testid="stTextInput"] {{
        margin-top: 0;
    }}

    div[data-testid="stTextInput"] input {{
        min-height: 58px !important;
        padding: 0 18px !important;
        border-radius: 16px !important;

        background: rgba(255,255,255,0.055) !important;

        color: #ffffff !important;

        border: 1px solid rgba(255,255,255,0.14) !important;

        font-size: 1rem !important;
    }}

    div[data-testid="stTextInput"] input::placeholder {{
        color: #818996 !important;
    }}

    div[data-testid="stTextInput"] input:focus {{
        border-color: #ff527d !important;

        box-shadow:
            0 0 0 2px
            rgba(255,82,125,0.13) !important;
    }}


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-testid="stSelectbox"] label {{
        color: #aeb6c3 !important;
    }}

    div[data-testid="stSelectbox"] > div > div {{
        background: rgba(255,255,255,0.055) !important;

        border: 1px solid rgba(255,255,255,0.13) !important;

        border-radius: 15px !important;

        color: #ffffff !important;
    }}


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton {{
        margin-top: 12px;
    }}

    .stButton > button {{
        width: 100%;
        min-height: 54px;

        border-radius: 16px;

        border: 1px solid rgba(255,255,255,0.08);

        background:
            linear-gradient(
                135deg,
                #ff527d,
                #d72f5d
            );

        color: #ffffff !important;

        font-size: 1rem;
        font-weight: 800;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);

        box-shadow:
            0 12px 30px
            rgba(255,82,125,0.30);
    }}


    /* ========================================================
       ANSWER MARKDOWN
       ======================================================== */

    .answer-content {{
        color: #d3d8e0;
        font-size: 1rem;
        line-height: 1.8;
    }}

    .answer-content h1,
    .answer-content h2,
    .answer-content h3 {{
        color: #ffffff !important;
    }}

    .answer-content p {{
        color: #d3d8e0 !important;
        line-height: 1.8;
    }}

    .answer-content li {{
        color: #d0d6df !important;
        line-height: 1.75;
        margin-bottom: 7px;
    }}

    .answer-content strong {{
        color: #ffffff;
    }}

    .answer-content code {{
        color: #ff9bb1;
    }}


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 15px;
    }}


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {{

        .block-container {{
            padding-left: 15px !important;
            padding-right: 15px !important;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP BAR
# ============================================================

st.html(
    """
    <div style="
        min-height:58px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding:12px 20px;
        border-radius:18px;
        background:rgba(7,9,17,0.72);
        border:1px solid rgba(255,255,255,0.10);
        backdrop-filter:blur(18px);
        box-shadow:0 12px 35px rgba(0,0,0,0.20);
        box-sizing:border-box;
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:9px;
            color:white;
            font-size:1.18rem;
            font-weight:800;
        ">

            <span style="
                color:#ff4f7b;
                font-size:1.55rem;
                text-shadow:0 0 15px rgba(255,79,123,0.55);
            ">
                ♥
            </span>

            <span>
                CardioPress AI
            </span>

        </div>

        <div style="
            display:flex;
            align-items:center;
            color:#c8ced8;
            font-size:0.84rem;
            font-weight:500;
        ">

            <span style="
                width:8px;
                height:8px;
                margin-right:8px;
                border-radius:50%;
                background:#4ade80;
                box-shadow:0 0 12px rgba(74,222,128,0.9);
                display:inline-block;
            "></span>

            Evidence system ready

        </div>

    </div>
    """
)


# ============================================================
# CONFIGURATION STATUS
# ============================================================

if not GEMINI_API_KEY:

    st.error(
        "Gemini API configuration is missing."
    )

    st.info(
        "For Streamlit Cloud, add GEMINI_API_KEY "
        "to your app Secrets. For local development, "
        "you can keep it inside .env."
    )


# ============================================================
# RETRIEVER STATUS
# ============================================================

if not retriever_ready:

    st.error(
        "CardioPress retrieval system could not be loaded."
    )

    if retriever_error is not None:

        with st.expander(
            "Show technical error"
        ):

            st.exception(
                retriever_error
            )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div style="
        text-align:center;
        margin-top:35px;
        padding:58px 30px 35px;
        border-radius:32px;

        background:
            linear-gradient(
                180deg,
                rgba(5,7,14,0.48),
                rgba(5,7,14,0.28)
            );

        border:1px solid rgba(255,255,255,0.08);

        backdrop-filter:blur(8px);

        box-shadow:
            0 30px 90px rgba(0,0,0,0.30);

        box-sizing:border-box;
    ">

        <div style="
            display:inline-flex;
            align-items:center;
            justify-content:center;

            padding:9px 19px;

            border-radius:999px;

            background:rgba(255,79,123,0.12);

            border:1px solid rgba(255,79,123,0.32);

            color:#ff9bb2;

            font-size:0.86rem;
            font-weight:700;

            letter-spacing:0.3px;

            margin-bottom:20px;

            box-shadow:
                0 0 25px rgba(255,79,123,0.08);
        ">
            🩺 Evidence-Grounded Clinical AI
        </div>


        <div style="
            margin:0;

            color:#ffffff;

            font-size:clamp(3.4rem,7vw,6.4rem);

            line-height:0.98;

            font-weight:900;

            letter-spacing:-4px;

            text-shadow:
                0 0 35px rgba(255,79,123,0.16);
        ">
            Cardio<span style="
                color:#ff527d;
                text-shadow:
                    0 0 30px rgba(255,82,125,0.30);
            ">Press</span> AI
        </div>


        <div style="
            max-width:780px;

            margin:22px auto 0;

            color:#d0d5de;

            font-size:1.08rem;

            line-height:1.75;
        ">
            Evidence-grounded cardiovascular clinical assistant
            powered by Retrieval-Augmented Generation.
        </div>


        <div style="
            max-width:700px;

            margin:8px auto 0;

            color:#9fa7b4;

            font-size:0.92rem;

            line-height:1.6;
        ">
            Ask questions and receive answers grounded in
            retrieved cardiovascular clinical evidence.
        </div>


        <div style="
            width:100%;
            height:82px;
            margin:30px auto 0;
            overflow:hidden;
        ">

            <svg
                viewBox="0 0 1200 82"
                preserveAspectRatio="none"
                style="
                    width:100%;
                    height:100%;
                "
            >

                <polyline
                    points="0,41 1200,41"
                    fill="none"
                    stroke="rgba(255,82,125,0.22)"
                    stroke-width="1"
                />

                <polyline
                    points="
                    0,41
                    110,41
                    170,41
                    190,41
                    202,41
                    212,20
                    222,62
                    232,41
                    270,41
                    390,41
                    450,41
                    470,41
                    482,41
                    492,18
                    502,64
                    512,41
                    550,41
                    670,41
                    730,41
                    750,41
                    762,41
                    772,20
                    782,62
                    792,41
                    830,41
                    950,41
                    1010,41
                    1030,41
                    1042,41
                    1052,18
                    1062,64
                    1072,41
                    1110,41
                    1200,41
                    "
                    fill="none"
                    stroke="#ff527d"
                    stroke-width="2.5"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    style="
                        filter:
                        drop-shadow(
                            0 0 7px rgba(255,82,125,0.9)
                        );
                    "
                />

            </svg>

        </div>

    </div>
    """
)


# ============================================================
# ASK CARD TITLE
# ============================================================

st.html(
    """
    <div style="
        margin-top:30px;

        padding:34px;

        border-radius:28px;

        background:rgba(7,9,17,0.78);

        border:1px solid rgba(255,255,255,0.10);

        backdrop-filter:blur(18px);

        box-shadow:
            0 25px 75px rgba(0,0,0,0.28);

        text-align:center;
    ">

        <div style="
            color:#ffffff;

            font-size:2rem;

            line-height:1.2;

            font-weight:850;

            margin-bottom:10px;
        ">
            Ask
            <span style="color:#ff6d91;">
                CardioPress AI
            </span>
        </div>

        <div style="
            max-width:700px;

            margin:0 auto;

            color:#aeb6c3;

            font-size:0.96rem;

            line-height:1.6;
        ">
            Ask a cardiovascular health question using
            the available clinical evidence.
        </div>

    </div>
    """
)


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Clinical question",
    placeholder=(
        "Example: How does reducing salt intake "
        "help prevent hypertension?"
    ),
    label_visibility="collapsed",
)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

suggestions = [
    "Choose a suggested question...",
    "How does reducing salt intake help prevent hypertension?",
    "What are the recommended blood pressure targets for most adults?",
    "What are the main classes of antihypertensive medications?",
    "What lifestyle changes can help reduce cardiovascular risk?",
]


selected_question = st.selectbox(
    "Suggested questions",
    suggestions,
)


if (
    selected_question != "Choose a suggested question..."
    and not question.strip()
):

    question = selected_question


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🫀 Ask CardioPress AI"
)


# ============================================================
# RAG PIPELINE
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a cardiovascular health question."
        )


    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    elif not GEMINI_API_KEY:

        st.error(
            "Gemini API key is not configured."
        )

        st.info(
            "If this app is deployed publicly on "
            "Streamlit Cloud, add GEMINI_API_KEY "
            "under App Settings → Secrets."
        )


    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    elif not retriever_ready:

        st.error(
            "The retrieval system is not available."
        )


    # --------------------------------------------------------
    # RUN RAG
    # --------------------------------------------------------

    else:

        # ====================================================
        # RETRIEVAL
        # ====================================================

        with st.spinner(
            "Searching clinical evidence..."
        ):

            try:

                evidence = retriever.search(
                    question
                )

            except Exception as error:

                st.error(
                    "Retrieval error."
                )

                with st.expander(
                    "Show technical details"
                ):

                    st.exception(
                        error
                    )

                evidence = []


        # ====================================================
        # GENERATION
        # ====================================================

        if evidence:

            with st.spinner(
                "Generating evidence-grounded answer..."
            ):

                try:

                    result = generate_answer(
                        question,
                        evidence,
                    )

                except Exception as error:

                    st.error(
                        "Generation error."
                    )

                    with st.expander(
                        "Show technical details"
                    ):

                        st.exception(
                            error
                        )

                    result = None


            # =================================================
            # RESULT
            # =================================================

            if result:

                # =============================================
                # GENERATION ERROR
                # =============================================

                generation_error = result.get(
                    "generation_error"
                )

                if generation_error:

                    st.error(
                        "The AI generation service could "
                        "not generate an answer."
                    )

                    with st.expander(
                        "Show technical details"
                    ):

                        st.code(
                            str(generation_error)
                        )

                    st.stop()


                # =============================================
                # EMPTY ANSWER
                # =============================================

                answer = result.get(
                    "answer",
                    ""
                )

                if not answer or not answer.strip():

                    st.error(
                        "The AI returned an empty answer."
                    )

                    st.stop()


                # =============================================
                # REFUSAL
                # =============================================

                refusal = result.get(
                    "refusal",
                    False,
                )


                # =============================================
                # SAFETY RESPONSE
                # =============================================

                if refusal:

                    st.html(
                        """
                        <div style="
                            margin-top:32px;
                            padding:30px;
                            text-align:center;
                            border-radius:25px;
                            background:rgba(7,9,16,0.86);
                            border:1px solid rgba(255,193,7,0.18);
                            backdrop-filter:blur(18px);
                        ">

                            <div style="
                                color:#ffffff;
                                font-size:1.55rem;
                                font-weight:850;
                                margin-bottom:12px;
                            ">
                                🛡️ Safety Response
                            </div>

                            <div style="
                                max-width:700px;
                                margin:auto;
                                color:#aeb6c3;
                                font-size:0.96rem;
                                line-height:1.6;
                            ">
                                I don't have sufficient evidence
                                in the available clinical sources
                                to answer this question reliably.
                            </div>

                        </div>
                        """
                    )


                # =============================================
                # NORMAL ANSWER
                # =============================================

                else:

                    # -----------------------------------------
                    # ANSWER HEADER
                    # -----------------------------------------

                    st.html(
                        """
                        <div style="
                            margin-top:32px;
                            padding:30px 30px 10px;
                            border-radius:26px 26px 0 0;
                            background:rgba(6,8,15,0.86);
                            border:1px solid rgba(255,82,125,0.18);
                            border-bottom:none;
                            backdrop-filter:blur(18px);
                            box-shadow:
                                0 25px 75px
                                rgba(0,0,0,0.30);
                            text-align:center;
                        ">

                            <div style="
                                color:#ffffff;
                                font-size:1.65rem;
                                font-weight:850;
                                margin-bottom:15px;
                            ">
                                🫀
                                <span style="color:#ff668b;">
                                    Clinical Answer
                                </span>
                            </div>

                        </div>
                        """
                    )


                    # -----------------------------------------
                    # ANSWER
                    # -----------------------------------------

                    st.markdown(
                        '<div class="answer-content">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        answer
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )


                    # -----------------------------------------
                    # EVIDENCE
                    # -----------------------------------------

                    cited_chunks = result.get(
                        "cited_chunk_ids",
                        [],
                    )


                    if cited_chunks:

                        st.html(
                            """
                            <div style="
                                margin-top:0;
                                padding:26px;
                                border-radius:0 0 24px 24px;
                                background:rgba(7,9,16,0.82);
                                border:1px solid rgba(255,255,255,0.09);
                                border-top:1px solid rgba(255,82,125,0.10);
                                backdrop-filter:blur(18px);
                                box-shadow:
                                    0 20px 55px
                                    rgba(0,0,0,0.25);
                            ">

                                <div style="
                                    text-align:center;
                                    color:#ffffff;
                                    font-size:1.55rem;
                                    font-weight:850;
                                    margin-bottom:7px;
                                ">
                                    📚 Evidence Sources
                                </div>

                                <div style="
                                    text-align:center;
                                    color:#929aa8;
                                    font-size:0.86rem;
                                    margin-bottom:20px;
                                ">
                                    Clinical evidence retrieved
                                    and used to ground this answer.
                                </div>
                            """
                        )


                        # =====================================
                        # CITED SOURCES
                        # =====================================

                        for chunk_id in cited_chunks:

                            safe_chunk_id = html.escape(
                                str(chunk_id)
                            )

                            st.html(
                                f"""
                                <div style="
                                    display:flex;
                                    align-items:center;
                                    padding:13px 16px;
                                    margin-bottom:9px;
                                    border-radius:13px;
                                    background:
                                        rgba(255,255,255,0.035);
                                    border:
                                        1px solid
                                        rgba(255,255,255,0.06);
                                    color:#d5dae2;
                                    font-size:0.94rem;
                                ">

                                    <span style="
                                        color:#ff668b;
                                        margin-right:10px;
                                        font-size:1rem;
                                    ">
                                        🫀
                                    </span>

                                    <span>
                                        Evidence source:
                                        <strong style="
                                            color:#ffffff;
                                            font-weight:750;
                                        ">
                                            {safe_chunk_id}
                                        </strong>
                                    </span>

                                </div>
                                """
                            )


                        # =====================================
                        # EVIDENCE DETAILS
                        # =====================================

                        st.html(
                            """
                            <div style="
                                margin-top:22px;
                                padding-top:22px;
                                border-top:
                                    1px solid
                                    rgba(255,255,255,0.08);
                            ">

                                <div style="
                                    text-align:center;
                                    color:#ffffff;
                                    font-size:1.35rem;
                                    font-weight:850;
                                    margin-bottom:8px;
                                ">
                                    🔎 Retrieved Evidence Details
                                </div>

                                <div style="
                                    text-align:center;
                                    color:#929aa8;
                                    font-size:0.84rem;
                                    margin-bottom:22px;
                                ">
                                    Details of the clinical evidence
                                    retrieved for this question.
                                </div>
                            """
                        )


                        # =====================================
                        # SHOW CITED EVIDENCE
                        # =====================================

                        for item in evidence:

                            if item.get(
                                "chunk_id"
                            ) not in cited_chunks:

                                continue


                            safe_id = html.escape(
                                str(
                                    item.get(
                                        "chunk_id",
                                        "N/A"
                                    )
                                )
                            )


                            safe_rank = html.escape(
                                str(
                                    item.get(
                                        "rank",
                                        "N/A"
                                    )
                                )
                            )


                            score_value = item.get(
                                "score",
                                0
                            )


                            try:

                                safe_score = (
                                    f"{float(score_value):.4f}"
                                )

                            except Exception:

                                safe_score = html.escape(
                                    str(score_value)
                                )


                            safe_index = html.escape(
                                str(
                                    item.get(
                                        "index",
                                        "N/A"
                                    )
                                )
                            )


                            safe_text = html.escape(
                                str(
                                    item.get(
                                        "text",
                                        "N/A"
                                    )
                                )
                            )


                            st.html(
                                f"""
                                <div style="
                                    margin-bottom:18px;
                                    padding:22px;
                                    border-radius:18px;

                                    background:
                                        linear-gradient(
                                            145deg,
                                            rgba(255,255,255,0.045),
                                            rgba(255,255,255,0.018)
                                        );

                                    border:
                                        1px solid
                                        rgba(255,255,255,0.08);

                                    box-shadow:
                                        0 12px 30px
                                        rgba(0,0,0,0.16);
                                ">


                                    <div style="
                                        display:flex;
                                        align-items:center;
                                        margin-bottom:15px;
                                    ">

                                        <span style="
                                            display:flex;
                                            align-items:center;
                                            justify-content:center;

                                            width:36px;
                                            height:36px;

                                            margin-right:11px;

                                            border-radius:11px;

                                            background:
                                                rgba(255,82,125,0.12);

                                            color:#ff668b;

                                            font-size:1.05rem;
                                        ">
                                            🫀
                                        </span>


                                        <div>

                                            <div style="
                                                color:#8f98a7;
                                                font-size:0.75rem;
                                                margin-bottom:3px;
                                                text-transform:uppercase;
                                                letter-spacing:0.8px;
                                            ">
                                                Evidence Source
                                            </div>

                                            <div style="
                                                color:#ffffff;
                                                font-size:1.05rem;
                                                font-weight:800;
                                            ">
                                                {safe_id}
                                            </div>

                                        </div>

                                    </div>


                                    <div style="
                                        display:grid;
                                        grid-template-columns:
                                            repeat(
                                                3,
                                                minmax(0,1fr)
                                            );

                                        gap:10px;
                                        margin-bottom:16px;
                                    ">


                                        <div style="
                                            padding:12px;
                                            border-radius:12px;
                                            background:
                                                rgba(0,0,0,0.16);
                                            border:
                                                1px solid
                                                rgba(255,255,255,0.05);
                                        ">

                                            <div style="
                                                color:#858e9d;
                                                font-size:0.72rem;
                                                margin-bottom:5px;
                                            ">
                                                Rank
                                            </div>

                                            <div style="
                                                color:#ffffff;
                                                font-size:0.92rem;
                                                font-weight:750;
                                            ">
                                                {safe_rank}
                                            </div>

                                        </div>


                                        <div style="
                                            padding:12px;
                                            border-radius:12px;
                                            background:
                                                rgba(0,0,0,0.16);
                                            border:
                                                1px solid
                                                rgba(255,255,255,0.05);
                                        ">

                                            <div style="
                                                color:#858e9d;
                                                font-size:0.72rem;
                                                margin-bottom:5px;
                                            ">
                                                Similarity Score
                                            </div>

                                            <div style="
                                                color:#ff91aa;
                                                font-size:0.92rem;
                                                font-weight:750;
                                            ">
                                                {safe_score}
                                            </div>

                                        </div>


                                        <div style="
                                            padding:12px;
                                            border-radius:12px;
                                            background:
                                                rgba(0,0,0,0.16);
                                            border:
                                                1px solid
                                                rgba(255,255,255,0.05);
                                        ">

                                            <div style="
                                                color:#858e9d;
                                                font-size:0.72rem;
                                                margin-bottom:5px;
                                            ">
                                                Chunk Index
                                            </div>

                                            <div style="
                                                color:#ffffff;
                                                font-size:0.92rem;
                                                font-weight:750;
                                            ">
                                                {safe_index}
                                            </div>

                                        </div>

                                    </div>


                                    <div style="
                                        color:#858e9d;
                                        font-size:0.75rem;
                                        text-transform:uppercase;
                                        letter-spacing:0.7px;
                                        margin-bottom:8px;
                                    ">
                                        Retrieved Clinical Text
                                    </div>


                                    <div style="
                                        padding:17px;
                                        border-radius:13px;
                                        background:
                                            rgba(0,0,0,0.22);
                                        border:
                                            1px solid
                                            rgba(255,255,255,0.05);
                                        color:#cbd1da;
                                        font-size:0.90rem;
                                        line-height:1.75;
                                        white-space:pre-wrap;
                                        word-break:break-word;
                                    ">
                                        {safe_text}
                                    </div>


                                </div>
                                """
                            )


                        st.html(
                            """
                            </div>
                            </div>
                            """
                        )


                    # -----------------------------------------
                    # INVALID REFERENCES
                    # -----------------------------------------

                    invalid_refs = result.get(
                        "invalid_references",
                        [],
                    )


                    if invalid_refs:

                        st.warning(
                            "Some generated evidence references "
                            "could not be validated."
                        )


            else:

                st.error(
                    "No answer was generated."
                )


        # ====================================================
        # NO EVIDENCE
        # ====================================================

        else:

            st.warning(
                "I couldn't find sufficient evidence "
                "in the available clinical sources."
            )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;

        margin-top:55px;

        padding:25px 15px;

        color:#7d8593;

        font-size:0.82rem;

        line-height:1.7;

        border-top:
            1px solid rgba(255,255,255,0.07);
    ">

        <strong style="color:#b9c0ca;">
            CardioPress AI
        </strong>

        <br>

        Evidence-grounded cardiovascular clinical
        decision support

        <br>

        Retrieval-Augmented Generation (RAG)

    </div>
    """
)
