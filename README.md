# 🫀 CardioPress AI

### Evidence-Grounded Cardiovascular Clinical Decision Support using Retrieval-Augmented Generation

CardioPress AI is an **evidence-grounded clinical AI system** designed to answer cardiovascular and hypertension-related clinical questions using **Retrieval-Augmented Generation (RAG)**.

Instead of relying only on the language model's internal knowledge, CardioPress AI retrieves relevant clinical evidence from a curated medical knowledge base and uses that evidence to generate a grounded response with traceable evidence references.

The system also includes a **safety and refusal mechanism** designed to avoid producing unsupported clinical answers when sufficient evidence is not available.

---

##  Project Overview

Clinical questions often require accurate, evidence-based answers rather than general language-model responses.

A major challenge with Large Language Models in healthcare is **hallucination**: the model may generate a medically plausible answer that is not actually supported by the available evidence.

CardioPress AI addresses this problem by combining:

* Medical knowledge sources
* Document preprocessing
* Semantic embeddings
* Vector-based retrieval
* Keyword retrieval
* Hybrid retrieval
* Optional reranking
* Evidence-grounded prompting
* Citation validation
* Safety/refusal detection
* Generative AI
* Interactive Streamlit interface

The final system follows the principle:

> **Retrieve first → Ground the answer → Validate the evidence → Respond safely**

---

#  Objectives

The main objectives of CardioPress AI are to:

1. Build a reliable cardiovascular clinical knowledge base.
2. Retrieve the most relevant clinical evidence for a user question.
3. Reduce hallucination through evidence-grounded generation.
4. Provide traceable evidence references with generated answers.
5. Refuse questions when sufficient supporting evidence is unavailable.
6. Compare different information-retrieval strategies.
7. Build an interactive and professional clinical AI interface.
8. Demonstrate a complete end-to-end RAG pipeline.

---

#  Clinical Scope

The project focuses on **cardiovascular conditions and hypertension-related clinical knowledge**.

The knowledge base was designed around cardiovascular topics that are clinically connected to blood pressure and cardiovascular risk.

The project particularly emphasizes:

* Hypertension
* Blood pressure control
* Cardiovascular risk
* Heart failure
* Atrial fibrillation
* Lifestyle interventions
* Antihypertensive treatment
* Cardiovascular prevention

The goal was not to create a general medical chatbot, but rather a **focused cardiovascular evidence-grounded assistant**.

---

#  Knowledge Sources

The knowledge base was built from trusted clinical sources and medical guideline documents.

A major foundation of the project is the:

**WHO Guideline for the Pharmacological Treatment of Hypertension in Adults**

Additional cardiovascular clinical documents were incorporated to expand the knowledge base around conditions related to hypertension and cardiovascular disease, including documents covering topics such as:

* Heart Failure
* Atrial Fibrillation

The documents were selected because they provide clinically relevant information that can support questions about cardiovascular disease, hypertension, risk factors, treatment, and management.

> **Source selection principle:** The system prioritizes authoritative clinical evidence rather than general web content.

---

#  System Architecture

The complete CardioPress AI pipeline can be summarized as:

```text
                 ┌─────────────────────┐
                 │   Clinical Sources  │
                 │  Guidelines / PDFs  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Document Processing │
                 │ Cleaning / Parsing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Chunking       │
                 │ Contextual Segments │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Embeddings      │
                 │ Semantic Vectors    │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │       Retrieval Layer      │
              │                            │
              │ Semantic / BM25 / Hybrid   │
              │ + Optional Reranking       │
              └────────────┬───────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Retrieved Evidence  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Grounded Prompt     │
                 │ + Safety Rules      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Generative Model    │
                 │       Gemini        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Validation Layer    │
                 │ Citations / Refusal │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Streamlit UI       │
                 │   CardioPress AI    │
                 └─────────────────────┘
```

---

# 🔄 End-to-End Workflow

## 1. Medical Document Collection

The first stage was collecting appropriate clinical documents.

Rather than feeding the model arbitrary medical text, the project uses a **curated clinical knowledge base**.

The selected documents were chosen according to:

* Clinical relevance
* Authority
* Cardiovascular relevance
* Relationship to hypertension
* Availability of sufficient textual content
* Suitability for retrieval-based question answering

---

# 2. PDF Processing

The clinical PDFs were processed before being used by the RAG system.

The preprocessing stage included:

* Extracting text from the documents
* Removing unnecessary formatting artifacts
* Cleaning extracted text
* Preserving clinically meaningful content
* Preparing the text for chunking

This step is important because raw PDF extraction can contain:

* Broken lines
* Unnecessary whitespace
* Page artifacts
* Formatting noise
* Repeated headers and footers

A clean knowledge base improves the quality of downstream retrieval.

---

# 3. Chunking

After preprocessing, the documents were divided into smaller pieces called **chunks**.

Chunking is one of the most important parts of a RAG system.

If chunks are too small:

* Context may be lost.
* Clinical recommendations may become incomplete.
* Important relationships between statements may disappear.

If chunks are too large:

* Retrieval becomes less precise.
* Irrelevant information may be included.
* The context window becomes less efficient.

Therefore, different chunk-size ranges were considered and compared.

The main experimental ranges included:

```text
400 – 600 tokens
700 – 900 tokens
```

The goal was to determine which chunking strategy provides the best balance between:

* Retrieval relevance
* Context preservation
* Evidence completeness
* Noise reduction

---

# 4. Embedding Generation

Each chunk was transformed into a numerical vector representation using a sentence-embedding model.

These vectors allow the system to compare the semantic meaning of:

```text
User Question
        ↓
Question Embedding
        ↓
Similarity with Document Embeddings
        ↓
Most Relevant Chunks
```

During the final retrieval-artifact validation, the knowledge base contained:

```text
Chunks: 43
Embedding dimension: 384
Device: CPU
```

The embedding artifacts were successfully loaded and validated before running the final RAG pipeline.

---

# 5. Retrieval

The retrieval layer is responsible for finding the clinical evidence most relevant to the user's question.

The project explored multiple retrieval approaches.

## Semantic Retrieval

Semantic retrieval uses embeddings to identify chunks that are conceptually similar to the question.

This is useful when the user uses different wording from the source document.

For example:

```text
Question:
How can salt reduction help control blood pressure?

Document:
Reducing dietary sodium intake contributes to improved blood pressure control.
```

Even though the wording is different, semantic retrieval can identify the relationship.

---

## BM25 Keyword Retrieval

BM25 provides lexical/keyword-based retrieval.

This is useful when the question contains important medical terms, abbreviations, or exact terminology.

Examples:

```text
hypertension
SBP
DBP
atrial fibrillation
heart failure
salt intake
```

---

## Hybrid Retrieval

Hybrid retrieval combines:

```text
Semantic Retrieval
        +
BM25 Retrieval
        ↓
Better evidence coverage
```

This approach attempts to benefit from both:

* Semantic similarity
* Exact/lexical matching

---

## Cross-Encoder Reranking

A reranking stage was also considered as an additional retrieval-quality layer.

The workflow becomes:

```text
Initial Retrieval
       ↓
Top Candidate Chunks
       ↓
Cross-Encoder Reranker
       ↓
Best Evidence
```

The purpose is to improve the ordering of retrieved evidence by evaluating the relationship between the question and each candidate passage more deeply.

---

#  Top-K Retrieval

Different values of Top-K were considered during the retrieval evaluation.

The main candidates included:

```text
Top-3
Top-5
Top-10
```

The objective was to determine how much evidence should be passed to the generation layer without introducing unnecessary noise.

Increasing K can improve recall, but excessive retrieved evidence may also introduce irrelevant context.

Therefore, Top-K is treated as an important retrieval hyperparameter rather than a fixed arbitrary value.

---

#  Retrieval Evaluation

The project was designed to evaluate retrieval quality quantitatively.

The evaluation framework includes:

### Recall@K

Recall@K measures whether the relevant evidence appears among the top K retrieved results.

```text
Recall@K =
Relevant Evidence Retrieved in Top-K
-------------------------------------
       Relevant Evidence
```

Higher Recall@K means the retrieval system is more likely to retrieve the required evidence.

---

### Mean Reciprocal Rank (MRR)

MRR evaluates how highly the first relevant result appears.

```text
MRR = average(1 / rank of first relevant result)
```

A higher MRR means relevant evidence tends to appear earlier in the ranking.

---

#  Retrieval Benchmark

A fixed benchmark was designed around different question types.

The benchmark includes:

### Direct Questions

Questions that closely match the terminology in the source.

Example:

```text
How does reducing salt intake help prevent hypertension?
```

### Paraphrased Questions

Questions asking for the same information using different wording.

### Abbreviation Questions

Questions using medical abbreviations such as:

```text
SBP
DBP
CVD
```

### Threshold Questions

Questions involving clinical numerical thresholds.

### Out-of-Scope Questions

Questions that are not supported by the available clinical knowledge base.

These questions are particularly important for evaluating the system's safety behavior.

---

#  Grounded Generation

After retrieving evidence, the selected passages are passed to the generation layer.

CardioPress AI uses **Gemini** for answer generation.

The generation process does not simply ask the model:

```text
"Answer this question."
```

Instead, the system constructs a grounded prompt containing:

```text
User Question
      +
Retrieved Evidence
      +
Clinical Answering Rules
      +
Citation Requirements
      +
Safety Constraints
```

The model is instructed to base its answer on the retrieved evidence.

---

#  Grounded Prompting

The generation layer contains a system prompt that establishes the expected behavior of the clinical assistant.

The generated response is structured around:

* Recommendation
* Supporting Evidence
* Citations
* Confidence & Safety

This makes the output easier to interpret and review.

Example:

```text
Recommendation:
Reduce salt intake to less than 5 g per day...

Supporting Evidence:
- High salt consumption contributes to high blood pressure.
- Maintaining salt intake below 5 g per day helps prevent hypertension.
- High dietary salt intake contributes to uncontrolled hypertension.

Citations:
[EVIDENCE 1], [EVIDENCE 2], [EVIDENCE 6]

Confidence & Safety:
High
```

---

# 🔗 Evidence & Citation Validation

One of the important parts of CardioPress AI is that citations are not treated as simple text generated by the model.

The generation module includes mechanisms for:

* Extracting evidence references
* Validating references
* Mapping evidence references to actual chunks
* Detecting invalid references
* Returning cited chunk IDs

The generation result contains structured information such as:

```python
{
    "answer": "...",
    "cited_chunk_ids": [...],
    "evidence_references": [...],
    "invalid_references": [],
    "refusal": False,
    "generation_error": None
}
```

This allows the application to distinguish between:

```text
Generated Answer
        ↓
Evidence References
        ↓
Actual Retrieved Chunks
```

rather than blindly trusting generated citations.

---

#  Safety Layer

Clinical AI systems require an additional safety mechanism.

CardioPress AI therefore includes a refusal mechanism.

When the available evidence is not sufficient to answer a question reliably, the system should not fabricate an answer.

Instead, it returns a safety response such as:

> I don't have sufficient evidence in the available clinical sources to answer this question reliably.

The safety behavior was explicitly tested using questions outside the available evidence.

This creates an important distinction:

```text
Supported Question
        ↓
Retrieve Evidence
        ↓
Generate Grounded Answer

Unsupported Question
        ↓
Insufficient Evidence
        ↓
Refusal
```

This is a key design principle of the project.

---

#  System Testing

The final system was tested using both supported and unsupported questions.

## Supported Question

Example:

```text
How does reducing salt intake help prevent hypertension?
```

The system successfully returned a grounded recommendation related to reducing salt intake and included supporting evidence references.

---

## Another Supported Question

Example:

```text
What are the recommended blood pressure targets for most adults?
```

The system returned an evidence-grounded answer discussing blood pressure control thresholds and cited the retrieved evidence.

---

## Unsupported Question

The system was also tested using a question that was not sufficiently supported by the available clinical sources.

Instead of generating an unsupported medical response, CardioPress AI returned:

```text
I don't have sufficient evidence in the available clinical
sources to answer this question reliably.
```

This confirmed the intended refusal behavior.

---

#  User Interface

The final interface was built using **Streamlit**.

The UI was designed specifically for a cardiovascular clinical AI application.

It includes:

* CardioPress AI branding
* Cardiovascular visual identity
* Real heart background
* Dark medical interface
* Pink/red cardiovascular accent colors
* ECG visual element
* Clinical question input
* Suggested questions
* Evidence-grounded answer section
* Evidence source section
* Safety response
* Loading indicators
* Clinical evidence status indicator

The heart background was implemented as a full-screen visual layer while maintaining readable contrast for the clinical content.

---

#  UI Design Philosophy

The interface was designed around three principles:

### 1. Medical Identity

The visual language uses:

* Heart imagery
* ECG elements
* Cardiovascular colors
* Clinical terminology

### 2. Evidence Visibility

The user can clearly distinguish between:

```text
Clinical Answer
        ↓
Supporting Evidence
        ↓
Evidence Sources
```

### 3. Safety Visibility

Unsupported questions produce a clearly identifiable:

```text
 Safety Response
```

rather than silently returning an uncertain answer.

---

#  Project Structure

The project follows a modular structure:

```text
CardioPress AI/
│
├── app.py
│
├── src/
│   ├── retrieval.py
│   ├── generation.py
│   ├── prompts.py
│   └── safety.py
│
├── artifacts/
│   ├── chunks
│   ├── embeddings
│   └── retrieval artifacts
│
├── assets/
│   └── heart_background.png
│
├── data/
│   └── clinical documents
│
├── notebooks/
│   └── experiments and evaluation
│
├── requirements.txt
│
├── .env
│
└── README.md
```

> The exact contents of `artifacts/`, `data/`, and `notebooks/` may vary according to the final repository version.

---

#  Core Modules

## `app.py`

The Streamlit application.

Responsible for:

* User interface
* Question input
* Suggested questions
* Calling retrieval
* Calling generation
* Displaying answers
* Displaying evidence
* Displaying safety responses

---

## `src/retrieval.py`

Responsible for the retrieval pipeline.

Main responsibilities include:

* Loading retrieval artifacts
* Loading embeddings
* Preparing the retriever
* Searching the knowledge base
* Returning relevant evidence

---

## `src/generation.py`

Responsible for evidence-grounded answer generation.

The module includes functionality for:

* Building grounded prompts
* Calling Gemini
* Generating answers
* Detecting refusal
* Extracting evidence references
* Mapping references to chunks
* Validating generation results

---

## `src/prompts.py`

Contains prompt definitions and generation instructions used by the system.

---

## `src/safety.py`

Contains safety-related logic used to control unsupported or unsafe responses.

---

#  Environment Variables

The generation layer uses an API key configured through environment variables.

A `.env` file is used locally.

Example:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key should **never be committed to GitHub**.

The `.env` file should therefore be included in `.gitignore`.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CardioPress-AI
```

## 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create:

```text
.env
```

and add:

```env
GEMINI_API_KEY=your_api_key_here
```

## 5. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

---

#  Technical Stack

| Component            | Technology                          |
| -------------------- | ----------------------------------- |
| Programming Language | Python                              |
| UI                   | Streamlit                           |
| Retrieval            | Semantic + BM25 + Hybrid approaches |
| Embeddings           | Sentence Embeddings                 |
| Embedding Dimension  | 384                                 |
| Generation           | Gemini                              |
| Knowledge Base       | Clinical PDFs / Guidelines          |
| Evaluation           | Recall@K, MRR                       |
| Environment          | Python Virtual Environment          |
| Version Control      | Git / GitHub                        |

---

#  Why RAG Instead of a Standalone LLM?

A standalone LLM can answer questions from its learned parameters, but this creates several challenges for clinical applications:

* Potential hallucinations
* Lack of traceable evidence
* Difficulty controlling the knowledge source
* Potentially outdated information
* Difficulty verifying the origin of an answer

RAG addresses these problems by retrieving relevant evidence before generation.

CardioPress AI therefore follows:

```text
Question
   ↓
Retrieve
   ↓
Verify Evidence
   ↓
Generate
   ↓
Validate Citations
   ↓
Safe Response
```

---

#  Clinical Safety Disclaimer

CardioPress AI is an **educational and research prototype** demonstrating evidence-grounded clinical question answering.

It is **not a replacement for a qualified healthcare professional**, clinical judgment, diagnosis, or individualized medical advice.

The system should be evaluated by qualified professionals before any real-world clinical deployment.

---

#  Future Improvements

Potential future improvements include:

* Larger validated cardiovascular knowledge base
* More comprehensive clinical guidelines
* Improved document metadata
* Page-level evidence citations
* Better chunk-level provenance
* More extensive retrieval benchmarking
* Larger evaluation datasets
* Automated retrieval evaluation
* Improved reranking
* Multi-document evidence synthesis
* More advanced safety evaluation
* Clinical expert evaluation
* User authentication and audit logging
* Production deployment

---

#  Key Features

### Evidence Grounding

Answers are generated using retrieved clinical evidence.

### Hybrid Retrieval

Combines semantic and keyword-based retrieval strategies.

### Citation Validation

Generated evidence references are checked against retrieved chunks.

### Safety Refusal

The system can refuse questions when sufficient evidence is unavailable.

### Clinical Focus

The knowledge base is focused on cardiovascular and hypertension-related information.

### Interactive UI

Users can ask clinical questions through a professional Streamlit interface.

### Modular Architecture

Retrieval, generation, safety, and UI components are separated into dedicated modules.

---

#  Example

### User Question

```text
How does reducing salt intake help prevent hypertension?
```

### CardioPress AI

```text
Recommendation:
Reduce salt intake to less than 5 g per day to help prevent
hypertension and manage blood pressure.

Supporting Evidence:
- High salt consumption contributes to high blood pressure.
- Lower salt intake helps prevent hypertension.
- High dietary salt intake contributes to uncontrolled hypertension.

Citations:
[EVIDENCE 1], [EVIDENCE 2], [EVIDENCE 6]

Confidence & Safety:
High
```

The system also exposes the corresponding retrieved chunk IDs so the evidence can be traced back to the retrieval layer.

---

#  Project Validation

The project was validated progressively rather than treating the final UI as the only test.

The validation process covered:

```text
Document Validation
        ↓
Text / Chunk Validation
        ↓
Embedding Validation
        ↓
Retrieval Validation
        ↓
Top-K Experiments
        ↓
Generation Validation
        ↓
Citation Validation
        ↓
Safety / Refusal Testing
        ↓
Streamlit Integration
        ↓
End-to-End Testing
```

The final CardioPress AI pipeline successfully demonstrated the complete path from a clinical question to a grounded answer with evidence references and safety behavior.

---

#  Project Purpose

CardioPress AI was developed as a practical demonstration of how **Retrieval-Augmented Generation can be applied to evidence-grounded clinical question answering**.

The project combines concepts from:

* Artificial Intelligence
* Natural Language Processing
* Information Retrieval
* Embeddings
* Large Language Models
* Prompt Engineering
* Clinical Knowledge Retrieval
* AI Safety
* Evaluation
* User Interface Development

The main idea is simple:

> **A clinical AI system should not only generate an answer — it should be able to show where that answer came from and know when the available evidence is not enough.**

---

# 🫀 CardioPress AI

### Evidence first. Generation second. Safety always.

---

```
```
