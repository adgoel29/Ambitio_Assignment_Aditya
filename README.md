# Ambitio Legal AI

A laptop-friendly Streamlit app that takes legal documents, extracts structured data, builds a Gemini-backed retrieval store, and generates a grounded fact summary with inline source citations.

The app also learns from your edits and instruction-style feedback so future drafts improve over time.

---

## Quick start (easy laptop setup)

### 1) Create a clean Python environment

```bash
cd /home/aditya/Desktop/ambit/ambitio_legal_ai
python3 -m venv .venv
source .venv/bin/activate
```

If you already use a virtualenv named `ambt`, activate that instead:

```bash
source /home/aditya/Desktop/ambit/ambt/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install OCR support (optional but recommended for scanned PDFs)

- Linux:
  ```bash
  sudo apt install tesseract-ocr
  ```
- macOS:
  ```bash
  brew install tesseract
  ```
- Windows:
  Download and install from https://github.com/tesseract-ocr/tesseract

### 4) Configure Gemini credentials

Create or export your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

For a one-time session on Linux/macOS:

```bash
GEMINI_API_KEY="your-key-here" streamlit run app.py
```

### 5) Run the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal, usually `http://localhost:8501`.

---

## What this app does

1. Accepts PDF, DOCX, or TXT legal documents.
2. Extracts raw text, optional OCR text, and structured fields.
3. Chunks the document and stores embeddings in a local FAISS vector store.
4. Retrieves evidence chunks and generates a draft summary in structured JSON.
5. Shows the draft with inline chunk citations.
6. Lets you edit the draft directly and provide feedback or instructions.
7. Learns reusable drafting rules from your feedback for future summaries.

---

## Short architecture overview

- `app.py`
  - Streamlit user interface.
  - Handles upload, review, feedback submission, and draft regeneration.

- `core/document_processor.py`
  - Converts PDF/DOCX/TXT into text.
  - Applies OCR fallback for scanned pages.
  - Extracts structured metadata and fields.

- `core/rag_engine.py`
  - Splits the document into evidence chunks.
  - Builds and queries a FAISS vector store using Gemini embeddings.

- `core/draft_generator.py`
  - Builds the prompt with evidence, structured fields, learned rules, and operator instructions.
  - Generates a JSON-formatted draft with inline citations.

- `core/feedback_learner.py`
  - Stores operator edits.
  - Extracts reusable rules from feedback using Gemini.
  - Ranks and persists top rules in `data/learned_rules.json`.

---

## Assumptions and tradeoffs

- **Assumptions**
  - Users will provide corrections or instruction-style feedback after editing the draft.
  - Legal text can be coarsely chunked and retrieved for summary generation.
  - A small local JSON store is sufficient for early-stage feedback learning.

- **Tradeoffs**
  - Uses JSON files for persistence instead of a database, which is easiest for laptop use but not ideal for heavy production workloads.
  - Uses Gemini for both embeddings and generation to minimize integration complexity.
  - Keeps the UX simple: direct draft editing plus a separate feedback/instruction box.
  - Generates a structured JSON draft rather than a free-form document so downstream review and evidence mapping are easier.

---

## Sample inputs and expected outputs

### Example input

- A legal agreement or report in PDF/DOCX/TXT form.
- Extracted structured fields such as parties, dates, amounts, and obligations.
- Operator feedback like:
  - "Focus more on the timeline and remove repeated obligations."
  - "Do not speculate on future events."
  - "Summarize the parties and their responsibilities clearly."

### Example output

The app will generate a draft with sections such as:

- `Document Overview`
- `Parties Involved`
- `Key Facts & Timeline`
- `Claims & Obligations`
- `Important Terms`
- `Gaps & Unclear Info`

Each section is backed by inline citations, for example:

- `The agreement was signed on 01/02/2025 [CHUNK_004].`
- `Party A is responsible for payment of legal fees [CHUNK_012].`

The UI also displays source chunk text under each citation and shows learned rules in the sidebar.

---

## Evaluation approach and results

### How the system is evaluated

- **Citation coverage**
  - Ensure each factual claim is supported by at least one chunk citation.
- **Retrieval relevance**
  - Spot-check the top evidence chunks returned for summary queries.
- **Feedback loop**
  - Submit operator feedback and verify that the regenerated draft applies the new instructions.
- **Rule persistence**
  - Confirm learned rules are saved in `data/learned_rules.json` and influence later summaries.

### Expected outcome

- Drafts should remain grounded in document evidence.
- Operator instructions should alter the next regenerated draft.
- Learned rules should accumulate over time and improve consistency.

---

## Notes

- If you use a laptop, keep the app local and use the browser UI at `http://localhost:8501`.
- If a PDF is scanned, install Tesseract for OCR.
- If the app does not follow feedback, provide explicit instructions in the feedback box and regenerate the draft.
- The simplest workflow is:
  1. upload document,
  2. review draft,
  3. edit and write feedback/instructions,
  4. submit and regenerate.
