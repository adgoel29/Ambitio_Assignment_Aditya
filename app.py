import json
import os
import tempfile

import streamlit as st

from config import FEEDBACK_STORE_PATH, LEARNED_RULES_PATH, VECTOR_STORE_PATH
from core.draft_generator import DraftGenerator
from core.document_processor import DocumentProcessor
from core.feedback_learner import FeedbackLearner
from core.rag_engine import RAGEngine


def ensure_data_directories() -> None:
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(FEEDBACK_STORE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LEARNED_RULES_PATH), exist_ok=True)
    os.makedirs(os.path.join("data", "temp_upload"), exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    target_dir = os.path.join("data", "temp_upload")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, uploaded_file.name)
    with open(target_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target_path


def main():
    st.set_page_config(page_title="Ambitio Legal AI", layout="wide")
    ensure_data_directories()

    if "stage" not in st.session_state:
        st.session_state.stage = "upload"
    if "learner" not in st.session_state:
        st.session_state.learner = FeedbackLearner()

    st.title("Ambitio Legal AI")
    st.markdown("A legal document intake and grounded summary generator with feedback learning.")

    if st.session_state.stage == "upload":
        uploaded_file = st.file_uploader("Upload a legal PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                temp_path = save_uploaded_file(uploaded_file)
                processor = DocumentProcessor()
                rag_engine = RAGEngine()
                draft_generator = DraftGenerator()

                try:
                    doc = processor.process(temp_path)
                    rag_engine.ingest(doc)
                    chunks = rag_engine.retrieve_for_summary()
                    learned_rules = st.session_state.learner.get_top_rules()
                    result = draft_generator.generate(chunks, doc["structured_fields"], learned_rules)

                    st.session_state.doc = doc
                    st.session_state.chunks = chunks
                    st.session_state.result = result
                    st.session_state.original_draft = result["draft_text"]
                    st.session_state.stage = "review"
                    st.session_state.draft_text_area = result["draft_text"]
                    st.rerun()
                except Exception as error:
                    st.error(f"Document processing failed: {error}")
    else:
        left, right = st.columns([4, 6])

        with left:
            st.subheader("Extracted Fields")
            st.json(st.session_state.doc.get("structured_fields", {}))

            st.subheader("Source Text")
            st.text_area(
                "Extracted document text",
                value=st.session_state.doc.get("raw_text", ""),
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

            st.subheader("Evidence Chunks")
            used_chunk_ids = set()
            for section in st.session_state.result.get("sections", {}).values():
                for cid in section.get("chunk_ids", []):
                    used_chunk_ids.add(cid)

            for chunk_id in sorted(used_chunk_ids):
                chunk = next((item for item in st.session_state.chunks if item["chunk_id"] == chunk_id), None)
                if chunk:
                    with st.expander(f"{chunk_id} — Page {chunk['page_num']}"):
                        st.write(chunk["text"])

        with right:
            st.subheader("Generated Draft")
            # Initialize session state key if not present
            if "draft_text_area" not in st.session_state:
                st.session_state.draft_text_area = st.session_state.result.get("draft_text", "")
            
            draft_text = st.text_area(
                "Draft content",
                key="draft_text_area",
                height=500,
                label_visibility="collapsed",
            )

        st.markdown("---")
        with st.expander("📎 Evidence Breakdown by Section", expanded=True):
            for section_name, section_data in st.session_state.result.get("sections", {}).items():
                st.markdown(f"**{section_name}**")
                st.write(section_data.get("text", ""))
                if section_data.get("chunk_ids"):
                    for cid in section_data.get("chunk_ids", []):
                        chunk = next((c for c in st.session_state.chunks if c["chunk_id"] == cid), None)
                        if chunk:
                            with st.expander(f"↳ {cid} (Page {chunk['page_num']})"):
                                st.caption(chunk["text"])
                else:
                    st.caption("_No source chunks — flagged as gap_")
                st.divider()

        st.markdown("---")
        st.subheader("Feedback")
        operator_feedback = st.text_area("Describe what you changed and why...", key="operator_feedback", height=150)
        if st.button("Submit Edits & Improve Future Drafts"):
            learner = st.session_state.learner
            learner.capture_edit(
                st.session_state.original_draft,
                st.session_state.get("draft_text_area", st.session_state.original_draft),
                operator_feedback,
                {"doc_id": st.session_state.doc.get("doc_id", ""), "file_name": st.session_state.doc.get("file_name", "")},
            )
            st.success(f"Edits captured. {len(learner.learned_rules)} rules now in store.")

        with st.sidebar:
            st.header("Learned Rules")
            if st.checkbox("Show Learned Rules", value=True):
                rules = st.session_state.learner.learned_rules
                if rules:
                    st.table(
                        [{"rule": item["rule"], "frequency": item["frequency"]} for item in rules]
                    )
                else:
                    st.write("No learned rules yet.")

        if st.button("Upload a different document"):
            st.session_state.stage = "upload"
            st.session_state.pop("doc", None)
            st.session_state.pop("chunks", None)
            st.session_state.pop("result", None)
            st.session_state.pop("original_draft", None)
            st.session_state.pop("draft_text_area", None)
            st.session_state.pop("operator_feedback", None)
            st.rerun()


if __name__ == "__main__":
    main()
