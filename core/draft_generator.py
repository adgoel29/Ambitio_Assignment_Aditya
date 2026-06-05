import json
import re
from typing import Dict, List

from config import GEMINI_API_KEY, LLM_MODEL
from core.genai_client import configure_genai, make_model, generate_with_model


class DraftGenerator:
    """Generate a grounded draft summary with inline citations."""

    def __init__(self):
        configure_genai()
        self.model = make_model(LLM_MODEL)

    def generate(
        self,
        chunks: List[Dict],
        structured_fields: Dict,
        learned_rules: List[str],
        operator_instructions: str = "",
    ) -> Dict:
        """Generate the draft and section-level evidence mapping."""
        evidence_block = self._build_evidence_block(chunks)
        prompt = self._build_prompt(evidence_block, structured_fields, learned_rules, operator_instructions)

        try:
            response = generate_with_model(self.model, prompt)
            response_text = self._response_text(response)
            # print(f"[DEBUG] response type: {type(response)}")
            # print(f"[DEBUG] response_text repr: {repr(response_text[:300])}")
            sections = self._parse_response(response_text)
            if not isinstance(sections, dict):
                print(f"[DraftGenerator] Unexpected parsed sections type: {type(sections)}")
                sections = self._fallback_sections(response_text)
        except Exception as exc:
            print(f"[DraftGenerator] Generation error: {exc}")
            sections = self._fallback_sections(response_text if 'response_text' in locals() else str(exc))

        if not isinstance(sections, dict):
            sections = self._fallback_sections(response_text if 'response_text' in locals() else "")

        draft_text = self._assemble_draft_text(sections)
        return {"draft_text": draft_text, "sections": sections}

    def _build_evidence_block(self, chunks: List[Dict]) -> str:
        lines = []
        for chunk in chunks:
            lines.append(
                f"[{chunk['chunk_id']} | Page {chunk['page_num']}]: {chunk['text'].replace('\n', ' ')}"
            )
        return "\n".join(lines)

    def _build_prompt(
        self,
        evidence_block: str,
        structured_fields: Dict,
        learned_rules: List[str],
        operator_instructions: str = "",
    ) -> str:
        learned_rules_block = ""
        if learned_rules:
            learned_rules_block = "Learned rules from previous edits:\n- " + "\n- ".join(learned_rules) + "\n\n"
            learned_rules_block += (
                "Apply these learned rules when writing the summary. "
                "If a rule cannot be supported by the evidence, keep the response evidence-based and do not invent facts.\n\n"
            )

        instruction_block = ""
        if operator_instructions:
            instruction_block = (
                "Operator feedback or further instructions:\n"
                f"{operator_instructions.strip()}\n\n"
                "Apply these instructions when generating the summary. "
                "If an instruction cannot be supported by the evidence, remain evidence-based and do not invent facts.\n\n"
            )

        return (
            "You are a legal document analyst. Write a case fact summary using ONLY the evidence chunks below.\n\n"
            "Rules:\n"
            "- Cite chunk IDs inline like [CHUNK_001] after every factual claim.\n"
            "- If information is absent from evidence, write \"Not identified in document.\"\n"
            "- Do not invent facts.\n\n"
            f"{learned_rules_block}"
            f"{instruction_block}"
            "Return ONLY valid JSON (no markdown fences) with this exact structure:\n"
            "You are a legal document analyst. Write a case fact summary using ONLY the evidence chunks below.\n\n"
            "Rules:\n"
            "- Cite chunk IDs inline like [CHUNK_001] after every factual claim.\n"
            "- If information is absent from evidence, write \"Not identified in document.\"\n"
            "- Do not invent facts.\n\n"
            f"{learned_rules_block}"
            "Return ONLY valid JSON (no markdown fences) with this exact structure:\n"
            "{\n"
            "  \"Document Overview\":    {\"text\": \"...\", \"chunk_ids\": [...]},\n"
            "  \"Parties Involved\":     {\"text\": \"...\", \"chunk_ids\": [...]},\n"
            "  \"Key Facts & Timeline\": {\"text\": \"...\", \"chunk_ids\": [...]},\n"
            "  \"Claims & Obligations\": {\"text\": \"...\", \"chunk_ids\": [...]},\n"
            "  \"Important Terms\":      {\"text\": \"...\", \"chunk_ids\": [...]},\n"
            "  \"Gaps & Unclear Info\":  {\"text\": \"...\", \"chunk_ids\": []}\n"
            "}\n\n"
            "Structured fields:\n"
            f"{json.dumps(structured_fields, indent=2)}\n\n"
            "Evidence:\n"
            f"{evidence_block}"
        )

    def _assemble_draft_text(self, sections: Dict) -> str:
        blocks = []
        for name, data in sections.items():
            if not isinstance(data, dict):
                data = {"text": str(data) if data is not None else "", "chunk_ids": []}
            blocks.append(f"## {name}\n\n{str(data.get('text', '')).strip()}\n")
        return "\n".join(blocks).strip()

    def _parse_response(self, response_text: str) -> Dict:
        text = self._strip_json(response_text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            print(f"[DraftGenerator] Parsed response is not a dict: {type(parsed)}")
        except Exception as e:
            print(f"[DraftGenerator] JSON parse failed: {e}\nRaw stripped text:\n{text[:500]}")
        return self._fallback_sections(response_text)

    def _response_text(self, response: object) -> str:
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "_get_text"):
            return response._get_text()
        return str(response)

    def _strip_json(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)  # close fences too
        text = text.strip()
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            return match.group(1)
        return text

    def _fallback_sections(self, raw_text: str) -> Dict:
        fallback = {
            "Document Overview": {"text": f"Parse failed. Raw response: {raw_text}", "chunk_ids": []},
            "Parties Involved": {"text": "Not identified in document.", "chunk_ids": []},
            "Key Facts & Timeline": {"text": "Not identified in document.", "chunk_ids": []},
            "Claims & Obligations": {"text": "Not identified in document.", "chunk_ids": []},
            "Important Terms": {"text": "Not identified in document.", "chunk_ids": []},
            "Gaps & Unclear Info": {"text": "Not identified in document.", "chunk_ids": []},
        }
        return fallback
