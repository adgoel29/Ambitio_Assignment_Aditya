import json
import os
import re
from datetime import datetime
from typing import Dict, List

from config import FEEDBACK_STORE_PATH, GEMINI_API_KEY, LLM_MODEL, LEARNED_RULES_PATH, MAX_LEARNED_RULES
from core.genai_client import configure_genai, make_model


class FeedbackLearner:
    """Capture operator edits and derive reusable learning rules."""

    def __init__(self):
        configure_genai()
        self.model = make_model(LLM_MODEL)
        self.feedback_store = self._load_json(FEEDBACK_STORE_PATH, default=[])
        self.learned_rules = self._load_json(LEARNED_RULES_PATH, default=[])

    def capture_edit(self, original_draft: str, edited_draft: str, operator_feedback: str, doc_metadata: Dict) -> None:
        """Store the edit record and extract new reusable rules."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "doc_metadata": doc_metadata,
            "original_draft": original_draft,
            "edited_draft": edited_draft,
            "operator_feedback": operator_feedback,
        }
        self.feedback_store.append(record)
        self._save_json(FEEDBACK_STORE_PATH, self.feedback_store)

        new_rules = self._extract_rules(original_draft, edited_draft, operator_feedback)
        self._update_rules(new_rules)

    def _extract_rules(self, original: str, edited: str, feedback: str) -> List[str]:
        """Ask Gemini to produce reusable editing rules from the operator change."""
        prompt = (
            "Analyze the edit below and extract 1-2 or more (as many as you seem fit) specific reusable instructions as a JSON array of strings.\n"
            "Return only valid JSON. No explanation, no markdown fences.\n\n"
            "Original draft:\n"
            f"{original}\n\n"
            "Edited draft:\n"
            f"{edited}\n\n"
            "Operator feedback:\n"
            f"{feedback}\n"
        )
        from core.genai_client import generate_with_model

        try:
            response = generate_with_model(self.model, prompt)
            text = self._response_text(response)
            json_text = self._strip_json(text)
            data = json.loads(json_text)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except Exception:
            pass
        return []

    def _update_rules(self, new_rules: List[str]) -> None:
        """Merge extracted rules with stored rules and rank by frequency."""
        for candidate in new_rules:
            normalized_candidate = self._normalize_text(candidate)
            found = False
            for item in self.learned_rules:
                if self._is_duplicate_rule(normalized_candidate, self._normalize_text(item.get("rule", ""))):
                    item["frequency"] = item.get("frequency", 1) + 1
                    item["last_seen"] = datetime.utcnow().isoformat() + "Z"
                    found = True
                    break
            if not found:
                self.learned_rules.append(
                    {
                        "rule": candidate,
                        "frequency": 1,
                        "last_seen": datetime.utcnow().isoformat() + "Z",
                    }
                )
        self.learned_rules.sort(key=lambda item: item.get("frequency", 1), reverse=True)
        self.learned_rules = self.learned_rules[:MAX_LEARNED_RULES]
        self._save_json(LEARNED_RULES_PATH, self.learned_rules)

    def get_top_rules(self, n: int = MAX_LEARNED_RULES) -> List[str]:
        """Return the most frequent learned rule strings."""
        return [item.get("rule", "") for item in self.learned_rules[:n]]

    def _load_json(self, path: str, default):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        except Exception:
            pass
        return default

    def _save_json(self, path: str, data) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception:
            pass

    def _response_text(self, response: object) -> str:
        if isinstance(response, dict):
            return response.get("text") or response.get("content") or str(response)
        return str(response)

    def _strip_json(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if match:
            return match.group(1)
        return text

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    def _is_duplicate_rule(self, candidate: str, existing: str) -> bool:
        if not candidate or not existing:
            return False
        candidate_words = set(candidate.split())
        existing_words = set(existing.split())
        if not candidate_words or not existing_words:
            return False
        overlap = candidate_words.intersection(existing_words)
        score = len(overlap) / max(len(candidate_words), len(existing_words))
        return score > 0.5
