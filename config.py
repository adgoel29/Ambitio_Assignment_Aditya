import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-key-here")
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5
MAX_LEARNED_RULES = 10
OCR_MIN_CHARS_PER_PAGE = 50

VECTOR_STORE_PATH = "data/vector_store/"
FEEDBACK_STORE_PATH = "data/feedback_store.json"
LEARNED_RULES_PATH = "data/learned_rules.json"
