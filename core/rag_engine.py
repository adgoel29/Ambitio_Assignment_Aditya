import json
import os
from typing import Dict, List

from config import EMBEDDING_MODEL, GEMINI_API_KEY, VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


SUMMARY_QUERIES = [
    "parties involved names roles plaintiff defendant claimant",
    "key facts events timeline chronological order",
    "legal claims obligations disputes charges",
    "dates deadlines timeframes effective dates",
    "outcomes decisions orders conclusions judgments",
    "terms conditions clauses agreements penalties",
]


class RAGEngine:
    """Handle chunking, embedding, and FAISS-based retrieval."""

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.vectorstore = None
        self.chunks: List[Dict] = []

    def ingest(self, doc: Dict) -> None:
        """Load an existing index if present, else build it from the document."""
        if self.load_index(doc["doc_id"]):
            return
        self.chunks = self._chunk_document(doc)
        self._build_vectorstore(self.chunks)
        self.save_index(doc["doc_id"])

    def _chunk_document(self, doc: Dict) -> List[Dict]:
        """Split raw text into chunks and infer page numbers."""
        chunks = []
        raw_chunks = self.splitter.split_text(doc["raw_text"])
        for index, chunk in enumerate(raw_chunks, start=1):
            page_num = 1
            snippet = chunk[:200]
            for page in doc.get("pages", []):
                if snippet and snippet in page.get("text", ""):
                    page_num = page["page_num"]
                    break
            chunks.append(
                {
                    "chunk_id": f"CHUNK_{index:03d}",
                    "doc_id": doc["doc_id"],
                    "text": chunk.strip(),
                    "page_num": page_num,
                }
            )
        return chunks

    def _build_vectorstore(self, chunks: List[Dict]) -> None:
        """Create a FAISS index from document chunks."""
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content=chunk["text"],
                metadata={"chunk_id": chunk["chunk_id"], "page_num": chunk["page_num"]},
            )
            for chunk in chunks
        ]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def save_index(self, doc_id: str) -> None:
        """Persist the FAISS index and chunk metadata locally."""
        if self.vectorstore is None:
            return
        index_path = os.path.join(VECTOR_STORE_PATH, doc_id)
        os.makedirs(index_path, exist_ok=True)
        self.vectorstore.save_local(index_path)
        with open(os.path.join(index_path, "chunks.json"), "w", encoding="utf-8") as handle:
            json.dump(self.chunks, handle, indent=2)

    def load_index(self, doc_id: str) -> bool:
        """Load a saved FAISS index if it exists."""
        from langchain_community.vectorstores import FAISS

        index_path = os.path.join(VECTOR_STORE_PATH, doc_id)
        if not os.path.exists(index_path):
            return False
        try:
            self.vectorstore = FAISS.load_local(index_path, self.embeddings)
            with open(os.path.join(index_path, "chunks.json"), "r", encoding="utf-8") as handle:
                self.chunks = json.load(handle)
            return True
        except Exception:
            return False

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Run a similarity search and return chunk metadata."""
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [
            {
                "chunk_id": result.metadata.get("chunk_id", ""),
                "page_num": result.metadata.get("page_num", 1),
                "text": result.page_content,
            }
            for result in results
        ]

    def retrieve_for_summary(self) -> List[Dict]:
        """Retrieve evidence chunks for summary generation using fixed queries."""
        combined = []
        seen = set()
        for query in SUMMARY_QUERIES:
            for item in self.retrieve(query):
                if item["chunk_id"] not in seen:
                    seen.add(item["chunk_id"])
                    combined.append(item)
        return combined
