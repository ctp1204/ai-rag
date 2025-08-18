from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .vector_database import VectorDatabase, create_vector_database
from .document_processor import DocumentProcessor
from config import settings
from typing import Tuple

# Optional BM25 for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None  # type: ignore
    _BM25_AVAILABLE = False

class RetrievalEngine:
    """Engine để truy xuất thông tin từ vector database"""

    def __init__(self, vector_db: Optional[VectorDatabase] = None):
        self.vector_db = vector_db or create_vector_database()
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.document_processor = DocumentProcessor()
        # State for BM25 corpus
        self._bm25 = None
        self._bm25_corpus = []
        # Build BM25 at startup if data already exists
        self._maybe_build_bm25()

    def add_document_from_file(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """Thêm document từ file vào database"""
        documents = self.document_processor.process_document(file_path, metadata)
        self.vector_db.add_documents(documents)
        self._maybe_build_bm25()
        return len(documents)

    def add_document_from_text(self, text: str, source: str = "direct_input", metadata: Dict[str, Any] = None) -> int:
        """Thêm document từ text trực tiếp vào database"""
        documents = self.document_processor.process_text_directly(text, source, metadata)
        self.vector_db.add_documents(documents)
        self._maybe_build_bm25()
        return len(documents)

    def search(self, query: str, k: int = None, min_score: float = None) -> List[Dict[str, Any]]:
        """Tìm kiếm documents liên quan đến query"""
        k = k or settings.max_retrieved_docs
        min_score = min_score or settings.similarity_threshold

        # Tạo embedding cho query (hỗ trợ e5 prefix nếu cần)
        query_text = ("query: " + query) if "e5" in settings.embedding_model.lower() else query
        query_embedding = self.embedding_model.encode([query_text])[0]

        # Vector search
        vector_results = self.vector_db.search(query_embedding, max(k, settings.max_retrieved_docs))

        # Hybrid: BM25 keyword search (optional)
        keyword_results: List[Dict[str, Any]] = []
        if _BM25_AVAILABLE and self._bm25 is not None:
            keyword_results = self._bm25_search(query, top_n=max(20, k))

        # Merge and re-rank
        merged = self._merge_and_rerank(query, vector_results, keyword_results)

        # Apply threshold and cut to top-k
        filtered = [d for d in merged if d.get('score', 0) >= min_score]
        return filtered[:k]

    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Lấy context liên quan cho query (để đưa vào LLM)"""
        results = self.search(query)

        if not results:
            return ""

        # Sắp xếp theo score giảm dần
        results.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Ghép text từ các results cho đến khi đạt max_context_length
        context_parts = []
        current_length = 0

        for result in results:
            text = result['text']
            if current_length + len(text) <= max_context_length:
                context_parts.append(f"[Nguồn: {result['metadata'].get('source', 'Unknown')}]\n{text}")
                current_length += len(text)
            else:
                # Thêm phần text còn lại nếu có thể
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # Chỉ thêm nếu còn đủ chỗ
                    truncated_text = text[:remaining_length-3] + "..."
                    context_parts.append(f"[Nguồn: {result['metadata'].get('source', 'Unknown')}]\n{truncated_text}")
                break

        return "\n\n".join(context_parts)

    def has_relevant_data(self, query: str, min_score: float = None) -> bool:
        """Kiểm tra xem có dữ liệu liên quan đến query không"""
        min_score = min_score or settings.similarity_threshold
        results = self.search(query, k=1, min_score=min_score)
        return len(results) > 0

    def delete_document(self, doc_id: str) -> None:
        """Xóa document khỏi database"""
        self.vector_db.delete_document(doc_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Lấy tất cả documents trong database"""
        return self.vector_db.get_all_documents()

    def get_database_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về database"""
        # Use vector_db.get_info() if available, fallback to manual counting
        if hasattr(self.vector_db, 'get_info'):
            return self.vector_db.get_info()
        else:
            # Fallback for databases that don't implement get_info()
            all_docs = self.get_all_documents()

            # Đếm số documents theo source
            source_counts = {}
            for doc in all_docs:
                source = doc.get('metadata', {}).get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1

            return {
                'total_documents': len(all_docs),
                'sources': source_counts,
                'vector_db_type': settings.vector_db_type
            }

    def clear_database(self) -> None:
        """Xóa toàn bộ database (cẩn thận!)"""
        if hasattr(self.vector_db, 'clear_all'):
            # Use optimized clear method if available
            self.vector_db.clear_all()
        else:
            # Fallback to deleting documents one by one
            all_docs = self.get_all_documents()
            for doc in all_docs:
                self.delete_document(doc['id'])

        # Reset BM25
        self._bm25 = None
        self._bm25_corpus = []

    # ----------------------
    # Hybrid retrieval utils
    # ----------------------
    def _maybe_build_bm25(self) -> None:
        """Build BM25 index lazily when data changes and package is available."""
        if not _BM25_AVAILABLE:
            return
        documents = self.get_all_documents()
        if not documents:
            self._bm25 = None
            self._bm25_corpus = []
            return
        # Simple tokenization
        self._bm25_corpus = [doc.get('text', '').lower().split() for doc in documents]
        try:
            self._bm25 = BM25Okapi(self._bm25_corpus)  # type: ignore
        except Exception:
            self._bm25 = None
            self._bm25_corpus = []

    def _bm25_search(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        if not _BM25_AVAILABLE or self._bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)  # type: ignore
        # Map back to documents
        documents = self.get_all_documents()
        indexed_docs: List[Tuple[int, float]] = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_n]
        results: List[Dict[str, Any]] = []
        for idx, score in indexed_docs:
            if idx < len(documents):
                doc = documents[idx].copy()
                doc['score'] = float(score) / (max(scores) or 1.0)
                results.append(doc)
        return results

    def _merge_and_rerank(self, query: str, vec_results: List[Dict[str, Any]], kw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hợp nhất kết quả vector và BM25; re-rank bằng weighted score đơn giản.
        Có thể nâng cấp lên cross-encoder nếu cần.
        """
        by_id: Dict[str, Dict[str, Any]] = {}

        def add(doc: Dict[str, Any], weight: float):
            key = doc.get('id') or f"{doc.get('source','')}-{doc.get('chunk_index',-1)}"
            if key not in by_id:
                by_id[key] = doc.copy()
                by_id[key]['score'] = float(doc.get('score', 0.0)) * weight
            else:
                by_id[key]['score'] += float(doc.get('score', 0.0)) * weight

        for d in vec_results:
            add(d, 1.0)
        for d in kw_results:
            add(d, 0.5)

        merged = list(by_id.values())
        merged.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return merged
