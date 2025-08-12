from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .vector_database import VectorDatabase, create_vector_database
from .document_processor import DocumentProcessor
from config import settings

class RetrievalEngine:
    """Engine để truy xuất thông tin từ vector database"""

    def __init__(self, vector_db: Optional[VectorDatabase] = None):
        self.vector_db = vector_db or create_vector_database()
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.document_processor = DocumentProcessor()

    def add_document_from_file(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """Thêm document từ file vào database"""
        documents = self.document_processor.process_document(file_path, metadata)
        self.vector_db.add_documents(documents)
        return len(documents)

    def add_document_from_text(self, text: str, source: str = "direct_input", metadata: Dict[str, Any] = None) -> int:
        """Thêm document từ text trực tiếp vào database"""
        documents = self.document_processor.process_text_directly(text, source, metadata)
        self.vector_db.add_documents(documents)
        return len(documents)

    def search(self, query: str, k: int = None, min_score: float = None) -> List[Dict[str, Any]]:
        """Tìm kiếm documents liên quan đến query"""
        k = k or settings.max_retrieved_docs
        min_score = min_score or settings.similarity_threshold

        # Tạo embedding cho query
        query_embedding = self.embedding_model.encode([query])[0]

        # Tìm kiếm trong vector database
        results = self.vector_db.search(query_embedding, k)

        # Filter theo similarity threshold
        filtered_results = [
            result for result in results
            if result.get('score', 0) >= min_score
        ]

        return filtered_results

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
