from typing import Dict, Any, Optional, List
import logging
from .retrieval_engine import RetrievalEngine
from .llm_client import LLMManager
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG Pipeline kết hợp retrieval và generation"""

    def __init__(self, retrieval_engine: Optional[RetrievalEngine] = None,
                 llm_manager: Optional[LLMManager] = None):
        self.retrieval_engine = retrieval_engine or RetrievalEngine()
        self.llm_manager = llm_manager or LLMManager()

        if not self.llm_manager.is_available():
            logger.warning("No LLM clients available. Please configure API keys.")

    def query(self, user_id: int, question: str, use_fallback: bool = True, provider: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Main query method với logic fallback

        Args:
            user_id: ID của người dùng để ghi log token
            question: Câu hỏi từ user
            use_fallback: Có sử dụng LLM API khi không tìm thấy dữ liệu local không
            provider: Nhà cung cấp LLM để sử dụng
            **kwargs: Các tham số khác cho LLM

        Returns:
            Dict chứa response và metadata
        """
        logger.info(f"Processing query: {question}")

        # Bước 1: Tìm kiếm thông tin liên quan trong database local
        found_documents = self.retrieval_engine.search(question)

        has_local_data = bool(found_documents)
        relevant_context = ""
        response = ""

        response_metadata = {
            'has_local_data': has_local_data,
            'used_fallback': not has_local_data and use_fallback,
            'source_documents': found_documents
        }

        try:
            if has_local_data:
                # Có dữ liệu local -> sử dụng RAG với context
                logger.info(f"Found {len(found_documents)} relevant documents, generating response with context")

                relevant_context = "\n\n".join([
                    f"[Nguồn: {doc.get('source', 'Không rõ')}]\n{doc['text']}"
                    for doc in found_documents
                ])

                response, usage_data = self.llm_manager.generate_response(
                    prompt=question,
                    context=relevant_context,
                    provider=provider,
                    **kwargs
                )
                response_metadata['source'] = 'local_rag'

                # Ghi lại token usage
                from . import database as db
                db.add_token_usage(user_id, "Chat", usage_data['provider'], usage_data['model_name'], usage_data['input_tokens'], usage_data['output_tokens'])


            elif use_fallback and self.llm_manager.is_available():
                # Không có dữ liệu local -> fallback to LLM API
                logger.info("No relevant local data found, using fallback LLM")

                relevant_context = "" # Đảm bảo context rỗng
                response, usage_data = self.llm_manager.generate_response(
                    prompt=question,
                    context=relevant_context,
                    provider=provider,
                    **kwargs
                )
                response_metadata['used_fallback'] = True
                response_metadata['source'] = 'fallback_llm'

                # Ghi lại token usage
                from . import database as db
                db.add_token_usage(user_id, "Chat", usage_data['provider'], usage_data['model_name'], usage_data['input_tokens'], usage_data['output_tokens'])

            else:
                # Không có dữ liệu và không dùng fallback
                response = self._generate_no_data_response(question)
                response_metadata['source'] = 'no_data'

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            response = f"Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn: {str(e)}"
            response_metadata['error'] = str(e)
            response_metadata['source'] = 'error'

        # Clean up the final response before returning
        cleaned_response = self._clean_response(response)

        return {
            'response': cleaned_response,
            'metadata': response_metadata,
            'context': relevant_context if has_local_data else None
        }

    def stream_query(self, user_id: int, question: str, use_fallback: bool = True, provider: Optional[str] = None, **kwargs):
        """
        Query method với streaming response.
        """
        logger.info(f"Processing streaming query: {question}")
        found_documents = self.retrieval_engine.search(question)
        has_local_data = bool(found_documents)
        relevant_context = ""

        try:
            if has_local_data:
                logger.info(f"Found {len(found_documents)} relevant documents, generating streaming response with context")
                relevant_context = "\n\n".join([
                    f"[Nguồn: {doc.get('source', 'Không rõ')}]\n{doc['text']}"
                    for doc in found_documents
                ])
                stream = self.llm_manager.generate_streaming_response(
                    prompt=question,
                    context=relevant_context,
                    provider=provider,
                    **kwargs
                )
                for chunk in stream:
                    yield chunk

            elif use_fallback and self.llm_manager.is_available():
                logger.info("No relevant local data found, using fallback LLM for streaming")
                stream = self.llm_manager.generate_streaming_response(
                    prompt=question,
                    context="",
                    provider=provider,
                    **kwargs
                )
                for chunk in stream:
                    yield chunk
            else:
                response = self._generate_no_data_response(question)
                yield response

        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}")
            yield f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"

    def _generate_no_data_response(self, question: str) -> str:
        """Tạo response khi không có dữ liệu và không dùng fallback"""
        return f"""Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi "{question}" trong cơ sở dữ liệu của mình.

Để tôi có thể trả lời tốt hơn, bạn có thể:
1. Thêm tài liệu liên quan vào hệ thống
2. Kiểm tra lại cách đặt câu hỏi
3. Bật chế độ fallback để tôi có thể sử dụng kiến thức tổng quát"""

    def _clean_response(self, text: str) -> str:
        """Removes unwanted prefixes from the response."""
        prefixes_to_remove = ["TRẢ LỜI:", "Trả lời:", "trả lời:"]
        for prefix in prefixes_to_remove:
            if text.strip().startswith(prefix):
                return text.strip()[len(prefix):].strip()
        return text

    def add_document_from_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Thêm document từ file"""
        try:
            chunks_added = self.retrieval_engine.add_document_from_file(file_path, metadata)
            logger.info(f"Added {chunks_added} chunks from {file_path}")
            return {
                'success': True,
                'chunks_added': chunks_added,
                'file_path': file_path
            }
        except Exception as e:
            logger.error(f"Error adding document from file {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }

    def add_document_from_text(self, text: str, source: str = "direct_input",
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Thêm document từ text"""
        try:
            chunks_added = self.retrieval_engine.add_document_from_text(text, source, metadata)
            logger.info(f"Added {chunks_added} chunks from text input")
            return {
                'success': True,
                'chunks_added': chunks_added,
                'source': source
            }
        except Exception as e:
            logger.error(f"Error adding document from text: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'source': source
            }

    def get_database_info(self) -> Dict[str, Any]:
        """Lấy thông tin về database"""
        return self.retrieval_engine.get_database_stats()

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm documents (để debug/admin)"""
        return self.retrieval_engine.search(query, k)

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Xóa document"""
        try:
            self.retrieval_engine.delete_document(doc_id)
            logger.info(f"Deleted document {doc_id}")
            return {'success': True, 'doc_id': doc_id}
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return {'success': False, 'error': str(e), 'doc_id': doc_id}

    def clear_database(self) -> Dict[str, Any]:
        """Xóa toàn bộ database"""
        try:
            self.retrieval_engine.clear_database()
            logger.info("Cleared entire database")
            return {'success': True}
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return {'success': False, 'error': str(e)}

    def health_check(self) -> Dict[str, Any]:
        """Kiểm tra tình trạng hệ thống"""
        return {
            'retrieval_engine': 'ok',
            'llm_available': self.llm_manager.is_available(),
            'database_stats': self.get_database_info(),
            'settings': {
                'vector_db_type': settings.vector_db_type,
                'llm_provider': settings.llm_provider,
                'embedding_model': settings.embedding_model
            }
        }
