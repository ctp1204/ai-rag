import random
import re
from typing import List, Dict, Any, Tuple, Optional
from thefuzz import fuzz
from .retrieval_engine import RetrievalEngine
from .llm_client import LLMManager
import logging

logger = logging.getLogger(__name__)

class QAGenerator:
    """Generator để tạo câu hỏi và đánh giá câu trả lời"""

    def __init__(self, retrieval_engine: RetrievalEngine, llm_manager: LLMManager):
        self.retrieval_engine = retrieval_engine
        self.llm_manager = llm_manager

    def generate_questions(self, num_questions: int = 3) -> List[Dict[str, Any]]:
        """Tạo câu hỏi dựa trên dữ liệu có sẵn trong Pinecone/Vector DB"""
        try:
            # Lấy tất cả documents từ vector database
            all_docs = self.retrieval_engine.get_all_documents()

            if not all_docs:
                logger.warning("Không có dữ liệu trong database, không thể tạo câu hỏi.")
                return []

            logger.info(f"Tìm thấy {len(all_docs)} documents trong database")

            # Tạo pool câu hỏi từ tất cả documents
            question_pool = []
            unique_questions = set()

            # Xáo trộn tài liệu để tăng tính ngẫu nhiên
            random.shuffle(all_docs)

            # Lặp cho đến khi có đủ câu hỏi hoặc không thể tạo thêm
            max_attempts = len(all_docs) * 2 # Giới hạn số lần thử để tránh lặp vô hạn
            attempts = 0
            doc_index = 0

            while len(question_pool) < num_questions and attempts < max_attempts:
                # Lấy doc và lặp vòng lại nếu cần
                doc = all_docs[doc_index % len(all_docs)]

                qa_pair = self._generate_qa_from_document(doc, existing_questions=list(unique_questions))

                if qa_pair:
                    question_text = qa_pair['question'].strip().lower()
                    if question_text not in unique_questions:
                        question_pool.append(qa_pair)
                        unique_questions.add(question_text)
                        logger.info(f"Đã tạo câu hỏi thứ {len(question_pool)}/{num_questions}")

                doc_index += 1
                attempts += 1

            if len(question_pool) < num_questions:
                logger.warning(f"Chỉ tạo được {len(question_pool)}/{num_questions} câu hỏi do không đủ dữ liệu hoặc dữ liệu không đa dạng.")

            # Gán ID cuối cùng
            for i, q in enumerate(question_pool):
                q['id'] = f"q_{i + 1}"

            return question_pool

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def _generate_qa_from_document(self, doc: Dict[str, Any], existing_questions: List[str] = []) -> Optional[Dict[str, Any]]:
        """
        Tạo một cặp câu hỏi và câu trả lời (QA) từ một document bằng LLM.
        """
        text = doc.get('text', '')
        if not text or len(text.split()) < 10:  # Bỏ qua các document quá ngắn
            return None

        # Thêm danh sách câu hỏi đã có vào prompt để yêu cầu câu hỏi mới
        existing_questions_prompt = ""
        if existing_questions:
            questions_str = "\n- ".join(existing_questions)
            existing_questions_prompt = f"""
        LƯU Ý: Hãy tạo một câu hỏi KHÁC với các câu hỏi sau đây:
        - {questions_str}
        """

        # Cải tiến prompt để LLM tạo câu hỏi và trả lời chất lượng hơn
        prompt = f"""
        Bạn là một chuyên gia tạo câu hỏi và câu trả lời.
        Dựa vào nội dung văn bản dưới đây, hãy tạo ra MỘT câu hỏi hay và câu trả lời chính xác tương ứng.

        Văn bản:
        ---
        {text}
        ---
        {existing_questions_prompt}
        Yêu cầu:
        1.  Câu hỏi phải tập trung vào một chi tiết quan trọng, cụ thể trong văn bản.
        2.  Câu trả lời phải được rút ra trực tiếp từ văn bản và chính xác tuyệt đối.
        3.  Không thêm bất kỳ thông tin nào không có trong văn bản.
        4.  Trả về kết quả dưới dạng một đối tượng JSON duy nhất có cấu trúc:
            {{"question": "câu hỏi của bạn", "answer": "câu trả lời chính xác"}}
        """

        try:
            if self.llm_manager.is_available():
                response = self.llm_manager.generate_response(prompt, context="")

                # Parse a JSON response
                import json
                qa_data = json.loads(response)

                question = qa_data.get("question")
                answer = qa_data.get("answer")

                if not question or not answer:
                    logger.warning(f"LLM response is not a valid QA JSON: {response}")
                    return None

                return {
                    'question': question,
                    'correct_answer': answer,
                    'source_doc': doc,
                    'type': 'generated_by_llm'
                }
            else:
                logger.warning("LLM client is not available. Cannot generate QA from document.")
                return None

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response}")
            # Cố gắng parse theo kiểu cũ nếu JSON lỗi
            question, answer = self._parse_qa_response(response)
            if question != "Câu hỏi không xác định":
                return {
                    'question': question,
                    'correct_answer': answer,
                    'source_doc': doc,
                    'type': 'generated_by_llm_fallback_parse'
                }
            return None
        except Exception as e:
            logger.error(f"Error generating QA from doc (ID: {doc.get('id')}): {str(e)}")
            return None

    def _parse_qa_response(self, response: str) -> Tuple[str, str]:
        """Parse response từ LLM để lấy câu hỏi và đáp án"""
        lines = response.strip().split('\n')
        question = ""
        answer = ""

        for line in lines:
            if line.startswith("Câu hỏi:"):
                question = line.replace("Câu hỏi:", "").strip()
            elif line.startswith("Đáp án:"):
                answer = line.replace("Đáp án:", "").strip()

        if not question or not answer:
            # Fallback parsing
            if "?" in response:
                parts = response.split("?", 1)
                question = parts[0].strip() + "?"
                answer = parts[1].strip() if len(parts) > 1 else "Không xác định"

        return question or "Câu hỏi không xác định", answer or "Đáp án không xác định"



    def evaluate_answers(self, questions: List[Dict[str, Any]], user_answers: List[str]) -> Dict[str, Any]:
        """Đánh giá câu trả lời của user"""
        if len(questions) != len(user_answers):
            raise ValueError("Số câu hỏi và câu trả lời không khớp")

        results = []
        total_score = 0

        for i, (question, user_answer) in enumerate(zip(questions, user_answers)):
            correct_answer = question['correct_answer']

            # Tính điểm cho câu trả lời
            score = self._calculate_answer_score(user_answer, correct_answer)

            # Tạo feedback
            feedback = self._generate_feedback(user_answer, correct_answer, score)

            result = {
                'question_id': question['id'],
                'question': question['question'],
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'score': score,
                'feedback': feedback
            }

            results.append(result)
            total_score += score

        # Tính tỷ lệ phần trăm
        percentage = (total_score / len(questions)) * 100

        # Tạo đánh giá tổng thể
        overall_feedback = self._generate_overall_feedback(percentage)

        return {
            'results': results,
            'total_score': total_score,
            'max_score': len(questions),
            'percentage': round(percentage, 1),
            'overall_feedback': overall_feedback
        }

    def _calculate_answer_score(self, user_answer: str, correct_answer: str) -> float:
        """
        Tính điểm cho câu trả lời bằng cách sử dụng fuzzy string matching.
        """
        if not user_answer.strip():
            return 0.0

        # Chuẩn hóa cả hai chuỗi để so sánh tốt hơn
        user_answer_norm = user_answer.lower().strip()
        correct_answer_norm = correct_answer.lower().strip()

        # Sử dụng token_sort_ratio để so sánh không phân biệt thứ tự từ
        # và xử lý tốt các từ đồng nghĩa gần đúng.
        similarity_ratio = fuzz.token_sort_ratio(user_answer_norm, correct_answer_norm)

        # Chuyển đổi tỷ lệ (0-100) thành điểm (0.0-1.0)
        # Các ngưỡng này có thể được điều chỉnh để phù hợp hơn
        if similarity_ratio >= 95:
            return 1.0  # Gần như hoàn hảo
        elif similarity_ratio >= 85:
            return 0.8  # Rất giống
        elif similarity_ratio >= 70:
            return 0.6  # Khá giống
        elif similarity_ratio >= 50:
            return 0.4  # Có một phần liên quan
        else:
            return 0.0

    def _generate_feedback(self, user_answer: str, correct_answer: str, score: float) -> str:
        """Tạo feedback cho câu trả lời"""
        if score >= 0.9:
            return "Xuất sắc! Câu trả lời hoàn toàn chính xác."
        elif score >= 0.7:
            return "Tốt! Câu trả lời gần đúng."
        elif score >= 0.5:
            return "Khá! Câu trả lời có một phần đúng."
        elif score >= 0.3:
            return "Cần cải thiện. Câu trả lời chỉ đúng một phần nhỏ."
        else:
            return "Chưa chính xác. Hãy xem lại đáp án đúng."

    def _generate_overall_feedback(self, percentage: float) -> str:
        """Tạo đánh giá tổng thể"""
        if percentage >= 90:
            return "🎉 Xuất sắc! Bạn đã nắm vững kiến thức rất tốt!"
        elif percentage >= 70:
            return "👍 Tốt! Bạn đã hiểu phần lớn nội dung."
        elif percentage >= 50:
            return "📚 Khá! Bạn cần ôn lại một số phần."
        elif percentage >= 30:
            return "💪 Cần cố gắng thêm! Hãy đọc kỹ tài liệu hơn."
        else:
            return "📖 Hãy dành thời gian học lại từ đầu nhé!"
