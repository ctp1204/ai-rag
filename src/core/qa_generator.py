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

    def generate_questions(self, num_questions: int = 3, source_document: Optional[str] = None, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Tạo câu hỏi dựa trên dữ liệu có sẵn trong Vector DB.
        Có thể lọc theo tài liệu nguồn cụ thể và chọn nhà cung cấp LLM.
        """
        try:
            # Lấy tất cả documents từ vector database
            all_docs_raw = self.retrieval_engine.get_all_documents()

            if not all_docs_raw:
                logger.warning("Không có dữ liệu trong database, không thể tạo câu hỏi.")
                return []

            # Lọc tài liệu theo nguồn nếu được chỉ định
            if source_document:
                import os
                all_docs = [
                    doc for doc in all_docs_raw
                    # Chỉ tìm kiếm dựa trên trường 'source' cho chính xác
                    if os.path.basename(doc.get('source', '')) == source_document
                ]
                logger.info(f"Đã lọc, chỉ sử dụng {len(all_docs)} chunks từ tài liệu '{source_document}'.")
            else:
                all_docs = all_docs_raw
                logger.info("Sử dụng tất cả tài liệu để tạo câu hỏi.")


            if not all_docs:
                logger.warning(f"Không tìm thấy tài liệu nào cho nguồn '{source_document}'.")
                return []

            logger.info(f"Tìm thấy {len(all_docs)} documents trong database")

            # Tạo pool câu hỏi từ tất cả documents
            question_pool = []
            unique_questions = set()
            normalized_answers: List[str] = []  # dùng để loại trùng đáp án

            # Xáo trộn tài liệu để tăng tính ngẫu nhiên
            random.shuffle(all_docs)

            # Lặp cho đến khi có đủ câu hỏi hoặc không thể tạo thêm
            max_attempts = len(all_docs) * 4 # tăng số lần thử để đa dạng hơn
            attempts = 0
            doc_index = 0

            while len(question_pool) < num_questions and attempts < max_attempts:
                # Lấy doc và lặp vòng lại nếu cần
                doc = all_docs[doc_index % len(all_docs)]

                existing_answers_text = [q['correct_answer'] for q in question_pool]
                qa_pair = self._generate_qa_from_document(
                    doc,
                    existing_questions=list(unique_questions),
                    existing_answers=existing_answers_text,
                    provider=provider
                )

                if qa_pair:
                    question_text = qa_pair['question'].strip().lower()
                    answer_text = qa_pair['correct_answer']
                    answer_norm = self._normalize_text(answer_text)

                    if question_text in unique_questions:
                        pass
                    elif self._is_duplicate_answer(answer_norm, normalized_answers):
                        logger.info("Bỏ qua câu hỏi do đáp án trùng với câu đã có")
                    else:
                        question_pool.append(qa_pair)
                        unique_questions.add(question_text)
                        normalized_answers.append(answer_norm)
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

    def _generate_qa_from_document(self, doc: Dict[str, Any], existing_questions: List[str] = [], existing_answers: List[str] = [], provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
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

        existing_answers_prompt = ""
        if existing_answers:
            # Không yêu cầu trùng đáp án đã có
            existing_answers_prompt = f"""
        TRÁNH LẶP LẠI CÙNG ĐÁP ÁN với các đáp án đã có sau đây (đừng tạo câu hỏi dẫn tới cùng một câu trả lời):
        - """ + "\n        - ".join(existing_answers)


        # Cải tiến prompt để LLM tạo câu hỏi và trả lời chất lượng hơn
        prompt = f"""
        Bạn là một chuyên gia tạo câu hỏi và câu trả lời.
        Dựa vào nội dung văn bản dưới đây, hãy tạo ra MỘT câu hỏi hay và câu trả lời chính xác tương ứng.

        Văn bản:
        ---
        {text}
        ---
        {existing_questions_prompt}
        {existing_answers_prompt}
        Yêu cầu:
        1.  Câu hỏi phải tập trung vào một chi tiết quan trọng, cụ thể trong văn bản.
        2.  Câu trả lời phải là một câu hoàn chỉnh, giải thích đầy đủ cho câu hỏi và được rút ra trực tiếp từ văn bản.
        3.  Câu hỏi phải tự nhiên, không bắt đầu bằng các cụm từ như "Theo văn bản," hay "Dựa vào nội dung,".
        4.  Câu trả lời phải chính xác tuyệt đối và không thêm bất kỳ thông tin nào không có trong văn bản.
        5.  Trả về kết quả dưới dạng một đối tượng JSON duy nhất có cấu trúc:
            {{"question": "câu hỏi của bạn", "answer": "câu trả lời đầy đủ và chính xác"}}
        """

        try:
            if self.llm_manager.is_available():
                response = self.llm_manager.generate_response(prompt, context="", provider=provider)

                # Parse a JSON response
                import json
                # Thêm bước làm sạch để loại bỏ markdown code block
                clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip())
                qa_data = json.loads(clean_response)

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

    # -------------------------------
    # De-dup utilities for answers
    # -------------------------------
    def _normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để so sánh trùng lặp (lower, bỏ dấu câu, khoảng trắng thừa)."""
        t = text.lower().strip()
        t = re.sub(r"[\s\n\r]+", " ", t)
        t = re.sub(r"[\.;:,!?()\[\]{}'\"]+", "", t)
        return t

    def _is_duplicate_answer(self, answer_norm: str, existing_normalized_answers: List[str], threshold: int = 92) -> bool:
        """Kiểm tra đáp án trùng với danh sách đã có bằng fuzzy matching.
        threshold mặc định 92/100 để chỉ loại khi gần như cùng một ý.
        """
        for a in existing_normalized_answers:
            if a == answer_norm:
                return True
            if fuzz.token_sort_ratio(a, answer_norm) >= threshold:
                return True
        return False



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
        Tính điểm cho câu trả lời bằng cách sử dụng LLM để so sánh ngữ nghĩa.
        """
        if not user_answer.strip():
            return 0.0

        user_answer_norm = user_answer.lower().strip()
        correct_answer_norm = correct_answer.lower().strip()

        # Ưu tiên 1: Nếu câu trả lời của người dùng chứa toàn bộ đáp án đúng -> điểm tuyệt đối
        if correct_answer_norm in user_answer_norm:
            return 1.0

        # Ưu tiên 2: Nếu có sự trùng khớp hoàn hảo -> điểm tuyệt đối
        if user_answer_norm == correct_answer_norm:
            return 1.0

        prompt = f"""
        Bạn là một giám khảo chuyên nghiệp, nhiệm vụ của bạn là đánh giá câu trả lời của thí sinh.

        **Đáp án đúng (Answer Key):**
        {correct_answer}

        **Câu trả lời của thí sinh (Candidate's Answer):**
        {user_answer}

        **Yêu cầu:**
        Hãy đánh giá xem "Câu trả lời của thí sinh" có chứa đựng đầy đủ và chính xác ý nghĩa của "Đáp án đúng" hay không.
        - Nếu câu trả lời của thí sinh bao hàm toàn bộ ý của đáp án đúng (có thể diễn đạt khác hoặc chứa thêm thông tin bổ sung hữu ích), hãy cho điểm 1.0.
        - Nếu câu trả lời của thí sinh nắm được ý chính nhưng thiếu một vài chi tiết quan trọng, hãy cho điểm thấp hơn (ví dụ: 0.6-0.8).
        - Nếu câu trả lời sai hoặc thiếu nhiều ý, hãy cho điểm thấp (ví dụ: 0.0-0.4).

        **Chỉ trả về một đối tượng JSON duy nhất có dạng:** {{"score": <điểm số của bạn>}}
        Ví dụ: {{"score": 1.0}}
        """

        try:
            if self.llm_manager.is_available():
                response = self.llm_manager.generate_response(prompt, context="")
                import json
                # Thêm bước làm sạch để loại bỏ markdown code block
                clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip())
                result = json.loads(clean_response)
                score = float(result.get("score", 0.0))
                # Đảm bảo điểm số nằm trong khoảng 0.0 và 1.0
                return max(0.0, min(1.0, score))
            else:
                logger.warning("LLM not available for scoring, falling back to fuzzy matching.")
                # Fallback to fuzzy matching if LLM is down
                return fuzz.token_sort_ratio(user_answer.lower(), correct_answer.lower()) / 100.0
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Error parsing LLM score response: {e}. Response: {response}. Falling back to fuzzy matching.")
            # Fallback to fuzzy matching on error
            return fuzz.token_sort_ratio(user_answer.lower(), correct_answer.lower()) / 100.0

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
