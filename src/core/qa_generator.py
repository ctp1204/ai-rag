import random
import re
from typing import List, Dict, Any, Tuple
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
                logger.warning("Không có dữ liệu trong database, sử dụng câu hỏi fallback")
                return self._get_fallback_questions(num_questions)

            logger.info(f"Tìm thấy {len(all_docs)} documents trong database")

            # Tạo pool câu hỏi từ tất cả documents
            question_pool = []

            # Phân tích từng document để tạo câu hỏi
            for doc in all_docs:
                doc_questions = self._extract_questions_from_document(doc)
                question_pool.extend(doc_questions)

            logger.info(f"Tạo được {len(question_pool)} câu hỏi từ dữ liệu thực")

            # Nếu có đủ câu hỏi từ dữ liệu thực, chọn ngẫu nhiên
            if len(question_pool) >= num_questions:
                selected_questions = random.sample(question_pool, num_questions)
                # Gán ID cho câu hỏi
                for i, q in enumerate(selected_questions):
                    q['id'] = f"q_{i + 1}"
                return selected_questions

            # Nếu không đủ, kết hợp với LLM generation
            questions = question_pool.copy()

            # Tạo thêm câu hỏi bằng LLM từ documents còn lại
            remaining_docs = [doc for doc in all_docs if not any(q.get('source_doc', {}).get('id') == doc.get('id') for q in questions)]

            for doc in remaining_docs:
                if len(questions) >= num_questions:
                    break

                llm_question = self._generate_question_from_doc(doc, len(questions) + 1)
                if llm_question:
                    questions.append(llm_question)

            # Nếu vẫn không đủ, dùng fallback
            while len(questions) < num_questions:
                fallback_question = self._get_fallback_question(len(questions) + 1)
                questions.append(fallback_question)

            # Gán ID cho tất cả câu hỏi
            for i, q in enumerate(questions):
                q['id'] = f"q_{i + 1}"

            return questions[:num_questions]

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return self._get_fallback_questions(num_questions)

    def _extract_questions_from_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Trích xuất câu hỏi từ một document cụ thể"""
        text = doc.get('text', '')
        if not text:
            return []

        # Phân tích text để tạo câu hỏi
        question_templates = self._analyze_text_for_questions(text)

        questions = []
        for template in question_templates:
            question = {
                'question': template['question'],
                'correct_answer': template['answer'],
                'source_doc': doc,
                'type': 'extracted_from_data'
            }
            questions.append(question)

        return questions

    def _generate_question_from_doc(self, doc: Dict[str, Any], question_num: int) -> Dict[str, Any]:
        """Tạo câu hỏi từ một document cụ thể"""
        text = doc['text']

        # Tạo prompt để LLM sinh câu hỏi
        prompt = f"""
        Dựa vào đoạn văn bản sau, hãy tạo 1 câu hỏi cụ thể và câu trả lời chính xác:

        Văn bản: {text}

        Yêu cầu:
        - Câu hỏi phải rõ ràng, cụ thể
        - Câu trả lời phải có trong văn bản
        - Trả về định dạng:
        Câu hỏi: [câu hỏi]
        Đáp án: [câu trả lời chính xác]
        """

        try:
            if self.llm_manager.is_available():
                response = self.llm_manager.generate_response(prompt, context="")
                question, answer = self._parse_qa_response(response)

                return {
                    'id': f"q_{question_num}",
                    'question': question,
                    'correct_answer': answer,
                    'source_doc': doc,
                    'type': 'generated'
                }
            else:
                # Fallback: tạo câu hỏi từ template
                return self._generate_template_question_from_doc(doc, question_num)

        except Exception as e:
            logger.error(f"Error generating question from doc: {str(e)}")
            return self._generate_template_question_from_doc(doc, question_num)

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

    def _generate_template_question_from_doc(self, doc: Dict[str, Any], question_num: int) -> Dict[str, Any]:
        """Tạo câu hỏi từ template dựa trên document"""
        text = doc['text']

        # Tìm thông tin cụ thể trong text
        if "năm" in text and any(year in text for year in ["2020", "2021", "2022", "2023", "2024"]):
            years = re.findall(r'\b(20\d{2})\b', text)
            if years:
                return {
                    'id': f"q_{question_num}",
                    'question': f"Theo tài liệu, năm nào được đề cập trong thông tin?",
                    'correct_answer': years[0],
                    'source_doc': doc,
                    'type': 'template'
                }

        # Template mặc định
        return {
            'id': f"q_{question_num}",
            'question': f"Nội dung chính của đoạn văn bản này là gì?",
            'correct_answer': text[:100] + "..." if len(text) > 100 else text,
            'source_doc': doc,
            'type': 'template'
        }

    def _generate_template_question(self, question_num: int) -> Dict[str, Any]:
        """Tạo câu hỏi từ dữ liệu thực tế trong database"""
        try:
            # Lấy tất cả documents từ database
            all_docs = self.retrieval_engine.get_all_documents()

            if not all_docs:
                return self._get_fallback_question(question_num)

            # Chọn document ngẫu nhiên
            import random
            selected_doc = random.choice(all_docs)

            # Tạo câu hỏi từ document thực tế
            return self._create_question_from_real_data(selected_doc, question_num)

        except Exception as e:
            logger.error(f"Error generating template question: {str(e)}")
            return self._get_fallback_question(question_num)

    def _create_question_from_real_data(self, doc: Dict[str, Any], question_num: int) -> Dict[str, Any]:
        """Tạo câu hỏi từ dữ liệu thực tế"""
        text = doc.get('text', '')

        # Phân tích text để tạo câu hỏi phù hợp
        question_templates = self._analyze_text_for_questions(text)

        if question_templates:
            # Chọn template phù hợp nhất
            selected_template = question_templates[0]
            return {
                'id': f"q_{question_num}",
                'question': selected_template['question'],
                'correct_answer': selected_template['answer'],
                'source_doc': doc,
                'type': 'real_data'
            }
        else:
            # Fallback: tạo câu hỏi tổng quát từ text
            return {
                'id': f"q_{question_num}",
                'question': f"Theo tài liệu, thông tin nào sau đây được đề cập?",
                'correct_answer': text[:200] + "..." if len(text) > 200 else text,
                'source_doc': doc,
                'type': 'real_data_general'
            }

    def _analyze_text_for_questions(self, text: str) -> List[Dict[str, str]]:
        """Phân tích text để tạo câu hỏi cụ thể"""
        questions = []
        text_lower = text.lower()

        # Pattern 1: Năm thành lập
        years = re.findall(r'(?:thành lập|được thành lập|ra đời).*?(\d{4})', text, re.IGNORECASE)
        if years:
            questions.append({
                'question': "Công ty được thành lập vào năm nào?",
                'answer': years[0]
            })

        # Pattern 2: Địa chỉ trụ sở chính
        address_match = re.search(r'địa chỉ trụ sở chính[:\s]*([^Đ\n]+?)(?=\s*Điện thoại|$)', text, re.IGNORECASE)
        if address_match:
            address = address_match.group(1).strip()
            # Clean up address - remove trailing numbers/phone if captured
            address = re.sub(r'\s*\d{3}-\d{4}-\d{4}.*$', '', address)
            questions.append({
                'question': "Địa chỉ của công ty là gì?",
                'answer': address
            })

        # Pattern 3: Số điện thoại
        phone_matches = re.findall(r'(?:điện thoại|phone|tel)[:\s]*([0-9\-\s\+\(\)]+)', text, re.IGNORECASE)
        if phone_matches:
            questions.append({
                'question': "Số điện thoại liên hệ của công ty là gì?",
                'answer': phone_matches[0].strip()
            })

        # Pattern 4: Email
        email_matches = re.findall(r'(?:email|mail)[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text, re.IGNORECASE)
        if email_matches:
            questions.append({
                'question': "Địa chỉ email của công ty là gì?",
                'answer': email_matches[0]
            })

        # Pattern 5: Giá cả sản phẩm cụ thể
        price_matches = re.findall(r'giá[:\s]*([0-9,\.\s]+\s*VNĐ[^-\n]*)', text, re.IGNORECASE)
        if price_matches:
            # Lấy giá đầu tiên tìm được
            questions.append({
                'question': "Giá của sản phẩm/dịch vụ là bao nhiêu?",
                'answer': price_matches[0].strip()
            })

        # Pattern 6: Tên CEO/Tổng giám đốc
        ceo_match = re.search(r'tổng giám đốc[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ][a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+?)(?:\s*-|\n|$)', text, re.IGNORECASE)
        if ceo_match:
            questions.append({
                'question': "Ai là người đứng đầu công ty?",
                'answer': ceo_match.group(1).strip()
            })

        # Pattern 7: Sản phẩm/Dịch vụ chính
        if 'hệ thống quản lý' in text_lower:
            # Tìm các hệ thống cụ thể
            systems = re.findall(r'hệ thống quản lý ([^(]+)', text, re.IGNORECASE)
            if systems:
                # Lấy tên hệ thống, loại bỏ phần trong ngoặc
                clean_systems = []
                for system in systems[:3]:
                    clean_name = re.sub(r'\s*\([^)]*\)', '', system).strip()
                    if clean_name:
                        clean_systems.append(clean_name)

                if clean_systems:
                    system_list = ', '.join(clean_systems)
                    questions.append({
                        'question': "Công ty cung cấp những sản phẩm/dịch vụ gì?",
                        'answer': f"Các hệ thống quản lý {system_list}"
                    })

        # Pattern 8: Số lượng nhân viên
        employee_matches = re.findall(r'đội ngũ nhân viên[:\s]*([0-9,\.\s]+\s*người)', text, re.IGNORECASE)
        if employee_matches:
            questions.append({
                'question': "Công ty có bao nhiêu nhân viên?",
                'answer': employee_matches[0].strip()
            })

        # Pattern 9: Khách hàng tiêu biểu
        if 'khách hàng tiêu biểu' in text_lower:
            # Tìm các công ty khách hàng cụ thể
            customer_companies = re.findall(r'(?:công ty|chuỗi|ngân hàng)\s+([A-Z][A-Za-z\s]+)(?:\s+(?:manufacturing|retail|bank))?', text, re.IGNORECASE)
            if customer_companies:
                # Lấy 2-3 khách hàng đầu tiên
                customers = customer_companies[:3]
                customer_list = ', '.join(customers)
                questions.append({
                    'question': "Khách hàng chính của công ty là ai?",
                    'answer': f"Các doanh nghiệp như {customer_list}"
                })

        # Pattern 10: Chính sách bảo hành
        if 'chính sách bảo hành' in text_lower:
            warranty_info = re.search(r'bảo hành phần mềm[:\s]*([^-\n]+)', text, re.IGNORECASE)
            if warranty_info:
                questions.append({
                    'question': "Chính sách bảo hành của công ty như thế nào?",
                    'answer': warranty_info.group(1).strip()
                })

        # Pattern 11: Thời gian làm việc
        working_time_matches = re.findall(r'(?:giờ làm việc|thời gian làm việc)[:\s]*([^.]+)', text, re.IGNORECASE)
        if working_time_matches:
            questions.append({
                'question': "Thời gian làm việc của công ty là gì?",
                'answer': working_time_matches[0].strip()
            })

        # Pattern 12: Website
        website_matches = re.findall(r'(?:website|web|trang web)[:\s]*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text, re.IGNORECASE)
        if website_matches:
            questions.append({
                'question': "Website của công ty là gì?",
                'answer': website_matches[0]
            })

        return questions

    def _get_fallback_question(self, question_num: int) -> Dict[str, Any]:
        """Câu hỏi dự phòng khi không có dữ liệu"""
        fallback_templates = [
            {
                'question': "Hệ thống này được sử dụng để làm gì?",
                'answer': "Hệ thống RAG được sử dụng để trả lời câu hỏi dựa trên dữ liệu đã được tải lên"
            },
            {
                'question': "RAG là viết tắt của gì?",
                'answer': "RAG là viết tắt của Retrieval-Augmented Generation"
            },
            {
                'question': "Vector database được sử dụng để làm gì?",
                'answer': "Vector database được sử dụng để lưu trữ và tìm kiếm embeddings của documents"
            }
        ]

        template = fallback_templates[(question_num - 1) % len(fallback_templates)]
        return {
            'id': f"q_{question_num}",
            'question': template['question'],
            'correct_answer': template['answer'],
            'source_doc': None,
            'type': 'fallback'
        }

    def _get_fallback_questions(self, num_questions: int) -> List[Dict[str, Any]]:
        """Câu hỏi dự phòng khi không thể tạo từ dữ liệu"""
        return [self._generate_template_question(i + 1) for i in range(num_questions)]

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
        """Tính điểm cho câu trả lời (0.0 - 1.0)"""
        if not user_answer.strip():
            return 0.0

        user_answer = user_answer.lower().strip()
        correct_answer = correct_answer.lower().strip()

        # Exact match
        if user_answer == correct_answer:
            return 1.0

        # Partial match
        if user_answer in correct_answer or correct_answer in user_answer:
            return 0.7

        # Keyword matching
        user_words = set(user_answer.split())
        correct_words = set(correct_answer.split())

        if user_words and correct_words:
            intersection = user_words.intersection(correct_words)
            union = user_words.union(correct_words)
            similarity = len(intersection) / len(union)

            if similarity >= 0.5:
                return 0.5
            elif similarity >= 0.3:
                return 0.3

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
