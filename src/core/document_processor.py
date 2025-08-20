import os
import uuid
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from config import settings
import re

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.chunk_size = 256  # Kích thước chunk tối ưu hơn cho RAG
        self.chunk_overlap = 50  # Overlap khoảng 20% chunk_size

    def extract_text_from_file(self, file_path: str) -> str:
        """Trích xuất text từ file (PDF, DOCX, TXT)"""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _extract_from_pdf(self, file_path: str) -> str:
        """Trích xuất text từ PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Trích xuất text từ DOCX"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _extract_from_txt(self, file_path: str) -> str:
        """Trích xuất text từ TXT"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def chunk_qna_text(self, text: str) -> List[str]:
        """Chia text theo từng cặp Câu hỏi/Trả lời."""
        import re
        # Tách văn bản thành các khối dựa trên "Câu hỏi:"
        # (re.split giữ lại delimiter trong kết quả)
        parts = re.split(r'(Câu hỏi:|Câu-hỏi:)', text)
        chunks = []

        # Bỏ qua phần tử rỗng đầu tiên nếu có
        if not parts[0].strip():
            parts = parts[1:]

        # Ghép lại delimiter với nội dung của nó
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                chunk = parts[i] + parts[i+1]
                # Loại bỏ các dòng trống và khoảng trắng thừa
                cleaned_chunk = "\n".join(line.strip() for line in chunk.splitlines() if line.strip())
                chunks.append(cleaned_chunk)

        # Nếu không tìm thấy mẫu Q&A, quay về chunking thông thường
        if not chunks:
            return self.chunk_text_by_size(text)

        return chunks

    def chunk_text_by_size(self, text: str) -> List[str]:
        """Chia text thành các chunk nhỏ theo kích thước."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            if i + self.chunk_size >= len(words):
                break

        return chunks

    def chunk_interview_scenarios(self, text: str) -> List[str]:
        """Chia văn bản tài liệu kịch bản phỏng vấn thành các chunk logic."""
        # Tách toàn bộ văn bản dựa trên "Tình huống \d+:"
        # Điều này tạo ra một danh sách với phần giới thiệu chung ở đầu
        # và mỗi tình huống là một phần tử.
        base_splits = re.split(r'(Tình huống \d+:)', text)

        chunks = []
        # Xử lý phần giới thiệu (trước tình huống đầu tiên)
        intro_part = base_splits[0]

        # Chia phần giới thiệu thành "Bảng chấm điểm" và "Thang đánh giá"
        scoring_guide_match = re.search(r'Bảng chấm điểm \(Scoring Guide\)', intro_part)
        evaluation_scale_match = re.search(r'Thang đánh giá tổng quát:', intro_part)

        if scoring_guide_match and evaluation_scale_match:
            # Tách phần Bảng chấm điểm
            scoring_guide_text = intro_part[scoring_guide_match.start():evaluation_scale_match.start()]
            chunks.append(scoring_guide_text.strip())

            # Phần còn lại là Thang đánh giá
            evaluation_scale_text = intro_part[evaluation_scale_match.start():]
            # Tìm và tách phần Danh sách tình huống nếu có
            scenario_list_match = re.search(r'Danh sách tình huống', evaluation_scale_text)
            if scenario_list_match:
                chunks.append(evaluation_scale_text[:scenario_list_match.start()].strip())
            else:
                 chunks.append(evaluation_scale_text.strip())

        else:
            # Nếu không tìm thấy cấu trúc cụ thể, coi cả phần giới thiệu là một chunk
            if intro_part.strip():
                chunks.append(intro_part.strip())

        # Ghép lại các phần "Tình huống" đã tách
        for i in range(1, len(base_splits), 2):
            if i + 1 < len(base_splits):
                # Ghép "Tình huống X:" với nội dung của nó
                scenario_chunk = base_splits[i] + base_splits[i+1]
                chunks.append(scenario_chunk.strip())

        return [chunk for chunk in chunks if chunk]


    def chunk_text(self, text: str, file_path: str = "") -> List[str]:
        """
        Chia text thành các chunk.
        Ưu tiên chia theo định dạng Q&A cho file .txt; nếu không, dùng chunking ngữ nghĩa.
        """
        # Xử lý đặc biệt cho file kịch bản phỏng vấn
        if 'bo_tinh_huong_phong_van_it.docx' in file_path:
            return self.chunk_interview_scenarios(text)

        # Ưu tiên định dạng Q&A nếu có
        if file_path.endswith('.txt') and ("Câu hỏi:" in text or "Câu-hỏi:" in text):
            return self.chunk_qna_text(text)

        # Mặc định: chunk ngữ nghĩa theo đoạn/câu
        return self.chunk_text_semantic(text)

    def chunk_text_semantic(self, text: str, max_words: int = 220, overlap_words: int = 40) -> List[str]:
        """Chia text theo đoạn và câu, tích lũy đến ngưỡng từ; có overlap để giữ ngữ cảnh."""
        # Tách theo đoạn trống trước
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        buffer_sentences: List[str] = []
        buffer_len = 0

        def sentence_tokenize(paragraph: str) -> List[str]:
            # Tách câu đơn giản theo dấu câu. Có thể cải tiến bằng nltk nếu cần.
            return [s.strip() for s in re.split(r'(?<=[\.!?…])\s+', paragraph) if s.strip()]

        def flush_buffer():
            nonlocal buffer_sentences, buffer_len
            if buffer_sentences:
                chunks.append(" ".join(buffer_sentences))
                # Giữ overlap ở cấp độ câu (ước lượng 10 từ/câu)
                if overlap_words > 0:
                    approx_words_per_sentence = 10
                    keep_sentences = max(0, min(len(buffer_sentences), overlap_words // approx_words_per_sentence))
                    buffer_sentences = buffer_sentences[-keep_sentences:]
                    buffer_len = len(" ".join(buffer_sentences).split())
                else:
                    buffer_sentences = []
                    buffer_len = 0

        for para in paragraphs:
            sentences = sentence_tokenize(para)
            for s in sentences:
                words_in_s = len(s.split())
                # Nếu một câu quá dài, cắt theo size truyền thống để tránh vượt ngưỡng quá nhiều
                if words_in_s >= max_words * 1.2:
                    # Cắt thô câu dài thành các đoạn nhỏ hơn
                    long_words = s.split()
                    for i in range(0, len(long_words), max_words):
                        segment = " ".join(long_words[i:i + max_words])
                        if buffer_len + len(segment.split()) > max_words and buffer_sentences:
                            flush_buffer()
                        buffer_sentences.append(segment)
                        buffer_len += len(segment.split())
                else:
                    if buffer_len + words_in_s > max_words and buffer_sentences:
                        flush_buffer()
                    buffer_sentences.append(s)
                    buffer_len += words_in_s

        # Flush phần còn lại
        if buffer_sentences:
            chunks.append(" ".join(buffer_sentences))

        # Fallback nếu không tạo được chunk
        if not chunks:
            return self.chunk_text_by_size(text)

        return chunks

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Tạo embeddings cho list text"""
        # Hỗ trợ tiền tố hóa cho các model họ E5 nếu người dùng đổi model trong cấu hình
        model_name = settings.embedding_model.lower()
        if "e5" in model_name:
            processed_texts = [("passage: " + t) for t in texts]
            embeddings = self.embedding_model.encode(processed_texts)
        else:
            embeddings = self.embedding_model.encode(texts)
        return embeddings

    def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Xử lý document hoàn chỉnh: extract text -> chunk -> embed"""
        # Extract text
        text = self.extract_text_from_file(file_path)

        # Chunk text, truyền file_path để có thể xử lý đặc biệt
        chunks = self.chunk_text(text, file_path)

        # Create embeddings
        embeddings = self.create_embeddings(chunks)

        # Tạo document objects
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                'id': str(uuid.uuid4()),
                'text': chunk,
                'embedding': embedding,
                'source': file_path,
                'chunk_index': i,
                'metadata': metadata or {}
            }
            documents.append(doc)

        return documents

    def process_text_directly(self, text: str, source: str = "direct_input", metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Xử lý text trực tiếp (không từ file)"""
        # Chunk text
        chunks = self.chunk_text(text) # Không có file_path, sẽ dùng chunking theo size

        # Create embeddings
        embeddings = self.create_embeddings(chunks)

        # Tạo document objects
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                'id': str(uuid.uuid4()),
                'text': chunk,
                'embedding': embedding,
                'source': source,
                'chunk_index': i,
                'metadata': metadata or {}
            }
            documents.append(doc)

        return documents
