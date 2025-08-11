import os
import uuid
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from config import settings

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.chunk_size = 500  # Kích thước chunk text
        self.chunk_overlap = 50  # Overlap giữa các chunk
    
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
    
    def chunk_text(self, text: str) -> List[str]:
        """Chia text thành các chunk nhỏ"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Tạo embeddings cho list text"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings
    
    def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Xử lý document hoàn chỉnh: extract text -> chunk -> embed"""
        # Extract text
        text = self.extract_text_from_file(file_path)
        
        # Chunk text
        chunks = self.chunk_text(text)
        
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
        chunks = self.chunk_text(text)
        
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
