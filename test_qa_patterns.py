#!/usr/bin/env python3
"""
Test script để kiểm tra các pattern regex trong Q&A generator
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.core.qa_generator import QAGenerator
from src.core.rag_pipeline import RAGPipeline

def test_patterns():
    print("🧪 Test Q&A Patterns")
    print("=" * 50)
    
    # Sample text từ sample_data.txt
    sample_texts = [
        # Text 1: Thông tin cơ bản
        "Công ty ABC Technology được thành lập năm 2020, chuyên phát triển các giải pháp công nghệ thông tin cho doanh nghiệp.",
        
        # Text 2: Sản phẩm
        "2.1 Hệ thống quản lý bán hàng (Sales Management System) - Quản lý khách hàng (CRM) - Giá: 5,000,000 VNĐ/năm cho gói cơ bản",
        
        # Text 3: Nhân sự
        "Tổng giám đốc: Nguyễn Văn An - 15 năm kinh nghiệm trong lĩnh vực IT. Đội ngũ nhân viên: 45 người",
        
        # Text 4: Liên hệ
        "Địa chỉ trụ sở chính: 123 Đường Láng, Đống Đa, Hà Nội Điện thoại: 024-1234-5678 Email: info@abctech.vn",
        
        # Text 5: Khách hàng
        "4.1 Công ty XYZ Manufacturing - Triển khai hệ thống quản lý sản xuất 4.2 Chuỗi cửa hàng DEF Retail - Triển khai hệ thống POS"
    ]
    
    # Initialize QA Generator
    rag = RAGPipeline()
    qa_generator = QAGenerator(rag.retrieval_engine, rag.llm_manager)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Test Text {i} ---")
        print(f"Text: {text[:100]}...")
        
        # Test pattern analysis
        questions = qa_generator._analyze_text_for_questions(text)
        
        if questions:
            print(f"✅ Tìm thấy {len(questions)} câu hỏi:")
            for j, q in enumerate(questions, 1):
                print(f"  {j}. Q: {q['question']}")
                print(f"     A: {q['answer']}")
        else:
            print("❌ Không tìm thấy câu hỏi nào")
    
    print(f"\n🔍 Test với dữ liệu thực từ sample_data.txt:")
    
    # Read actual sample_data.txt
    try:
        with open('sample_data.txt', 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print(f"📄 Đọc file sample_data.txt ({len(full_text)} ký tự)")
        
        # Analyze full text
        all_questions = qa_generator._analyze_text_for_questions(full_text)
        
        print(f"✅ Tổng cộng tìm thấy {len(all_questions)} câu hỏi từ file:")
        
        # Group by question type
        question_types = {}
        for q in all_questions:
            question = q['question']
            if question not in question_types:
                question_types[question] = []
            question_types[question].append(q['answer'])
        
        for question, answers in question_types.items():
            print(f"\n📝 {question}")
            for answer in answers:
                print(f"   ✅ {answer}")
        
        # Test specific patterns
        print(f"\n🎯 Test các pattern cụ thể:")
        
        import re
        
        # Test năm thành lập
        years = re.findall(r'(?:thành lập|được thành lập|ra đời).*?(\d{4})', full_text, re.IGNORECASE)
        print(f"📅 Năm thành lập: {years}")
        
        # Test số nhân viên
        employees = re.findall(r'đội ngũ nhân viên[:\s]*([0-9,\.\s]+\s*người)', full_text, re.IGNORECASE)
        print(f"👥 Số nhân viên: {employees}")
        
        # Test địa chỉ
        address = re.search(r'địa chỉ trụ sở chính[:\s]*([^Đ\n]+)', full_text, re.IGNORECASE)
        print(f"📍 Địa chỉ: {address.group(1).strip() if address else 'Không tìm thấy'}")
        
        # Test email
        email = re.findall(r'(?:email|mail)[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', full_text, re.IGNORECASE)
        print(f"📧 Email: {email}")
        
        # Test điện thoại
        phone = re.findall(r'(?:điện thoại|phone|tel)[:\s]*([0-9\-\s\+\(\)]+)', full_text, re.IGNORECASE)
        print(f"📞 Điện thoại: {phone}")
        
        # Test CEO
        ceo = re.findall(r'(?:ceo|giám đốc|tổng giám đốc|chủ tịch)[:\s]*([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ][a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+)', full_text, re.IGNORECASE)
        print(f"👔 CEO: {ceo}")
        
        # Test giá
        prices = re.findall(r'giá[:\s]*([0-9,\.\s]+\s*VNĐ[^-\n]*)', full_text, re.IGNORECASE)
        print(f"💰 Giá: {prices[:3]}")  # Show first 3
        
    except FileNotFoundError:
        print("❌ Không tìm thấy file sample_data.txt")

if __name__ == "__main__":
    test_patterns()
