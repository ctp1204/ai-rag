#!/usr/bin/env python3
"""
Test script cho Q&A system
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.core.rag_pipeline import RAGPipeline
from src.core.qa_generator import QAGenerator

def main():
    print("🧪 Test Q&A System")
    print("=" * 50)
    
    # Initialize components
    print("📊 Khởi tạo RAG pipeline...")
    rag = RAGPipeline()
    
    print("🎯 Khởi tạo QA Generator...")
    qa_generator = QAGenerator(rag.retrieval_engine, rag.llm_manager)
    
    # Check database
    db_info = rag.get_database_info()
    print(f"📚 Database có {db_info['total_documents']} documents")
    
    if db_info['total_documents'] == 0:
        print("❌ Database trống! Hãy upload dữ liệu trước:")
        print("   python3 upload_sample_data.py")
        return
    
    # Test generate questions
    print("\n🎲 Tạo 3 câu hỏi...")
    try:
        questions = qa_generator.generate_questions(3)
        
        print(f"✅ Đã tạo {len(questions)} câu hỏi:")
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q['question']}")
            print(f"   Đáp án: {q['correct_answer']}")
            print(f"   Loại: {q['type']}")
        
        # Test evaluation
        print("\n🧪 Test đánh giá câu trả lời...")
        
        # Simulate user answers
        test_answers = [
            "Công ty được thành lập năm 2020",  # Correct
            "Không biết",                       # Wrong
            "Hệ thống quản lý"                 # Partial
        ]
        
        print("📝 Câu trả lời test:")
        for i, answer in enumerate(test_answers, 1):
            print(f"   {i}. {answer}")
        
        # Evaluate
        evaluation = qa_generator.evaluate_answers(questions, test_answers)
        
        print(f"\n📊 Kết quả đánh giá:")
        print(f"   Điểm: {evaluation['total_score']}/{evaluation['max_score']}")
        print(f"   Tỷ lệ: {evaluation['percentage']}%")
        print(f"   Nhận xét: {evaluation['overall_feedback']}")
        
        print(f"\n📋 Chi tiết từng câu:")
        for result in evaluation['results']:
            print(f"   Câu {result['question_id']}: {result['score']*100:.0f}% - {result['feedback']}")
        
        print("\n✅ Test hoàn thành!")
        print("\n💡 Để test trên web:")
        print("1. Chạy server: python3 -m uvicorn src.api.main:app --reload")
        print("2. Truy cập: http://localhost:8000/qa")
        
    except Exception as e:
        print(f"❌ Lỗi test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
