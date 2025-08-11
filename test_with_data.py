#!/usr/bin/env python3
"""
Script để test hệ thống RAG với dữ liệu mẫu
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.core.rag_pipeline import RAGPipeline

def main():
    print("🚀 Testing RAG System với dữ liệu công ty ABC Technology")
    print("=" * 60)
    
    # Initialize RAG pipeline
    print("📊 Khởi tạo RAG pipeline...")
    rag = RAGPipeline()
    
    # Clear existing data
    print("🗑️ Xóa dữ liệu cũ...")
    rag.clear_database()
    
    # Add sample data from file
    print("📚 Thêm dữ liệu mẫu từ file...")
    result = rag.add_document_from_file(
        "sample_data.txt",
        metadata={"title": "Thông tin Công ty ABC Technology", "type": "company_info"}
    )
    
    if result['success']:
        print(f"✅ Đã thêm {result['chunks_added']} chunks từ file sample_data.txt")
    else:
        print(f"❌ Lỗi: {result['error']}")
        return
    
    # Database info
    print("\n📊 Thông tin database:")
    db_info = rag.get_database_info()
    print(f"  - Tổng documents: {db_info['total_documents']}")
    print(f"  - Vector DB: {db_info['vector_db_type']}")
    
    # Test queries
    print("\n❓ Test các câu hỏi về công ty ABC Technology:")
    
    test_questions = [
        "Công ty ABC Technology được thành lập năm nào?",
        "Sản phẩm chính của công ty ABC Technology là gì?",
        "Giá của hệ thống quản lý bán hàng là bao nhiêu?",
        "Ai là Tổng giám đốc của công ty?",
        "Công ty có bao nhiêu nhân viên?",
        "Địa chỉ trụ sở chính của công ty ở đâu?",
        "Chính sách bảo hành như thế nào?",
        "Khách hàng tiêu biểu của công ty là ai?",
        "Công ty có những chứng nhận gì?",
        "Giờ làm việc của công ty?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ {question}")
        
        # Query with fallback disabled to test local data only
        result = rag.query(question, use_fallback=False)
        metadata = result['metadata']
        
        print(f"   📊 Có dữ liệu local: {metadata['has_local_data']}")
        print(f"   🔍 Nguồn: {metadata['source']}")
        
        if metadata['has_local_data']:
            print(f"   ✅ Trả lời: {result['response']}")
        else:
            print(f"   ❌ Không tìm thấy: {result['response'][:100]}...")
        
        print("-" * 50)
    
    # Test search functionality
    print("\n🔍 Test tìm kiếm documents:")
    search_terms = ["giá cả", "nhân viên", "sản phẩm", "khách hàng"]
    
    for term in search_terms:
        results = rag.search_documents(term, k=3)
        print(f"\n🔍 Tìm kiếm '{term}': {len(results)} kết quả")
        for j, doc in enumerate(results, 1):
            score = doc.get('score', 0) * 100
            print(f"  {j}. Score: {score:.1f}% - {doc['text'][:80]}...")
    
    print("\n✅ Test hoàn thành!")
    print("\n💡 Bạn có thể:")
    print("1. Chạy 'python3 main.py' để khởi động web interface")
    print("2. Truy cập http://localhost:8000 để chat với AI")
    print("3. Upload thêm dữ liệu khác để mở rộng kiến thức")

if __name__ == "__main__":
    main()
