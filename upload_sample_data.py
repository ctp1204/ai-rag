#!/usr/bin/env python3
"""
Script để upload dữ liệu từ sample_data.txt vào RAG system
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
    print("🚀 Upload dữ liệu từ sample_data.txt")
    print("=" * 50)
    
    # Kiểm tra file sample_data.txt có tồn tại không
    sample_file = "sample_data.txt"
    if not os.path.exists(sample_file):
        print(f"❌ Không tìm thấy file {sample_file}")
        print("💡 Hãy đảm bảo file sample_data.txt có trong thư mục hiện tại")
        return
    
    # Initialize RAG pipeline
    print("📊 Khởi tạo RAG pipeline...")
    rag = RAGPipeline()
    
    # Kiểm tra database hiện tại
    print("🔍 Kiểm tra database hiện tại...")
    db_info = rag.get_database_info()
    print(f"  - Tổng documents hiện tại: {db_info['total_documents']}")
    
    if db_info['total_documents'] > 0:
        choice = input("\n⚠️ Database đã có dữ liệu. Bạn có muốn xóa hết và thêm mới? (y/N): ")
        if choice.lower() == 'y':
            print("🗑️ Xóa dữ liệu cũ...")
            rag.clear_database()
            print("✅ Đã xóa dữ liệu cũ")
        else:
            print("📝 Sẽ thêm dữ liệu mới vào database hiện tại")
    
    # Upload file sample_data.txt
    print(f"\n📚 Đang upload file {sample_file}...")
    
    metadata = {
        "title": "Thông tin Công ty ABC Technology",
        "type": "company_info",
        "source_file": sample_file,
        "description": "Dữ liệu chi tiết về công ty ABC Technology bao gồm sản phẩm, dịch vụ, nhân sự và chính sách"
    }
    
    result = rag.add_document_from_file(sample_file, metadata)
    
    if result['success']:
        print(f"✅ Upload thành công!")
        print(f"  - File: {result['file_path']}")
        print(f"  - Số chunks được tạo: {result['chunks_added']}")
    else:
        print(f"❌ Upload thất bại: {result['error']}")
        return
    
    # Kiểm tra database sau khi upload
    print("\n📊 Thông tin database sau khi upload:")
    db_info = rag.get_database_info()
    print(f"  - Tổng documents: {db_info['total_documents']}")
    print(f"  - Vector DB type: {db_info['vector_db_type']}")
    
    if db_info['sources']:
        print("  - Nguồn dữ liệu:")
        for source, count in db_info['sources'].items():
            print(f"    + {source}: {count} chunks")
    
    # Test một số câu hỏi mẫu
    print("\n🧪 Test một số câu hỏi mẫu:")
    
    test_questions = [
        "Công ty ABC Technology được thành lập năm nào?",
        "Số điện thoại của công ty là gì?",
        "Giá của hệ thống quản lý bán hàng là bao nhiêu?",
        "Ai là Tổng giám đốc?",
        "Địa chỉ trụ sở chính ở đâu?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ❓ {question}")
        
        # Test với dữ liệu local only (không dùng API)
        result = rag.query(question, use_fallback=False)
        metadata = result['metadata']
        
        if metadata['has_local_data']:
            print(f"   ✅ Tìm thấy dữ liệu: {result['response']}")
        else:
            print(f"   ❌ Không tìm thấy dữ liệu liên quan")
    
    print("\n🎉 Hoàn thành!")
    print("\n💡 Bây giờ bạn có thể:")
    print("1. Truy cập http://localhost:8000 để chat với AI")
    print("2. Hỏi các câu hỏi về công ty ABC Technology")
    print("3. Upload thêm dữ liệu khác nếu cần")
    
    print("\n📝 Một số câu hỏi gợi ý:")
    print("- 'Công ty có những sản phẩm gì?'")
    print("- 'Chính sách bảo hành như thế nào?'")
    print("- 'Khách hàng tiêu biểu của công ty?'")
    print("- 'Thông tin liên hệ của công ty?'")

if __name__ == "__main__":
    main()
