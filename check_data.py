#!/usr/bin/env python3
"""
Kiểm tra dữ liệu trong database
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
    print("🔍 Kiểm tra dữ liệu trong RAG System")
    print("=" * 40)
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Check database info
    db_info = rag.get_database_info()
    print(f"📊 Tổng documents: {db_info['total_documents']}")
    print(f"🗄️ Vector DB type: {db_info['vector_db_type']}")
    
    if db_info['total_documents'] == 0:
        print("\n❌ Database đang TRỐNG!")
        print("\n💡 Để thêm dữ liệu, bạn có thể:")
        print("1. Chạy: python3 test_with_data.py")
        print("2. Hoặc truy cập http://localhost:8000 và upload file/text")
        print("3. Hoặc dùng API để thêm dữ liệu")
    else:
        print(f"\n✅ Database có {db_info['total_documents']} documents")
        print("\n📋 Nguồn dữ liệu:")
        for source, count in db_info['sources'].items():
            print(f"  - {source}: {count} chunks")
        
        print("\n🧪 Test câu hỏi mẫu:")
        test_query = "thông tin"
        result = rag.query(test_query, use_fallback=False)
        print(f"Query: '{test_query}'")
        print(f"Tìm thấy dữ liệu: {result['metadata']['has_local_data']}")
        if result['metadata']['has_local_data']:
            print(f"Response: {result['response'][:100]}...")

if __name__ == "__main__":
    main()
