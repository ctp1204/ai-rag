#!/usr/bin/env python3
"""
Demo script để test RAG system
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
    print("🚀 RAG System Demo")
    print("=" * 50)

    # Initialize RAG pipeline
    print("📊 Initializing RAG pipeline...")
    rag = RAGPipeline()

    # Health check
    print("\n🔍 Health check:")
    health = rag.health_check()
    print(f"  - Retrieval engine: {health['retrieval_engine']}")
    print(f"  - LLM available: {health['llm_available']}")
    print(f"  - Vector DB: {health['settings']['vector_db_type']}")
    print(f"  - Embedding model: {health['settings']['embedding_model']}")

    # Add sample data
    print("\n📚 Adding sample data...")

    sample_texts = [
        {
            "text": """
            Python là một ngôn ngữ lập trình bậc cao, được thiết kế với triết lý nhấn mạnh vào khả năng đọc của code.
            Python hỗ trợ nhiều mô hình lập trình như lập trình hướng đối tượng, lập trình thủ tục, và lập trình hàm.
            Python được sử dụng rộng rãi trong phát triển web, khoa học dữ liệu, trí tuệ nhân tạo, và tự động hóa.
            """,
            "source": "python_info",
            "title": "Thông tin về Python"
        },
        {
            "text": """
            Machine Learning (Học máy) là một nhánh của trí tuệ nhân tạo (AI) tập trung vào việc xây dựng các hệ thống
            có thể học hỏi từ dữ liệu mà không cần được lập trình cụ thể cho từng tác vụ.
            Các thuật toán machine learning xây dựng mô hình toán học dựa trên dữ liệu huấn luyện để đưa ra dự đoán
            hoặc quyết định mà không cần được lập trình rõ ràng để thực hiện tác vụ đó.
            """,
            "source": "ml_info",
            "title": "Machine Learning cơ bản"
        },
        {
            "text": """
            FastAPI là một web framework hiện đại, nhanh (hiệu suất cao) để xây dựng API với Python 3.7+
            dựa trên standard Python type hints. FastAPI được thiết kế để dễ sử dụng và học,
            nhanh để code, sẵn sàng cho production. Nó tự động tạo ra interactive API documentation.
            """,
            "source": "fastapi_info",
            "title": "FastAPI Framework"
        }
    ]

    for i, sample in enumerate(sample_texts, 1):
        result = rag.add_document_from_text(
            text=sample["text"],
            source=sample["source"],
            metadata={"title": sample["title"]}
        )
        print(f"  {i}. {sample['title']}: {result['chunks_added']} chunks added")

    # Database info
    print("\n📊 Database info:")
    db_info = rag.get_database_info()
    print(f"  - Total documents: {db_info['total_documents']}")
    print(f"  - Sources: {list(db_info['sources'].keys())}")

    # Test queries
    print("\n❓ Testing queries:")

    test_queries = [
        "Python là gì?",
        "Machine Learning hoạt động như thế nào?",
        "FastAPI có ưu điểm gì?",
        "Làm thế nào để học lập trình?",  # This should trigger fallback if enabled
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")

        # Test with fallback disabled first
        result = rag.query(query, use_fallback=False)
        metadata = result['metadata']

        print(f"   - Has local data: {metadata['has_local_data']}")
        print(f"   - Source: {metadata['source']}")
        print(f"   - Response: {result['response'][:100]}...")

        if not metadata['has_local_data']:
            print("   - (No local data found, would use fallback if enabled)")

    # Search test
    print("\n🔍 Testing search functionality:")
    search_results = rag.search_documents("Python programming", k=3)
    print(f"Found {len(search_results)} relevant documents for 'Python programming':")
    for j, doc in enumerate(search_results, 1):
        score = doc.get('score', 0) * 100
        print(f"  {j}. Score: {score:.1f}% - {doc['text'][:80]}...")

    # Debug: Test with lower threshold
    print("\n🔧 Debug: Testing with very low threshold...")
    debug_results = rag.retrieval_engine.search("Python", k=5, min_score=0.0)
    print(f"Found {len(debug_results)} documents with min_score=0.0:")
    for j, doc in enumerate(debug_results, 1):
        score = doc.get('score', 0) * 100
        print(f"  {j}. Score: {score:.1f}% - {doc['text'][:80]}...")

    # Debug: Direct FAISS test
    print("\n🔧 Debug: Direct FAISS test...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(["Python"])
    print(f"Query embedding shape: {query_emb.shape}")

    # Direct search in vector database
    vector_db = rag.retrieval_engine.vector_db
    print(f"Index total: {vector_db.index.ntotal}")
    if vector_db.index.ntotal > 0:
        raw_results = vector_db.search(query_emb[0], k=5)
        print(f"Raw FAISS results: {len(raw_results)}")
        for j, doc in enumerate(raw_results, 1):
            score = doc.get('score', 0)
            print(f"  {j}. Raw Score: {score:.4f} - {doc['text'][:50]}...")

    print("\n✅ Demo completed!")
    print("\nTo start the web interface, run: python main.py")

if __name__ == "__main__":
    main()
