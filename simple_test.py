#!/usr/bin/env python3
"""
Simple test without API key
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
    print("🚀 Simple RAG Test (No API needed)")
    print("=" * 40)
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Clear and add simple data
    rag.clear_database()
    
    print("📚 Adding simple test data...")
    result = rag.add_document_from_text(
        "Công ty ABC được thành lập năm 2020. Giám đốc là Nguyễn Văn An. Công ty có 50 nhân viên.",
        "test_company"
    )
    print(f"Added {result['chunks_added']} chunks")
    
    # Test with very simple queries and no API fallback
    test_queries = [
        "ABC",
        "2020", 
        "Nguyễn Văn An",
        "nhân viên"
    ]
    
    print("\n❓ Testing simple queries (no API fallback):")
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Search with no threshold
        results = rag.retrieval_engine.search(query, k=3, min_score=0.0)
        print(f"  Raw results: {len(results)}")
        
        if results:
            for i, doc in enumerate(results, 1):
                score = doc.get('score', 0)
                print(f"    {i}. Score: {score:.4f} - {doc['text'][:50]}...")
        
        # Test RAG query without fallback
        rag_result = rag.query(query, use_fallback=False)
        print(f"  RAG found data: {rag_result['metadata']['has_local_data']}")
        if rag_result['metadata']['has_local_data']:
            print(f"  Response: {rag_result['response'][:100]}...")

if __name__ == "__main__":
    main()
