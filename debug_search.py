#!/usr/bin/env python3
"""
Debug script để kiểm tra search functionality
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
    print("🔧 Debug RAG Search")
    print("=" * 40)
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Add simple test data
    print("📚 Adding test data...")
    rag.add_document_from_text(
        "Công ty ABC Technology được thành lập năm 2020. Tổng giám đốc là Nguyễn Văn An.",
        "test_data",
        {"title": "Test Info"}
    )
    
    # Test direct search with no threshold
    print("\n🔍 Testing direct search...")
    query = "ABC Technology"
    
    # Get raw search results
    results = rag.retrieval_engine.search(query, k=5, min_score=0.0)
    print(f"Raw search results for '{query}': {len(results)}")
    
    for i, doc in enumerate(results, 1):
        score = doc.get('score', 0)
        print(f"  {i}. Score: {score:.4f} - {doc['text'][:50]}...")
    
    # Test with different queries
    test_queries = [
        "ABC Technology",
        "công ty",
        "thành lập",
        "Nguyễn Văn An",
        "2020"
    ]
    
    print(f"\n🔍 Testing multiple queries:")
    for query in test_queries:
        results = rag.retrieval_engine.search(query, k=3, min_score=0.0)
        print(f"'{query}': {len(results)} results")
        if results:
            best_score = max(r.get('score', 0) for r in results)
            print(f"  Best score: {best_score:.4f}")

if __name__ == "__main__":
    main()
