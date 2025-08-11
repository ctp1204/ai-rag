#!/usr/bin/env python3
"""
Debug FAISS directly
"""

import sys
import os
from pathlib import Path
import numpy as np
import faiss

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from sentence_transformers import SentenceTransformer

def main():
    print("🔧 Debug FAISS directly")
    print("=" * 40)
    
    # Initialize embedding model
    print("📊 Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create test data
    texts = [
        "Công ty ABC được thành lập năm 2020",
        "Giám đốc là Nguyễn Văn An", 
        "Công ty có 50 nhân viên"
    ]
    
    print(f"📚 Creating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Create FAISS index
    print("🔍 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product
    
    # Normalize embeddings
    embeddings_norm = embeddings.astype('float32')
    faiss.normalize_L2(embeddings_norm)
    
    # Add to index
    index.add(embeddings_norm)
    print(f"Index total: {index.ntotal}")
    
    # Test queries
    test_queries = ["ABC", "2020", "Nguyễn Văn An", "nhân viên"]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Create query embedding
        query_emb = model.encode([query]).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = index.search(query_emb, 3)
        
        print(f"  Scores: {scores[0]}")
        print(f"  Indices: {indices[0]}")
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                print(f"    {i+1}. Score: {score:.4f} - {texts[idx]}")

if __name__ == "__main__":
    main()
