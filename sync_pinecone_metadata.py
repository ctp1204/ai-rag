#!/usr/bin/env python3
"""
Script để sync metadata từ Pinecone về local storage
Sử dụng khi đã có dữ liệu trong Pinecone nhưng chưa có local metadata
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.core.rag_pipeline import RAGPipeline
import json
import uuid

def sync_pinecone_metadata():
    print("🔄 Sync Pinecone Metadata to Local Storage")
    print("=" * 60)
    
    # Initialize RAG pipeline
    print("📊 Khởi tạo RAG pipeline...")
    rag = RAGPipeline()
    
    # Check if using Pinecone
    if not hasattr(rag.retrieval_engine.vector_db, 'index'):
        print("❌ Không phải Pinecone database!")
        return
    
    vector_db = rag.retrieval_engine.vector_db
    
    print(f"🔍 Đang quét dữ liệu trong Pinecone index: {vector_db.index_name}")
    
    try:
        # Query Pinecone to get sample data and estimate total
        # We'll use a dummy vector to query and get some results
        import numpy as np
        
        # Create a random query vector to sample data
        dummy_vector = np.random.random(384).tolist()
        
        # Query with high top_k to get as many results as possible
        results = vector_db.index.query(
            vector=dummy_vector,
            top_k=10000,  # Max allowed by Pinecone
            include_metadata=True
        )
        
        print(f"📄 Tìm thấy {len(results['matches'])} documents trong Pinecone")
        
        if not results['matches']:
            print("❌ Không có dữ liệu trong Pinecone!")
            return
        
        # Convert to local metadata format
        local_metadata = {}
        
        for match in results['matches']:
            doc_id = match['id']
            metadata = match.get('metadata', {})
            
            local_metadata[doc_id] = {
                'id': doc_id,
                'text': metadata.get('text', ''),
                'source': metadata.get('source', 'Unknown'),
                'chunk_index': metadata.get('chunk_index', -1),
                'metadata': {k: v for k, v in metadata.items() 
                           if k not in ['text', 'source', 'chunk_index']}
            }
        
        # Save to local metadata file
        metadata_dir = Path("data/pinecone_metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / "documents.json"
        
        print(f"💾 Lưu metadata vào {metadata_file}")
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(local_metadata, f, ensure_ascii=False, indent=2)
        
        # Update the vector_db instance
        vector_db.local_metadata = local_metadata
        
        print(f"✅ Đã sync {len(local_metadata)} documents!")
        
        # Show statistics
        source_counts = {}
        for doc in local_metadata.values():
            source = doc.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\n📊 Thống kê theo nguồn:")
        for source, count in source_counts.items():
            print(f"   - {source}: {count} chunks")
        
        # Test the updated system
        print(f"\n🧪 Test hệ thống sau khi sync:")
        db_info = rag.get_database_info()
        print(f"   📚 Tổng documents: {db_info['total_documents']}")
        print(f"   🗄️ Vector DB type: {db_info['vector_db_type']}")
        
        if db_info['total_documents'] > 0:
            print(f"\n✅ Sync thành công! Bây giờ có thể chạy Q&A:")
            print("   python3 test_qa_with_real_data.py")
        
    except Exception as e:
        print(f"❌ Lỗi sync: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try alternative approach: scan with describe_index_stats
        try:
            print(f"\n🔄 Thử phương pháp khác...")
            stats = vector_db.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            print(f"📊 Pinecone stats: {total_vectors} vectors total")
            
            if total_vectors > 0:
                print(f"💡 Có dữ liệu trong Pinecone nhưng không thể sync tự động.")
                print(f"   Hãy thử upload lại dữ liệu để tạo local metadata:")
                print(f"   python3 upload_sample_data.py")
            else:
                print(f"❌ Pinecone index trống!")
                
        except Exception as e2:
            print(f"❌ Không thể lấy stats từ Pinecone: {str(e2)}")

def main():
    sync_pinecone_metadata()

if __name__ == "__main__":
    main()
