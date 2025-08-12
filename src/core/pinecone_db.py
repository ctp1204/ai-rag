import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from .vector_database import VectorDatabase
from config import settings

class PineconeVectorDB(VectorDatabase):
    """Pinecone implementation"""

    def __init__(self, dimension: int = 384):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if not self.api_key or not self.environment or not self.index_name:
            raise ValueError("Pinecone API key, environment, or index name not found in environment variables.")

        self.pinecone = Pinecone(api_key=self.api_key)

        # Check if the index exists and has the correct dimension
        if self.index_name in self.pinecone.list_indexes().names():
            index_info = self.pinecone.describe_index(self.index_name)
            if index_info.dimension != dimension:
                print(f"Warning: Index '{self.index_name}' has dimension {index_info.dimension}, but model requires {dimension}.")
                print(f"Deleting and recreating index '{self.index_name}' with the correct dimension.")
                self.pinecone.delete_index(self.index_name)
                self._create_index_if_not_exists(dimension)
        else:
            self._create_index_if_not_exists(dimension)

        self.index = self.pinecone.Index(self.index_name)

        # Local metadata storage for tracking documents
        self.metadata_dir = Path("data/pinecone_metadata")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.metadata_dir / "documents.json"

        # Load existing metadata
        self.local_metadata = self._load_metadata()

    def _create_index_if_not_exists(self, dimension: int):
        """Helper function to create a new index."""
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"Creating new Pinecone index '{self.index_name}' with dimension {dimension}.")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

    def _load_metadata(self) -> Dict[str, Any]:
        """Load local metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata file: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """Save local metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.local_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata file: {e}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Thêm documents vào Pinecone và lưu metadata local"""
        vectors = []
        for doc in documents:
            metadata = {
                'text': doc['text'],
                'source': doc['source'],
                'chunk_index': doc['chunk_index'],
                **doc.get('metadata', {})
            }
            vectors.append({
                'id': doc['id'],
                'values': doc['embedding'].tolist(),
                'metadata': metadata
            })

            # Save to local metadata
            self.local_metadata[doc['id']] = {
                'id': doc['id'],
                'text': doc['text'],
                'source': doc['source'],
                'chunk_index': doc['chunk_index'],
                'metadata': doc.get('metadata', {})
            }

        if vectors:
            self.index.upsert(vectors=vectors)
            self._save_metadata()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm documents tương tự"""
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )

        documents = []
        if results['matches']:
            for match in results['matches']:
                # Extract the main fields and keep the rest in a 'metadata' sub-dictionary
                metadata = match.get('metadata', {})
                doc = {
                    'id': match['id'],
                    'score': match['score'],
                    'text': metadata.pop('text', ''),
                    'source': metadata.pop('source', ''),
                    'chunk_index': metadata.pop('chunk_index', -1),
                    'metadata': metadata  # The rest of the metadata
                }
                documents.append(doc)

        return documents

    def delete_document(self, doc_id: str) -> None:
        """Xóa document"""
        self.index.delete(ids=[doc_id])

        # Remove from local metadata
        if doc_id in self.local_metadata:
            del self.local_metadata[doc_id]
            self._save_metadata()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Lấy tất cả documents từ local metadata"""
        return list(self.local_metadata.values())

    def get_info(self) -> Dict[str, Any]:
        """Lấy thông tin về database"""
        # Count documents from local metadata
        total_docs = len(self.local_metadata)

        # Count by source
        source_counts = {}
        for doc in self.local_metadata.values():
            source = doc.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        return {
            'total_documents': total_docs,
            'sources': source_counts,
            'vector_db_type': 'pinecone'
        }

    def clear_all(self) -> None:
        """Xóa toàn bộ dữ liệu"""
        # Delete all vectors from Pinecone
        try:
            # Get all IDs from local metadata
            all_ids = list(self.local_metadata.keys())
            if all_ids:
                # Delete in batches (Pinecone has limits)
                batch_size = 1000
                for i in range(0, len(all_ids), batch_size):
                    batch_ids = all_ids[i:i + batch_size]
                    self.index.delete(ids=batch_ids)

            # Clear local metadata
            self.local_metadata = {}
            self._save_metadata()

        except Exception as e:
            print(f"Error clearing Pinecone data: {e}")
            # Still clear local metadata even if Pinecone delete fails
            self.local_metadata = {}
            self._save_metadata()
