import os
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

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Thêm documents vào Pinecone"""
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

        if vectors:
            self.index.upsert(vectors=vectors)

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
                doc = {
                    'id': match['id'],
                    'score': match['score'],
                    **match['metadata']
                }
                documents.append(doc)

        return documents

    def delete_document(self, doc_id: str) -> None:
        """Xóa document"""
        self.index.delete(ids=[doc_id])

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Lấy tất cả documents (Pinecone không hỗ trợ lấy tất cả documents một cách hiệu quả)"""
        # This is not a recommended operation in Pinecone as it can be slow and memory-intensive.
        # A proper implementation might require iterating through the index, which is not directly supported.
        # For now, we return an empty list as a placeholder.
        print("Warning: get_all_documents is not efficiently supported by Pinecone and will return an empty list.")
        return []
