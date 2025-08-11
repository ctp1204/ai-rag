import os
import json
import pickle
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
import faiss
from config import settings

# ChromaDB availability flag
CHROMADB_AVAILABLE = False
chromadb = None

class VectorDatabase(ABC):
    """Abstract base class cho Vector Database"""

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        pass

class ChromaVectorDB(VectorDatabase):
    """ChromaDB implementation"""

    def __init__(self, persist_directory: str = None):
        global chromadb, CHROMADB_AVAILABLE

        # Try to import chromadb when actually needed
        if not CHROMADB_AVAILABLE:
            try:
                import chromadb as _chromadb
                chromadb = _chromadb
                CHROMADB_AVAILABLE = True
            except (ImportError, RuntimeError) as e:
                raise RuntimeError(f"ChromaDB is not available: {e}. Please use FAISS instead.")

        self.persist_directory = persist_directory or settings.vector_db_path
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Thêm documents vào ChromaDB"""
        ids = [doc['id'] for doc in documents]
        embeddings = [doc['embedding'].tolist() for doc in documents]
        texts = [doc['text'] for doc in documents]
        metadatas = []

        for doc in documents:
            metadata = {
                'source': doc['source'],
                'chunk_index': doc['chunk_index'],
                **doc.get('metadata', {})
            }
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm documents tương tự"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        documents = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i]
                }
                documents.append(doc)

        return documents

    def delete_document(self, doc_id: str) -> None:
        """Xóa document"""
        self.collection.delete(ids=[doc_id])

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Lấy tất cả documents"""
        results = self.collection.get()
        documents = []

        if results['ids']:
            for i in range(len(results['ids'])):
                doc = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                documents.append(doc)

        return documents

class FAISSVectorDB(VectorDatabase):
    """FAISS implementation"""

    def __init__(self, persist_directory: str = None, dimension: int = 384):
        self.persist_directory = persist_directory or settings.vector_db_path
        os.makedirs(self.persist_directory, exist_ok=True)

        self.dimension = dimension
        self.index_path = os.path.join(self.persist_directory, "faiss.index")
        self.metadata_path = os.path.join(self.persist_directory, "metadata.json")
        self.embeddings_path = os.path.join(self.persist_directory, "embeddings.pkl")

        # Load existing index or create new one
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            # Load embeddings for rebuilding
            if os.path.exists(self.embeddings_path):
                with open(self.embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
            else:
                self.embeddings = []
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.metadata = {}
            self.embeddings = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Thêm documents vào FAISS"""
        embeddings = np.array([doc['embedding'] for doc in documents]).astype('float32')

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        start_id = self.index.ntotal
        self.index.add(embeddings)

        # Save embeddings for rebuilding
        for embedding in embeddings:
            self.embeddings.append(embedding)

        # Save metadata
        for i, doc in enumerate(documents):
            doc_metadata = {
                'id': doc['id'],
                'text': doc['text'],
                'source': doc['source'],
                'chunk_index': doc['chunk_index'],
                'metadata': doc.get('metadata', {})
            }
            self.metadata[str(start_id + i)] = doc_metadata

        self._save_index()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm documents tương tự"""
        if self.index.ntotal == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

        documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and str(idx) in self.metadata:
                doc = self.metadata[str(idx)].copy()
                # Convert inner product to cosine similarity (already normalized)
                # FAISS IndexFlatIP returns inner product, which is cosine similarity for normalized vectors
                doc['score'] = max(0.0, float(score))  # Ensure non-negative
                documents.append(doc)

        return documents

    def delete_document(self, doc_id: str) -> None:
        """Xóa document (FAISS không hỗ trợ delete trực tiếp, cần rebuild index)"""
        # Remove from metadata
        indices_to_remove = []
        for idx, metadata in self.metadata.items():
            if metadata['id'] == doc_id:
                indices_to_remove.append(idx)

        for idx in indices_to_remove:
            del self.metadata[idx]

        # Rebuild index (expensive operation)
        self._rebuild_index()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Lấy tất cả documents"""
        return list(self.metadata.values())

    def _save_index(self):
        """Lưu index, metadata và embeddings"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

    def _rebuild_index(self):
        """Rebuild index sau khi xóa documents"""
        if not self.metadata:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.embeddings = []
            self._save_index()
            return

        # Rebuild index with remaining embeddings
        valid_embeddings = []
        new_metadata = {}
        new_idx = 0

        for old_idx, doc_metadata in self.metadata.items():
            old_idx_int = int(old_idx)
            if old_idx_int < len(self.embeddings):
                valid_embeddings.append(self.embeddings[old_idx_int])
                new_metadata[str(new_idx)] = doc_metadata
                new_idx += 1

        # Create new index
        self.index = faiss.IndexFlatIP(self.dimension)
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings).astype('float32')
            self.index.add(embeddings_array)

        self.metadata = new_metadata
        self.embeddings = valid_embeddings
        self._save_index()

def create_vector_database() -> VectorDatabase:
    """Factory function để tạo vector database"""
    if settings.vector_db_type.lower() == "chromadb":
        if not CHROMADB_AVAILABLE:
            print("Warning: ChromaDB not available, falling back to FAISS")
            return FAISSVectorDB()
        return ChromaVectorDB()
    elif settings.vector_db_type.lower() == "faiss":
        return FAISSVectorDB()
    else:
        raise ValueError(f"Unsupported vector database type: {settings.vector_db_type}")
