import sys
import os
import pytest
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.core.document_processor import DocumentProcessor
from src.core.vector_database import ChromaVectorDB
from src.core.retrieval_engine import RetrievalEngine
from src.core.rag_pipeline import RAGPipeline

class TestDocumentProcessor:
    def test_chunk_text(self):
        """Test text chunking functionality"""
        processor = DocumentProcessor()
        text = "This is a test. " * 100  # Create a long text
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_create_embeddings(self):
        """Test embedding creation"""
        processor = DocumentProcessor()
        texts = ["Hello world", "This is a test"]
        embeddings = processor.create_embeddings(texts)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Should have some dimensions
    
    def test_process_text_directly(self):
        """Test processing text directly"""
        processor = DocumentProcessor()
        text = "This is a test document with some content."
        documents = processor.process_text_directly(text)
        
        assert len(documents) > 0
        assert all('id' in doc for doc in documents)
        assert all('text' in doc for doc in documents)
        assert all('embedding' in doc for doc in documents)

class TestVectorDatabase:
    def test_chromadb_basic_operations(self):
        """Test basic ChromaDB operations"""
        # Use temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaVectorDB(persist_directory=temp_dir)
            
            # Test adding documents
            processor = DocumentProcessor()
            documents = processor.process_text_directly("Test document content")
            
            db.add_documents(documents)
            
            # Test searching
            query_embedding = processor.create_embeddings(["test"])[0]
            results = db.search(query_embedding, k=1)
            
            assert len(results) > 0
            assert 'score' in results[0]

class TestRetrievalEngine:
    def test_add_and_search(self):
        """Test adding documents and searching"""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create retrieval engine with temporary database
            from src.core.vector_database import ChromaVectorDB
            vector_db = ChromaVectorDB(persist_directory=temp_dir)
            engine = RetrievalEngine(vector_db=vector_db)
            
            # Add a test document
            test_text = "Python is a programming language. It is used for web development, data science, and AI."
            chunks_added = engine.add_document_from_text(test_text, "test_source")
            
            assert chunks_added > 0
            
            # Search for relevant content
            results = engine.search("programming language")
            assert len(results) > 0
            
            # Test context retrieval
            context = engine.get_relevant_context("Python programming")
            assert len(context) > 0
            assert "Python" in context

class TestRAGPipeline:
    def test_pipeline_without_llm(self):
        """Test RAG pipeline without LLM (no API keys)"""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline with temporary database
            from src.core.vector_database import ChromaVectorDB
            vector_db = ChromaVectorDB(persist_directory=temp_dir)
            retrieval_engine = RetrievalEngine(vector_db=vector_db)
            
            # Create pipeline without LLM manager (will fail gracefully)
            pipeline = RAGPipeline(retrieval_engine=retrieval_engine, llm_manager=None)
            
            # Add test data
            result = pipeline.add_document_from_text(
                "Artificial Intelligence is the simulation of human intelligence in machines.",
                "test_ai_doc"
            )
            assert result['success'] == True
            
            # Test query without fallback (should return no data response)
            response = pipeline.query("What is AI?", use_fallback=False)
            assert 'response' in response
            assert 'metadata' in response
    
    def test_health_check(self):
        """Test system health check"""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.core.vector_database import ChromaVectorDB
            vector_db = ChromaVectorDB(persist_directory=temp_dir)
            retrieval_engine = RetrievalEngine(vector_db=vector_db)
            pipeline = RAGPipeline(retrieval_engine=retrieval_engine, llm_manager=None)
            
            health = pipeline.health_check()
            assert 'retrieval_engine' in health
            assert 'database_stats' in health
            assert health['retrieval_engine'] == 'ok'

def test_config_loading():
    """Test configuration loading"""
    from config import settings
    
    # Test that settings can be loaded
    assert hasattr(settings, 'vector_db_type')
    assert hasattr(settings, 'embedding_model')
    assert hasattr(settings, 'similarity_threshold')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
