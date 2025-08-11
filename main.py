#!/usr/bin/env python3
"""
RAG System - Main Entry Point
Hệ thống Retrieval-Augmented Generation với fallback API
"""

import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.api.main import app
from config import settings

def main():
    """Main function to run the RAG system"""
    print("🚀 Starting RAG System...")
    db_type = settings.vector_db_type
    if "pinecone" in db_type.lower():
        db_display_name = "Pinecone"
    elif "chroma" in db_type.lower():
        db_display_name = "ChromaDB"
    else:
        db_display_name = "FAISS"
    print(f"📊 Vector Database: {db_display_name}")
    print(f"🤖 LLM Provider: {settings.llm_provider}")
    print(f"🔗 Server: http://{settings.host}:{settings.port}")
    print("=" * 50)

    # Ensure required directories exist
    os.makedirs(settings.documents_path, exist_ok=True)
    os.makedirs(settings.vector_db_path, exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
