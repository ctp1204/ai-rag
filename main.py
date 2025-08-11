#!/usr/bin/env python3
"""
RAG System - Main Entry Point
Hệ thống Retrieval-Augmented Generation với fallback API
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.api.main import app
from config import settings

def main():
    """Main function to run the RAG system"""
    print("🚀 Starting RAG System...")
    print(f"📊 Vector Database: {settings.vector_db_type}")
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
