import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Vector Database
    vector_db_type: str = os.getenv("VECTOR_DB_TYPE", "pinecone")
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")

    # Document Storage
    documents_path: str = os.getenv("DOCUMENTS_PATH", "./data/documents")

    # Model Settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # Retrieval Settings
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
    max_retrieved_docs: int = int(os.getenv("MAX_RETRIEVED_DOCS", "5"))

    # Server Settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

settings = Settings()
