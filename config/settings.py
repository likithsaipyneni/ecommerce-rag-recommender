"""
Configuration settings for the E-commerce RAG system.
"""
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for ML models."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_results: int = 10
    similarity_threshold: float = 0.7

@dataclass
class DatabaseConfig:
    """Configuration for vector database."""
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "product_embeddings"
    distance_metric: str = "cosine"

@dataclass
class AppConfig:
    """Configuration for Streamlit app."""
    page_title: str = "E-commerce RAG Assistant"
    page_icon: str = "🛒"
    layout: str = "wide"
    sidebar_state: str = "expanded"

@dataclass
class SystemConfig:
    """System-wide configuration."""
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    cache_dir: Path = Path(".cache")
    log_level: str = "INFO"
    max_concurrent_requests: int = 100

# Global configuration instances
MODEL_CONFIG = ModelConfig()
DATABASE_CONFIG = DatabaseConfig()
APP_CONFIG = AppConfig()
SYSTEM_CONFIG = SystemConfig()

# Environment variable overrides
if os.getenv("EMBEDDING_MODEL"):
    MODEL_CONFIG.embedding_model = os.getenv("EMBEDDING_MODEL")

if os.getenv("CHROMA_DB_PATH"):
    DATABASE_CONFIG.chroma_db_path = os.getenv("CHROMA_DB_PATH")

if os.getenv("MAX_RESULTS"):
    MODEL_CONFIG.max_results = int(os.getenv("MAX_RESULTS"))

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "model": MODEL_CONFIG,
        "database": DATABASE_CONFIG,
        "app": APP_CONFIG,
        "system": SYSTEM_CONFIG
    }