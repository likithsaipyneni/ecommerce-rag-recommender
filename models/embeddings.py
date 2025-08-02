"""
Embedding models and utilities for the E-commerce RAG system.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import pickle
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
    SentenceTransformer = None

@dataclass
class EmbeddingResult:
    """Result of embedding computation."""
    embeddings: np.ndarray
    model_name: str
    dimension: int
    texts: List[str]

class EmbeddingModel:
    """Handles text embeddings using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not available")
            
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> Optional[EmbeddingResult]:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            self.logger.error("Embedding model not available")
            return None
        
        if not texts:
            return None
        
        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            result = EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                dimension=embeddings.shape[1],
                texts=texts.copy()
            )
            
            self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None
    
    def embed_single(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text."""
        if self.model is None:
            return None
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0]
        except Exception as e:
            self.logger.error(f"Error generating single embedding: {e}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar embeddings to a query."""
        if len(candidate_embeddings) == 0:
            return []
        
        try:
            # Compute similarities
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                sim = self.compute_similarity(query_embedding, candidate)
                similarities.append({
                    'index': i,
                    'similarity': sim
                })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar embeddings: {e}")
            return []
    
    def save_embeddings(self, result: EmbeddingResult, filename: str):
        """Save embeddings to cache."""
        try:
            cache_path = self.cache_dir / f"{filename}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            self.logger.info(f"Embeddings saved to {cache_path}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filename: str) -> Optional[EmbeddingResult]:
        """Load embeddings from cache."""
        try:
            cache_path = self.cache_dir / f"{filename}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                self.logger.info(f"Embeddings loaded from {cache_path}")
                return result
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
        return None

class ProductEmbeddingManager:
    """Manages embeddings specifically for product data."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = EmbeddingModel(model_name)
        self.logger = logging.getLogger(__name__)
    
    def create_product_embeddings(self, chunks: List[Any]) -> Dict[str, Any]:
        """Create embeddings for product chunks."""
        if not chunks:
            return {}
        
        # Extract texts and metadata
        texts = []
        metadata = []
        
        for chunk in chunks:
            # Handle different chunk formats
            if hasattr(chunk, 'content'):
                text = chunk.content
                meta = {
                    'id': chunk.id,
                    'product_id': chunk.product_id,
                    'chunk_type': chunk.chunk_type,
                    'metadata': chunk.metadata
                }
            else:
                text = str(chunk)
                meta = {'id': len(texts)}
            
            texts.append(text)
            metadata.append(meta)
        
        # Generate embeddings
        embedding_result = self.embedding_model.embed_texts(texts)
        
        if embedding_result is None:
            return {}
        
        return {
            'embeddings': embedding_result.embeddings,
            'texts': texts,
            'metadata': metadata,
            'model_name': embedding_result.model_name,
            'dimension': embedding_result.dimension
        }
    
    def search_similar_products(self, query: str, product_embeddings: Dict[str, Any], top_k: int = 10, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar products based on query."""
        if not product_embeddings or 'embeddings' not in product_embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_single(query)
        if query_embedding is None:
            return []
        
        # Find similar embeddings
        similar_results = self.embedding_model.find_most_similar(
            query_embedding,
            product_embeddings['embeddings'],
            top_k=top_k
        )
        
        # Filter by threshold and add metadata
        results = []
        for result in similar_results:
            if result['similarity'] >= similarity_threshold:
                idx = result['index']
                results.append({
                    'similarity': result['similarity'],
                    'text': product_embeddings['texts'][idx],
                    'metadata': product_embeddings['metadata'][idx]
                })
        
        return results
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        if len(embeddings) == 0:
            return {}
        
        return {
            'count': len(embeddings),
            'dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1)))
        }

# Fallback implementation for when sentence-transformers is not available
class MockEmbeddingModel:
    """Mock embedding model for testing when dependencies are not available."""
    
    def __init__(self, model_name: str = "mock", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using mock embedding model - install sentence-transformers for real embeddings")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        """Generate mock embeddings."""
        embeddings = np.random.randn(len(texts), self.dimension)
        # Normalize to unit vectors
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            dimension=self.dimension,
            texts=texts.copy()
        )
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate mock embedding for single text."""
        embedding = np.random.randn(self.dimension)
        return embedding / np.linalg.norm(embedding)

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingModel:
    """Get embedding model, with fallback to mock if dependencies unavailable."""
    try:
        return EmbeddingModel(model_name)
    except Exception:
        return MockEmbeddingModel(model_name)