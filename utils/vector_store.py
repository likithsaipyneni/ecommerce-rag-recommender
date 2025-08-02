"""
Vector database utilities using ChromaDB for the E-commerce RAG system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import uuid
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not installed. Install with: pip install chromadb")

from config.settings import DATABASE_CONFIG

class VectorStore:
    """ChromaDB-based vector store for product embeddings."""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        self.db_path = db_path or DATABASE_CONFIG.chroma_db_path
        self.collection_name = collection_name or DATABASE_CONFIG.collection_name
        self.logger = logging.getLogger(__name__)
        
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            self.logger.error("ChromaDB not available - using mock implementation")
            return
        
        try:
            # Create persistent client
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "E-commerce product embeddings"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str] = None) -> bool:
        """Add embeddings to the vector store."""
        if not self.collection:
            self.logger.error("Collection not available")
            return False
        
        if len(embeddings) != len(texts) or len(embeddings) != len(metadatas):
            self.logger.error("Embeddings, texts, and metadatas must have the same length")
            return False
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
            # Convert numpy array to list for ChromaDB
            embeddings_list = embeddings.tolist()
            
            # Ensure metadata values are JSON serializable
            clean_metadatas = []
            for metadata in metadatas:
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    else:
                        clean_metadata[key] = str(value)
                clean_metadatas.append(clean_metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings_list,
                documents=texts,
                metadatas=clean_metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(embeddings)} embeddings to collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding embeddings: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, n_results: int = 10, where: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for similar embeddings."""
        if not self.collection:
            self.logger.error("Collection not available")
            return {}
        
        try:
            # Convert query embedding to list
            query_list = query_embedding.tolist()
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'ids': results['ids'][0] if results['ids'] else []
            }
            
            # Convert distances to similarities (ChromaDB uses L2 distance)
            similarities = []
            for distance in formatted_results['distances']:
                # Convert L2 distance to cosine similarity approximation
                similarity = max(0.0, 1.0 - (distance / 2.0))
                similarities.append(similarity)
            
            formatted_results['similarities'] = similarities
            
            self.logger.info(f"Search returned {len(formatted_results['documents'])} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error searching embeddings: {e}")
            return {}
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'path': self.db_path
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        if not self.client:
            return False
        
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = None
            self.logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete and recreate)."""
        if not self.client:
            return False
        
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass  # Collection might not exist
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "E-commerce product embeddings"}
            )
            
            self.logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            return False

class MockVectorStore:
    """Mock vector store for testing when ChromaDB is not available."""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Using mock vector store - install ChromaDB for real functionality")
        
        self.embeddings = []
        self.texts = []
        self.metadatas = []
        self.ids = []
    
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str] = None) -> bool:
        """Mock add embeddings."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
        
        self.embeddings.extend(embeddings.tolist())
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        self.logger.info(f"Mock: Added {len(embeddings)} embeddings")
        return True
    
    def search(self, query_embedding: np.ndarray, n_results: int = 10, where: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock search."""
        if not self.embeddings:
            return {
                'documents': [],
                'metadatas': [],
                'distances': [],
                'similarities': [],
                'ids': []
            }
        
        # Simple cosine similarity calculation
        similarities = []
        for emb in self.embeddings:
            emb_array = np.array(emb)
            
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            emb_norm = emb_array / np.linalg.norm(emb_array)
            
            # Cosine similarity
            similarity = np.dot(query_norm, emb_norm)
            similarities.append(float(similarity))
        
        # Sort by similarity
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        
        # Return top results
        n_results = min(n_results, len(sorted_indices))
        top_indices = sorted_indices[:n_results]
        
        return {
            'documents': [self.texts[i] for i in top_indices],
            'metadatas': [self.metadatas[i] for i in top_indices],
            'distances': [1.0 - similarities[i] for i in top_indices],
            'similarities': [similarities[i] for i in top_indices],
            'ids': [self.ids[i] for i in top_indices]
        }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Mock collection info."""
        return {
            'name': 'mock_collection',
            'count': len(self.embeddings),
            'path': 'mock_path'
        }
    
    def delete_collection(self) -> bool:
        """Mock delete collection."""
        self.embeddings = []
        self.texts = []
        self.metadatas = []
        self.ids = []
        return True
    
    def reset_collection(self) -> bool:
        """Mock reset collection."""
        return self.delete_collection()

def get_vector_store(db_path: str = None, collection_name: str = None) -> VectorStore:
    """Get vector store instance with fallback to mock if ChromaDB unavailable."""
    if CHROMADB_AVAILABLE:
        return VectorStore(db_path, collection_name)
    else:
        return MockVectorStore(db_path, collection_name)