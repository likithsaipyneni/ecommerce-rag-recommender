"""
Data ingestion and preprocessing pipeline for the E-commerce RAG system.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
from dataclasses import dataclass

@dataclass
class ProductChunk:
    """Represents a chunk of product information."""
    id: str
    product_id: str
    content: str
    chunk_type: str  # description, specs, review, feature
    metadata: Dict[str, Any]

class ProductDataProcessor:
    """Processes and chunks product data for embedding."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
    
    def load_products(self, file_path: str) -> List[Dict[str, Any]]:
        """Load products from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                products = json.load(f)
            self.logger.info(f"Loaded {len(products)} products from {file_path}")
            return products
        except Exception as e:
            self.logger.error(f"Error loading products: {e}")
            return []
    
    def chunk_text(self, text: str, chunk_type: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_product(self, product: Dict[str, Any]) -> List[ProductChunk]:
        """Process a single product into chunks."""
        chunks = []
        product_id = product['id']
        
        # Process description
        if 'description' in product:
            desc_chunks = self.chunk_text(product['description'], 'description')
            for i, chunk in enumerate(desc_chunks):
                chunks.append(ProductChunk(
                    id=f"{product_id}_desc_{i}",
                    product_id=product_id,
                    content=chunk,
                    chunk_type='description',
                    metadata={
                        'product_name': product.get('name', ''),
                        'category': product.get('category', ''),
                        'brand': product.get('brand', ''),
                        'price': product.get('price', 0)
                    }
                ))
        
        # Process specifications
        if 'specifications' in product:
            specs_text = self._format_specifications(product['specifications'])
            spec_chunks = self.chunk_text(specs_text, 'specifications')
            for i, chunk in enumerate(spec_chunks):
                chunks.append(ProductChunk(
                    id=f"{product_id}_spec_{i}",
                    product_id=product_id,
                    content=chunk,
                    chunk_type='specifications',
                    metadata={
                        'product_name': product.get('name', ''),
                        'category': product.get('category', ''),
                        'brand': product.get('brand', ''),
                        'price': product.get('price', 0)
                    }
                ))
        
        # Process reviews
        if 'reviews' in product:
            for review_idx, review in enumerate(product['reviews']):
                review_text = f"Rating: {review.get('rating', 0)}/5. {review.get('text', '')}"
                review_chunks = self.chunk_text(review_text, 'review')
                for i, chunk in enumerate(review_chunks):
                    chunks.append(ProductChunk(
                        id=f"{product_id}_review_{review_idx}_{i}",
                        product_id=product_id,
                        content=chunk,
                        chunk_type='review',
                        metadata={
                            'product_name': product.get('name', ''),
                            'category': product.get('category', ''),
                            'brand': product.get('brand', ''),
                            'price': product.get('price', 0),
                            'rating': review.get('rating', 0),
                            'verified': review.get('verified', False),
                            'date': review.get('date', '')
                        }
                    ))
        
        # Process features
        if 'features' in product:
            features_text = "Key features: " + ", ".join(product['features'])
            feature_chunks = self.chunk_text(features_text, 'features')
            for i, chunk in enumerate(feature_chunks):
                chunks.append(ProductChunk(
                    id=f"{product_id}_feat_{i}",
                    product_id=product_id,
                    content=chunk,
                    chunk_type='features',
                    metadata={
                        'product_name': product.get('name', ''),
                        'category': product.get('category', ''),
                        'brand': product.get('brand', ''),
                        'price': product.get('price', 0)
                    }
                ))
        
        return chunks
    
    def _format_specifications(self, specs: Dict[str, Any]) -> str:
        """Format specifications as readable text."""
        formatted = []
        for key, value in specs.items():
            # Clean up key names
            clean_key = key.replace('_', ' ').title()
            formatted.append(f"{clean_key}: {value}")
        
        return "Technical specifications: " + ". ".join(formatted)
    
    def process_all_products(self, products: List[Dict[str, Any]]) -> List[ProductChunk]:
        """Process all products into chunks."""
        all_chunks = []
        
        for product in products:
            try:
                product_chunks = self.process_product(product)
                all_chunks.extend(product_chunks)
                self.logger.info(f"Processed product {product.get('name', 'Unknown')} into {len(product_chunks)} chunks")
            except Exception as e:
                self.logger.error(f"Error processing product {product.get('id', 'Unknown')}: {e}")
        
        self.logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

def initialize_system():
    """Initialize the RAG system with sample data."""
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("utils").mkdir(exist_ok=True)
    Path(".cache").mkdir(exist_ok=True)
    
    print("✅ E-commerce RAG system initialized successfully!")
    print(f"📁 Sample data available: {Path('data/sample_products.json').exists()}")
    print("🚀 Run 'streamlit run app/main.py' to start the application")

if __name__ == "__main__":
    initialize_system()