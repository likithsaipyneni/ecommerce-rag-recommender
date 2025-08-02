"""
Advanced data preprocessing utilities for the E-commerce RAG system.
"""
import re
import string
from typing import List, Dict, Any, Set
import logging
from collections import Counter
import numpy as np

class TextPreprocessor:
    """Advanced text preprocessing for product data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stopwords = self._get_stopwords()
    
    def _get_stopwords(self) -> Set[str]:
        """Get common English stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'you', 'your', 'this', 'these',
            'they', 'their', 'them', 'than', 'or', 'but', 'if', 'then', 'else',
            'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\-\,\:]', ' ', text)
        
        # Clean up spaces around punctuation
        text = re.sub(r'\s+([\.!\?])', r'\1', text)
        text = re.sub(r'([\.!\?])\s*', r'\1 ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key terms from text."""
        # Clean text
        clean_text = self.clean_text(text)
        
        # Tokenize
        words = clean_text.split()
        
        # Filter out stopwords and short words
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in self.stopwords
            and not word.isdigit()
        ]
        
        # Count frequency
        word_counts = Counter(keywords)
        
        return [word for word, _ in word_counts.most_common(top_k)]
    
    def extract_technical_specs(self, text: str) -> Dict[str, str]:
        """Extract technical specifications from text."""
        specs = {}
        
        # Common spec patterns
        patterns = [
            r'(\w+):\s*([^\n\.,]+)',  # Key: Value
            r'(\w+)\s*-\s*([^\n\.,]+)',  # Key - Value
            r'(\d+(?:\.\d+)?)\s*(gb|tb|ghz|mhz|mp|inch|hours?|lbs?)\b',  # Number + unit
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                if len(match.groups()) == 2:
                    key, value = match.groups()
                    specs[key.strip()] = value.strip()
        
        return specs
    
    def normalize_price(self, price_text: str) -> float:
        """Extract and normalize price from text."""
        if isinstance(price_text, (int, float)):
            return float(price_text)
        
        # Extract price using regex
        price_match = re.search(r'[\$]?(\d+(?:,\d{3})*(?:\.\d{2})?)', str(price_text))
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            return float(price_str)
        
        return 0.0
    
    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract features that might indicate sentiment."""
        features = {
            'exclamation_count': len(re.findall(r'!', text)),
            'question_count': len(re.findall(r'\?', text)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'positive_words': 0,
            'negative_words': 0,
            'length': len(text.split())
        }
        
        # Simple positive/negative word lists
        positive_words = {
            'excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'perfect',
            'outstanding', 'incredible', 'awesome', 'brilliant', 'superb', 'love',
            'best', 'good', 'nice', 'happy', 'satisfied', 'recommend'
        }
        
        negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointed',
            'poor', 'useless', 'broken', 'defective', 'waste', 'regret', 'cheap',
            'fake', 'scam', 'avoid', 'never', 'don\'t'
        }
        
        words = text.lower().split()
        features['positive_words'] = sum(1 for word in words if word in positive_words)
        features['negative_words'] = sum(1 for word in words if word in negative_words)
        
        return features

class ReviewAnalyzer:
    """Analyze product reviews for authenticity and sentiment."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.logger = logging.getLogger(__name__)
    
    def calculate_authenticity_score(self, review: Dict[str, Any]) -> float:
        """Calculate authenticity score for a review (0-1, higher is more authentic)."""
        score = 1.0
        text = review.get('text', '')
        
        if not text:
            return 0.0
        
        # Length analysis - very short or very long reviews might be fake
        length = len(text.split())
        if length < 10:
            score -= 0.3
        elif length > 200:
            score -= 0.2
        
        # Generic content detection
        generic_phrases = [
            'great product', 'highly recommend', 'fast shipping', 'good quality',
            'value for money', 'as described', 'perfect', 'amazing', 'excellent'
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
        if generic_count > 3:
            score -= 0.4
        
        # Repetitive patterns
        words = text.lower().split()
        if len(set(words)) < len(words) * 0.7:  # High repetition
            score -= 0.3
        
        # Verified purchase bonus
        if review.get('verified', False):
            score += 0.2
        
        # Sentiment features
        features = self.preprocessor.extract_sentiment_features(text)
        
        # Balanced sentiment is more authentic
        if features['positive_words'] > 0 and features['negative_words'] > 0:
            score += 0.1
        
        # Too many caps might indicate fake enthusiasm
        if features['caps_ratio'] > 0.1:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def extract_pros_cons(self, reviews: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract pros and cons from reviews."""
        pros = []
        cons = []
        
        for review in reviews:
            text = review.get('text', '')
            rating = review.get('rating', 3)
            
            # Extract sentences
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Classify based on rating and content
                if rating >= 4:
                    # Look for positive aspects
                    if any(word in sentence.lower() for word in ['great', 'excellent', 'love', 'perfect', 'amazing']):
                        pros.append(sentence)
                elif rating <= 2:
                    # Look for negative aspects
                    if any(word in sentence.lower() for word in ['bad', 'terrible', 'poor', 'hate', 'disappointed']):
                        cons.append(sentence)
        
        return {
            'pros': list(set(pros))[:5],  # Top 5 unique pros
            'cons': list(set(cons))[:5]   # Top 5 unique cons
        }

def preprocess_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess all product data."""
    preprocessor = TextPreprocessor()
    analyzer = ReviewAnalyzer()
    
    processed_products = []
    
    for product in products:
        processed_product = product.copy()
        
        # Clean description
        if 'description' in product:
            processed_product['description_clean'] = preprocessor.clean_text(product['description'])
            processed_product['keywords'] = preprocessor.extract_keywords(product['description'])
        
        # Normalize price
        if 'price' in product:
            processed_product['price_normalized'] = preprocessor.normalize_price(product['price'])
        
        # Process reviews
        if 'reviews' in product:
            processed_reviews = []
            for review in product['reviews']:
                processed_review = review.copy()
                processed_review['authenticity_score'] = analyzer.calculate_authenticity_score(review)
                processed_review['sentiment_features'] = preprocessor.extract_sentiment_features(review.get('text', ''))
                processed_reviews.append(processed_review)
            
            processed_product['reviews'] = processed_reviews
            processed_product['review_analysis'] = analyzer.extract_pros_cons(product['reviews'])
        
        processed_products.append(processed_product)
    
    return processed_products