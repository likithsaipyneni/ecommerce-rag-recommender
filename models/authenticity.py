"""
Review authenticity detection for identifying fake or low-quality reviews.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
from collections import Counter
import numpy as np
from datetime import datetime, timedelta

@dataclass
class AuthenticityResult:
    """Result of authenticity analysis."""
    score: float  # 0 to 1, higher is more authentic
    confidence: float  # 0 to 1
    flags: List[str]  # List of potential issues
    details: Dict[str, Any]

class AuthenticityDetector:
    """Detects potentially fake or low-quality reviews."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common patterns in fake reviews
        self.generic_phrases = {
            'great product', 'highly recommend', 'fast shipping', 'good quality',
            'value for money', 'as described', 'perfect', 'amazing product',
            'excellent quality', 'very satisfied', 'would buy again', 'five stars',
            'best purchase', 'exceeded expectations', 'really happy', 'love it',
            'exactly what i wanted', 'arrived quickly', 'well made', 'good price'
        }
        
        self.spam_indicators = {
            'buy now', 'click here', 'visit our', 'check out', 'discount code',
            'free shipping', 'limited time', 'special offer', 'contact us',
            'website', 'http', 'www', '@', '.com', '.net'
        }
        
        # Words that often appear in fake reviews
        self.fake_indicators = {
            'definitely', 'absolutely', 'perfect', 'flawless', 'incredible',
            'unbelievable', 'magical', 'life-changing', 'revolutionary',
            'best ever', 'nothing wrong', 'no complaints', 'no issues'
        }
    
    def analyze_authenticity(self, review: Dict[str, Any]) -> AuthenticityResult:
        """Analyze a single review for authenticity."""
        text = review.get('text', '').strip()
        rating = review.get('rating', 0)
        verified = review.get('verified', False)
        date_str = review.get('date', '')
        
        if not text:
            return AuthenticityResult(
                score=0.0,
                confidence=1.0,
                flags=['empty_text'],
                details={'error': 'No text content'}
            )
        
        flags = []
        details = {}
        
        # Length analysis
        word_count = len(text.split())
        char_count = len(text)
        
        details['word_count'] = word_count
        details['char_count'] = char_count
        
        # Very short reviews are suspicious
        if word_count < 5:
            flags.append('too_short')
        
        # Very long reviews might be copy-pasted
        if word_count > 300:
            flags.append('unusually_long')
        
        # Generic content analysis
        generic_score = self._check_generic_content(text)
        details['generic_score'] = generic_score
        
        if generic_score > 0.3:
            flags.append('generic_content')
        
        # Spam indicators
        spam_score = self._check_spam_indicators(text)
        details['spam_score'] = spam_score
        
        if spam_score > 0.2:
            flags.append('spam_indicators')
        
        # Language quality analysis
        language_score = self._analyze_language_quality(text)
        details['language_score'] = language_score
        
        if language_score < 0.3:
            flags.append('poor_language_quality')
        
        # Rating-text consistency
        consistency_score = self._check_rating_consistency(text, rating)
        details['consistency_score'] = consistency_score
        
        if consistency_score < 0.5:
            flags.append('rating_text_mismatch')
        
        # Repetition analysis
        repetition_score = self._check_repetition(text)
        details['repetition_score'] = repetition_score
        
        if repetition_score > 0.4:
            flags.append('high_repetition')
        
        # Fake review indicators
        fake_score = self._check_fake_indicators(text)
        details['fake_indicators_score'] = fake_score
        
        if fake_score > 0.3:
            flags.append('fake_indicators')
        
        # Calculate overall authenticity score
        base_score = 1.0
        
        # Penalties for various issues
        if 'too_short' in flags:
            base_score -= 0.3
        if 'generic_content' in flags:
            base_score -= 0.4
        if 'spam_indicators' in flags:
            base_score -= 0.5
        if 'poor_language_quality' in flags:
            base_score -= 0.3
        if 'rating_text_mismatch' in flags:
            base_score -= 0.3
        if 'high_repetition' in flags:
            base_score -= 0.3
        if 'fake_indicators' in flags:
            base_score -= 0.4
        
        # Bonuses for authenticity indicators
        if verified:
            base_score += 0.2
        
        if 20 <= word_count <= 150:  # Good length range
            base_score += 0.1
        
        if consistency_score > 0.8:
            base_score += 0.1
        
        # Clamp score to [0, 1]
        final_score = max(0.0, min(1.0, base_score))
        
        # Confidence based on number of checks performed
        confidence = min(1.0, len(details) / 8.0)
        
        return AuthenticityResult(
            score=final_score,
            confidence=confidence,
            flags=flags,
            details=details
        )
    
    def _check_generic_content(self, text: str) -> float:
        """Check for generic phrases that might indicate fake reviews."""
        text_lower = text.lower()
        
        # Count generic phrases
        generic_count = 0
        for phrase in self.generic_phrases:
            if phrase in text_lower:
                generic_count += 1
        
        # Calculate ratio
        total_phrases = len(self.generic_phrases)
        generic_ratio = generic_count / total_phrases
        
        return min(1.0, generic_ratio * 3)  # Scale up to make it more sensitive
    
    def _check_spam_indicators(self, text: str) -> float:
        """Check for spam indicators."""
        text_lower = text.lower()
        
        spam_count = 0
        for indicator in self.spam_indicators:
            if indicator in text_lower:
                spam_count += 1
        
        # Any spam indicator is highly suspicious
        return min(1.0, spam_count / 3.0)
    
    def _analyze_language_quality(self, text: str) -> float:
        """Analyze language quality indicators."""
        # Basic grammar and structure checks
        score = 1.0
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.strip()) > 3]
        
        if len(valid_sentences) == 0:
            return 0.0
        
        # Check capitalization
        words = text.split()
        if words:
            first_word_cap = words[0][0].isupper() if words[0] else False
            if not first_word_cap:
                score -= 0.2
        
        # Check for excessive punctuation
        punct_count = sum(text.count(p) for p in '!?...')
        if punct_count > len(text) * 0.1:
            score -= 0.3
        
        # Check for proper spacing
        if '  ' in text or text.count(' ') < len(words) - 1:
            score -= 0.2
        
        # Check for mix of upper/lower case (all caps is suspicious)
        if text.isupper() and len(text) > 20:
            score -= 0.4
        
        return max(0.0, score)
    
    def _check_rating_consistency(self, text: str, rating: int) -> float:
        """Check if text sentiment matches the rating."""
        if rating == 0:
            return 0.5  # No rating to compare
        
        text_lower = text.lower()
        
        # Positive words
        positive_words = [
            'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome',
            'fantastic', 'wonderful', 'outstanding', 'brilliant', 'superb',
            'satisfied', 'happy', 'pleased', 'recommend', 'good', 'nice'
        ]
        
        # Negative words
        negative_words = [
            'terrible', 'awful', 'horrible', 'hate', 'worst', 'bad', 'poor',
            'disappointed', 'waste', 'useless', 'broken', 'defective',
            'slow', 'expensive', 'problem', 'issue', 'unfortunately'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # High rating (4-5) should have more positive words
        if rating >= 4:
            if pos_count > neg_count:
                return 1.0
            elif pos_count == neg_count and pos_count > 0:
                return 0.7
            else:
                return 0.3
        
        # Low rating (1-2) should have more negative words
        elif rating <= 2:
            if neg_count > pos_count:
                return 1.0
            elif neg_count == pos_count and neg_count > 0:
                return 0.7
            else:
                return 0.3
        
        # Medium rating (3) can be mixed
        else:
            return 0.8
    
    def _check_repetition(self, text: str) -> float:
        """Check for excessive repetition."""
        words = text.lower().split()
        
        if len(words) < 5:
            return 0.0
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Calculate repetition ratio
        unique_words = len(word_counts)
        total_words = len(words)
        
        repetition_ratio = 1.0 - (unique_words / total_words)
        
        # Check for repeated phrases
        phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrases.append(phrase)
        
        phrase_counts = Counter(phrases)
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)
        
        if repeated_phrases > 0:
            repetition_ratio += 0.2
        
        return min(1.0, repetition_ratio)
    
    def _check_fake_indicators(self, text: str) -> float:
        """Check for words/patterns common in fake reviews."""
        text_lower = text.lower()
        
        fake_count = 0
        for indicator in self.fake_indicators:
            if indicator in text_lower:
                fake_count += 1
        
        # Multiple fake indicators are suspicious
        return min(1.0, fake_count / 5.0)
    
    def analyze_review_batch(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze authenticity for a batch of reviews."""
        analyzed_reviews = []
        
        for review in reviews:
            review_copy = review.copy()
            authenticity = self.analyze_authenticity(review)
            
            review_copy['authenticity'] = {
                'score': authenticity.score,
                'confidence': authenticity.confidence,
                'flags': authenticity.flags,
                'details': authenticity.details
            }
            
            analyzed_reviews.append(review_copy)
        
        return analyzed_reviews
    
    def get_authenticity_summary(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get authenticity summary for a list of reviews."""
        if not reviews:
            return {}
        
        scores = []
        flags_count = Counter()
        
        for review in reviews:
            if 'authenticity' in review:
                auth = review['authenticity']
                scores.append(auth['score'])
                flags_count.update(auth.get('flags', []))
        
        if not scores:
            return {}
        
        summary = {
            'mean_authenticity': float(np.mean(scores)),
            'std_authenticity': float(np.std(scores)),
            'median_authenticity': float(np.median(scores)),
            'low_authenticity_count': sum(1 for s in scores if s < 0.3),
            'medium_authenticity_count': sum(1 for s in scores if 0.3 <= s <= 0.7),
            'high_authenticity_count': sum(1 for s in scores if s > 0.7),
            'total_reviews': len(scores),
            'common_flags': dict(flags_count.most_common(10))
        }
        
        # Calculate percentage of potentially fake reviews
        summary['potentially_fake_ratio'] = summary['low_authenticity_count'] / len(scores)
        
        return summary

def analyze_product_authenticity(product: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze authenticity for all reviews of a product."""
    detector = AuthenticityDetector()
    
    if 'reviews' not in product or not product['reviews']:
        return {}
    
    # Analyze individual reviews
    analyzed_reviews = detector.analyze_review_batch(product['reviews'])
    
    # Get overall authenticity summary
    authenticity_summary = detector.get_authenticity_summary(analyzed_reviews)
    
    return {
        'reviews': analyzed_reviews,
        'authenticity_summary': authenticity_summary
    }