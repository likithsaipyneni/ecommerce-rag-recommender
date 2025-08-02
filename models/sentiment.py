"""
Sentiment analysis models for product reviews.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass
from collections import Counter
import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not installed. Install with: pip install textblob")

@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    score: float  # -1 to 1, negative to positive
    confidence: float  # 0 to 1
    label: str  # positive, negative, neutral
    details: Dict[str, Any]

class SentimentAnalyzer:
    """Multi-model sentiment analysis for product reviews."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vader_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models."""
        if VADER_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("VADER sentiment analyzer initialized")
            except Exception as e:
                self.logger.error(f"Error initializing VADER: {e}")
        
        self.logger.info(f"Sentiment analyzer initialized - VADER: {VADER_AVAILABLE}, TextBlob: {TEXTBLOB_AVAILABLE}")
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using multiple approaches."""
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label="neutral",
                details={"error": "Empty text"}
            )
        
        results = {}
        
        # VADER analysis
        if self.vader_analyzer:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                results['vader'] = vader_scores
            except Exception as e:
                self.logger.error(f"VADER analysis error: {e}")
        
        # TextBlob analysis
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                results['textblob'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                self.logger.error(f"TextBlob analysis error: {e}")
        
        # Rule-based analysis
        rule_based = self._rule_based_sentiment(text)
        results['rule_based'] = rule_based
        
        # Combine results
        final_sentiment = self._combine_sentiments(results)
        
        return final_sentiment
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis."""
        text_lower = text.lower()
        
        # Positive words with weights
        positive_words = {
            'excellent': 3, 'amazing': 3, 'outstanding': 3, 'fantastic': 3,
            'great': 2, 'good': 2, 'nice': 2, 'love': 2, 'wonderful': 2,
            'perfect': 2, 'awesome': 2, 'brilliant': 2, 'superb': 2,
            'satisfied': 1, 'happy': 1, 'pleased': 1, 'recommend': 1,
            'quality': 1, 'fast': 1, 'reliable': 1, 'helpful': 1
        }
        
        # Negative words with weights
        negative_words = {
            'terrible': -3, 'awful': -3, 'horrible': -3, 'worst': -3,
            'hate': -3, 'disgusting': -3, 'pathetic': -3,
            'bad': -2, 'poor': -2, 'disappointed': -2, 'waste': -2,
            'useless': -2, 'broken': -2, 'defective': -2, 'cheap': -2,
            'slow': -1, 'expensive': -1, 'difficult': -1, 'problem': -1,
            'issue': -1, 'concern': -1, 'unfortunately': -1
        }
        
        # Intensifiers
        intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'absolutely': 2.0, 'totally': 1.8,
            'completely': 1.8, 'really': 1.3, 'quite': 1.2, 'pretty': 1.1,
            'somewhat': 0.8, 'rather': 0.9, 'fairly': 0.9
        }
        
        # Negations
        negation_words = {
            'not', 'no', 'never', 'nothing', 'none', 'nobody', 'nowhere',
            'neither', 'nor', 'without', 'lack', 'lacking', 'miss', 'missing'
        }
        
        words = text_lower.split()
        score = 0.0
        positive_count = 0
        negative_count = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            word_score = 0
            intensifier = 1.0
            
            # Check for intensifiers
            if i > 0 and words[i-1] in intensifiers:
                intensifier = intensifiers[words[i-1]]
            
            # Check for negation in previous 2 words
            negated = False
            for j in range(max(0, i-2), i):
                if words[j] in negation_words:
                    negated = True
                    break
            
            # Score the word
            if word in positive_words:
                word_score = positive_words[word] * intensifier
                positive_count += 1
            elif word in negative_words:
                word_score = negative_words[word] * intensifier
                negative_count += 1
            
            # Apply negation
            if negated:
                word_score = -word_score
            
            score += word_score
            i += 1
        
        # Normalize score
        total_words = positive_count + negative_count
        if total_words > 0:
            normalized_score = score / (total_words * 3)  # Max weight is 3
        else:
            normalized_score = 0.0
        
        # Clamp to [-1, 1]
        normalized_score = max(-1.0, min(1.0, normalized_score))
        
        return {
            'score': normalized_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_sentiment_words': total_words
        }
    
    def _combine_sentiments(self, results: Dict[str, Any]) -> SentimentResult:
        """Combine multiple sentiment analysis results."""
        scores = []
        confidences = []
        
        # VADER
        if 'vader' in results:
            vader = results['vader']
            scores.append(vader['compound'])
            # VADER confidence based on the sum of absolute values
            conf = min(1.0, abs(vader['pos']) + abs(vader['neg']) + abs(vader['neu']))
            confidences.append(conf)
        
        # TextBlob
        if 'textblob' in results:
            tb = results['textblob']
            scores.append(tb['polarity'])
            # Confidence based on subjectivity (more subjective = more confident about sentiment)
            confidences.append(tb['subjectivity'])
        
        # Rule-based
        if 'rule_based' in results:
            rb = results['rule_based']
            scores.append(rb['score'])
            # Confidence based on number of sentiment words found
            conf = min(1.0, rb['total_sentiment_words'] / 10.0)
            confidences.append(conf)
        
        if not scores:
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label="neutral",
                details={"error": "No sentiment analysis available"}
            )
        
        # Weighted average based on confidence
        if sum(confidences) > 0:
            weights = np.array(confidences) / sum(confidences)
            final_score = np.average(scores, weights=weights)
            final_confidence = np.mean(confidences)
        else:
            final_score = np.mean(scores)
            final_confidence = 0.5
        
        # Determine label
        if final_score > 0.1:
            label = "positive"
        elif final_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(
            score=float(final_score),
            confidence=float(final_confidence),
            label=label,
            details=results
        )
    
    def analyze_review_batch(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of reviews."""
        analyzed_reviews = []
        
        for review in reviews:
            review_copy = review.copy()
            text = review.get('text', '')
            
            if text:
                sentiment = self.analyze_sentiment(text)
                review_copy['sentiment'] = {
                    'score': sentiment.score,
                    'confidence': sentiment.confidence,
                    'label': sentiment.label
                }
            else:
                review_copy['sentiment'] = {
                    'score': 0.0,
                    'confidence': 0.0,
                    'label': 'neutral'
                }
            
            analyzed_reviews.append(review_copy)
        
        return analyzed_reviews
    
    def get_sentiment_summary(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get sentiment summary for a list of reviews."""
        if not reviews:
            return {}
        
        sentiments = []
        ratings = []
        
        for review in reviews:
            if 'sentiment' in review:
                sentiments.append(review['sentiment']['score'])
            
            if 'rating' in review:
                ratings.append(review['rating'])
        
        if not sentiments:
            return {}
        
        # Basic statistics
        summary = {
            'mean_sentiment': float(np.mean(sentiments)),
            'std_sentiment': float(np.std(sentiments)),
            'median_sentiment': float(np.median(sentiments)),
            'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments),
            'neutral_ratio': sum(1 for s in sentiments if -0.1 <= s <= 0.1) / len(sentiments),
            'total_reviews': len(sentiments)
        }
        
        # Rating correlation if available
        if ratings and len(ratings) == len(sentiments):
            correlation = np.corrcoef(ratings, sentiments)[0, 1]
            summary['rating_sentiment_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return summary

def analyze_product_sentiment(product: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze sentiment for all reviews of a product."""
    analyzer = SentimentAnalyzer()
    
    if 'reviews' not in product or not product['reviews']:
        return {}
    
    # Analyze individual reviews
    analyzed_reviews = analyzer.analyze_review_batch(product['reviews'])
    
    # Get overall sentiment summary
    sentiment_summary = analyzer.get_sentiment_summary(analyzed_reviews)
    
    return {
        'reviews': analyzed_reviews,
        'sentiment_summary': sentiment_summary
    }