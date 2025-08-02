"""
Scoring utilities for ranking and relevance assessment.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import math

class RelevanceScorer:
    """Calculates relevance scores for search results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_relevance_score(self, 
                                query: str, 
                                result: Dict[str, Any], 
                                user_preferences: Dict[str, Any] = None) -> float:
        """Calculate overall relevance score for a search result."""
        if not result:
            return 0.0
        
        # Base similarity score from vector search
        similarity_score = result.get('similarity', 0.0)
        
        # Metadata-based scoring
        metadata_score = self._calculate_metadata_score(result.get('metadata', {}), user_preferences or {})
        
        # Text quality score
        text_score = self._calculate_text_quality_score(result.get('text', ''))
        
        # Combine scores with weights
        weights = {
            'similarity': 0.5,
            'metadata': 0.3,
            'text_quality': 0.2
        }
        
        final_score = (
            weights['similarity'] * similarity_score +
            weights['metadata'] * metadata_score +
            weights['text_quality'] * text_score
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_metadata_score(self, metadata: Dict[str, Any], user_preferences: Dict[str, Any]) -> float:
        """Calculate score based on metadata and user preferences."""
        score = 0.5  # Base score
        
        # Price preference scoring
        if 'price' in metadata and 'max_budget' in user_preferences:
            product_price = float(metadata.get('price', 0))
            max_budget = float(user_preferences.get('max_budget', float('inf')))
            
            if product_price <= max_budget:
                # Reward products within budget
                price_ratio = product_price / max_budget if max_budget > 0 else 0
                score += 0.2 * (1.0 - price_ratio)  # Cheaper products get higher score
            else:
                # Penalize products over budget
                score -= 0.3
        
        # Brand preference
        if 'brand' in metadata and 'preferred_brands' in user_preferences:
            preferred_brands = user_preferences.get('preferred_brands', [])
            if isinstance(preferred_brands, str):
                preferred_brands = [preferred_brands]
            
            product_brand = metadata.get('brand', '').lower()
            if any(brand.lower() in product_brand for brand in preferred_brands):
                score += 0.2
        
        # Category relevance
        if 'category' in metadata and 'target_category' in user_preferences:
            target_category = user_preferences.get('target_category', '').lower()
            product_category = metadata.get('category', '').lower()
            
            if target_category in product_category or product_category in target_category:
                score += 0.2
        
        # Authenticity and sentiment bonuses
        if 'authenticity_score' in metadata:
            auth_score = float(metadata.get('authenticity_score', 0.5))
            score += 0.1 * auth_score
        
        if 'sentiment_score' in metadata:
            sentiment = float(metadata.get('sentiment_score', 0.0))
            if sentiment > 0:
                score += 0.1 * min(sentiment, 1.0)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate score based on text quality indicators."""
        if not text:
            return 0.0
        
        score = 0.5
        
        # Length appropriateness
        word_count = len(text.split())
        if 10 <= word_count <= 200:
            score += 0.2
        elif word_count < 5:
            score -= 0.3
        
        # Information density (variety of words)
        words = text.lower().split()
        unique_words = set(words)
        if words:
            diversity_ratio = len(unique_words) / len(words)
            score += 0.2 * diversity_ratio
        
        # Readability indicators
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences > 0 and word_count > 0:
            avg_sentence_length = word_count / sentences
            if 5 <= avg_sentence_length <= 25:  # Reasonable sentence length
                score += 0.1
        
        return min(1.0, max(0.0, score))

class RankingSystem:
    """Advanced ranking system for search results."""
    
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.logger = logging.getLogger(__name__)
    
    def rank_results(self, 
                    results: List[Dict[str, Any]], 
                    query: str, 
                    user_preferences: Dict[str, Any] = None,
                    ranking_strategy: str = 'balanced') -> List[Dict[str, Any]]:
        """Rank search results using specified strategy."""
        if not results:
            return []
        
        user_prefs = user_preferences or {}
        
        # Calculate scores for each result
        scored_results = []
        for result in results:
            relevance_score = self.relevance_scorer.calculate_relevance_score(
                query, result, user_prefs
            )
            
            scored_result = result.copy()
            scored_result['relevance_score'] = relevance_score
            scored_result['final_score'] = self._apply_ranking_strategy(
                scored_result, ranking_strategy, user_prefs
            )
            scored_results.append(scored_result)
        
        # Sort by final score
        ranked_results = sorted(
            scored_results, 
            key=lambda x: x['final_score'], 
            reverse=True
        )
        
        # Add ranking position
        for i, result in enumerate(ranked_results):
            result['rank'] = i + 1
        
        self.logger.info(f"Ranked {len(ranked_results)} results using {ranking_strategy} strategy")
        return ranked_results
    
    def _apply_ranking_strategy(self, 
                               result: Dict[str, Any], 
                               strategy: str, 
                               user_preferences: Dict[str, Any]) -> float:
        """Apply specific ranking strategy to calculate final score."""
        base_score = result.get('relevance_score', 0.0)
        similarity = result.get('similarity', 0.0)
        metadata = result.get('metadata', {})
        
        if strategy == 'similarity_only':
            return similarity
        
        elif strategy == 'price_focused':
            # Prioritize products within budget and lower prices
            price = float(metadata.get('price', float('inf')))
            max_budget = float(user_preferences.get('max_budget', float('inf')))
            
            if price <= max_budget:
                price_score = 1.0 - (price / max_budget) if max_budget > 0 else 0.5
                return 0.3 * base_score + 0.7 * price_score
            else:
                return 0.1 * base_score  # Heavy penalty for over-budget
        
        elif strategy == 'rating_focused':
            # Prioritize highly rated products
            rating = float(metadata.get('rating', 0))
            sentiment_score = float(metadata.get('sentiment_score', 0))
            
            rating_score = rating / 5.0 if rating > 0 else 0.5
            sentiment_boost = max(0, sentiment_score) * 0.5
            
            return 0.4 * base_score + 0.4 * rating_score + 0.2 * sentiment_boost
        
        elif strategy == 'authenticity_focused':
            # Prioritize authentic reviews and reliable information
            auth_score = float(metadata.get('authenticity_score', 0.5))
            verified = metadata.get('verified', False)
            
            authenticity_boost = auth_score * 0.5
            if verified:
                authenticity_boost += 0.2
            
            return 0.6 * base_score + 0.4 * authenticity_boost
        
        elif strategy == 'balanced':
            # Balanced approach considering all factors
            return base_score
        
        else:
            self.logger.warning(f"Unknown ranking strategy: {strategy}")
            return base_score
    
    def diversify_results(self, 
                         results: List[Dict[str, Any]], 
                         diversity_factor: float = 0.3,
                         max_per_category: int = 3) -> List[Dict[str, Any]]:
        """Diversify results to avoid over-representation of similar products."""
        if not results or diversity_factor <= 0:
            return results
        
        diversified = []
        category_counts = defaultdict(int)
        brand_counts = defaultdict(int)
        
        # Sort by score first
        sorted_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
        
        for result in sorted_results:
            metadata = result.get('metadata', {})
            category = metadata.get('category', 'unknown').lower()
            brand = metadata.get('brand', 'unknown').lower()
            
            # Check diversity constraints
            category_penalty = min(category_counts[category] * diversity_factor, 0.5)
            brand_penalty = min(brand_counts[brand] * diversity_factor * 0.5, 0.3)
            
            # Apply penalties
            original_score = result.get('final_score', 0)
            diversified_score = original_score - category_penalty - brand_penalty
            
            # Skip if too many from same category
            if category_counts[category] >= max_per_category:
                continue
            
            result_copy = result.copy()
            result_copy['diversified_score'] = diversified_score
            result_copy['category_penalty'] = category_penalty
            result_copy['brand_penalty'] = brand_penalty
            
            diversified.append(result_copy)
            category_counts[category] += 1
            brand_counts[brand] += 1
        
        # Re-sort by diversified score
        diversified.sort(key=lambda x: x.get('diversified_score', 0), reverse=True)
        
        self.logger.info(f"Diversified results: {len(results)} -> {len(diversified)}")
        return diversified

class PersonalizationEngine:
    """Personalization engine for user-specific recommendations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_user_preferences(self, 
                                query_history: List[str] = None,
                                interaction_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract user preferences from history."""
        preferences = {
            'preferred_brands': [],
            'target_categories': [],
            'price_sensitivity': 'medium',  # low, medium, high
            'quality_focus': 'balanced',    # price, quality, balanced
            'feature_priorities': []
        }
        
        if query_history:
            # Analyze query patterns
            all_queries = ' '.join(query_history).lower()
            
            # Extract brand mentions
            common_brands = ['apple', 'samsung', 'sony', 'dell', 'hp', 'microsoft', 'google']
            for brand in common_brands:
                if brand in all_queries:
                    preferences['preferred_brands'].append(brand)
            
            # Extract price sensitivity
            price_keywords = ['cheap', 'budget', 'affordable', 'expensive', 'premium', 'high-end']
            budget_mentions = sum(1 for keyword in ['cheap', 'budget', 'affordable'] if keyword in all_queries)
            premium_mentions = sum(1 for keyword in ['expensive', 'premium', 'high-end'] if keyword in all_queries)
            
            if budget_mentions > premium_mentions:
                preferences['price_sensitivity'] = 'high'
            elif premium_mentions > budget_mentions:
                preferences['price_sensitivity'] = 'low'
        
        if interaction_history:
            # Analyze interaction patterns
            categories = [item.get('metadata', {}).get('category', '') for item in interaction_history]
            category_counts = defaultdict(int)
            for cat in categories:
                if cat:
                    category_counts[cat.lower()] += 1
            
            # Most common categories
            preferences['target_categories'] = [
                cat for cat, count in category_counts.most_common(3)
            ]
        
        return preferences
    
    def personalize_results(self, 
                           results: List[Dict[str, Any]], 
                           user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply personalization to search results."""
        if not results or not user_preferences:
            return results
        
        personalized = []
        
        for result in results:
            result_copy = result.copy()
            
            # Calculate personalization boost
            boost = self._calculate_personalization_boost(result_copy, user_preferences)
            
            # Apply boost to final score
            original_score = result_copy.get('final_score', 0)
            personalized_score = min(1.0, original_score * (1 + boost))
            
            result_copy['personalized_score'] = personalized_score
            result_copy['personalization_boost'] = boost
            
            personalized.append(result_copy)
        
        # Re-sort by personalized score
        personalized.sort(key=lambda x: x.get('personalized_score', 0), reverse=True)
        
        return personalized
    
    def _calculate_personalization_boost(self, 
                                       result: Dict[str, Any], 
                                       preferences: Dict[str, Any]) -> float:
        """Calculate personalization boost for a result."""
        boost = 0.0
        metadata = result.get('metadata', {})
        
        # Brand preference boost
        preferred_brands = preferences.get('preferred_brands', [])
        result_brand = metadata.get('brand', '').lower()
        if any(brand.lower() in result_brand for brand in preferred_brands):
            boost += 0.2
        
        # Category preference boost
        target_categories = preferences.get('target_categories', [])
        result_category = metadata.get('category', '').lower()
        if any(cat.lower() in result_category for cat in target_categories):
            boost += 0.15
        
        # Price sensitivity adjustment
        price_sensitivity = preferences.get('price_sensitivity', 'medium')
        price = float(metadata.get('price', 0))
        
        if price_sensitivity == 'high':  # Budget-conscious
            if price < 100:
                boost += 0.1
            elif price > 500:
                boost -= 0.1
        elif price_sensitivity == 'low':  # Premium-focused
            if price > 500:
                boost += 0.1
            elif price < 100:
                boost -= 0.1
        
        return min(0.5, max(-0.3, boost))

def create_scoring_pipeline(user_preferences: Dict[str, Any] = None, 
                          ranking_strategy: str = 'balanced') -> Tuple[RankingSystem, PersonalizationEngine]:
    """Create a complete scoring pipeline."""
    ranking_system = RankingSystem()
    personalization_engine = PersonalizationEngine()
    
    return ranking_system, personalization_engine