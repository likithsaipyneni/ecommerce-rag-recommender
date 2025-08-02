"""
Visualization utilities for the E-commerce RAG system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not installed")

class ProductVisualization:
    """Visualization utilities for product analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_comparison_radar_chart(self, products: List[Dict[str, Any]], specs_to_compare: List[str] = None) -> Optional[go.Figure]:
        """Create a radar chart comparing product specifications."""
        if not PLOTLY_AVAILABLE or not products:
            return None
        
        if specs_to_compare is None:
            specs_to_compare = ['performance', 'price_value', 'build_quality', 'features', 'user_rating']
        
        fig = go.Figure()
        
        for product in products:
            # Extract or calculate spec values
            values = []
            for spec in specs_to_compare:
                value = self._get_normalized_spec_value(product, spec)
                values.append(value)
            
            # Close the radar chart
            values.append(values[0])
            spec_labels = specs_to_compare + [specs_to_compare[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=spec_labels,
                fill='toself',
                name=product.get('name', 'Unknown Product')[:30],
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Product Comparison Radar Chart",
            font=dict(size=12)
        )
        
        return fig
    
    def create_sentiment_trend_chart(self, reviews: List[Dict[str, Any]]) -> Optional[go.Figure]:
        """Create a sentiment trend chart from reviews."""
        if not PLOTLY_AVAILABLE or not reviews:
            return None
        
        # Prepare data
        dates = []
        sentiments = []
        ratings = []
        
        for review in reviews:
            if 'date' in review and 'sentiment' in review:
                dates.append(review['date'])
                sentiments.append(review['sentiment'].get('score', 0))
                ratings.append(review.get('rating', 0))
        
        if not dates:
            return None
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add sentiment trend
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sentiments,
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            secondary_y=False
        )
        
        # Add rating trend
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ratings,
                mode='lines+markers',
                name='Rating',
                line=dict(color='orange', width=2),
                marker=dict(size=6)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Rating", secondary_y=True)
        
        fig.update_layout(
            title="Sentiment and Rating Trends Over Time",
            hovermode='x unified'
        )
        
        return fig
    
    def create_authenticity_distribution(self, reviews: List[Dict[str, Any]]) -> Optional[go.Figure]:
        """Create authenticity score distribution chart."""
        if not PLOTLY_AVAILABLE or not reviews:
            return None
        
        authenticity_scores = []
        for review in reviews:
            if 'authenticity' in review:
                score = review['authenticity'].get('score', 0.5)
                authenticity_scores.append(score)
        
        if not authenticity_scores:
            return None
        
        fig = go.Figure()
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=authenticity_scores,
            nbinsx=20,
            name='Authenticity Distribution',
            marker_color='green',
            opacity=0.7
        ))
        
        # Add vertical lines for thresholds
        fig.add_vline(x=0.3, line_dash="dash", line_color="red", 
                     annotation_text="Low Authenticity", annotation_position="top")
        fig.add_vline(x=0.7, line_dash="dash", line_color="orange",
                     annotation_text="High Authenticity", annotation_position="top")
        
        fig.update_layout(
            title="Review Authenticity Score Distribution",
            xaxis_title="Authenticity Score",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        
        return fig
    
    def create_price_feature_scatter(self, products: List[Dict[str, Any]], 
                                   feature_name: str = 'performance') -> Optional[go.Figure]:
        """Create price vs feature scatter plot."""
        if not PLOTLY_AVAILABLE or not products:
            return None
        
        prices = []
        features = []
        names = []
        categories = []
        
        for product in products:
            price = product.get('price', 0)
            if price > 0:  # Only include products with valid prices
                prices.append(price)
                features.append(self._get_normalized_spec_value(product, feature_name))
                names.append(product.get('name', 'Unknown'))
                categories.append(product.get('category', 'Unknown'))
        
        if not prices:
            return None
        
        fig = go.Figure()
        
        # Group by category for different colors
        category_colors = {}
        unique_categories = list(set(categories))
        colors = px.colors.qualitative.Set1[:len(unique_categories)]
        
        for i, cat in enumerate(unique_categories):
            category_colors[cat] = colors[i % len(colors)]
        
        for category in unique_categories:
            cat_prices = [prices[i] for i, c in enumerate(categories) if c == category]
            cat_features = [features[i] for i, c in enumerate(categories) if c == category]
            cat_names = [names[i] for i, c in enumerate(categories) if c == category]
            
            fig.add_trace(go.Scatter(
                x=cat_prices,
                y=cat_features,
                mode='markers',
                name=category,
                text=cat_names,
                marker=dict(
                    size=10,
                    color=category_colors[category],
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>Price: $%{x}<br>' + 
                            f'{feature_name.title()}: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Price vs {feature_name.title()} Analysis",
            xaxis_title="Price ($)",
            yaxis_title=f"{feature_name.title()} Score",
            hovermode='closest'
        )
        
        return fig
    
    def create_review_summary_chart(self, products: List[Dict[str, Any]]) -> Optional[go.Figure]:
        """Create a summary chart of review statistics."""
        if not PLOTLY_AVAILABLE or not products:
            return None
        
        product_names = []
        avg_ratings = []
        review_counts = []
        avg_sentiments = []
        avg_authenticity = []
        
        for product in products:
            reviews = product.get('reviews', [])
            if reviews:
                product_names.append(product.get('name', 'Unknown')[:30])
                
                # Calculate averages
                ratings = [r.get('rating', 0) for r in reviews]
                avg_ratings.append(np.mean(ratings) if ratings else 0)
                review_counts.append(len(reviews))
                
                sentiments = [r.get('sentiment', {}).get('score', 0) for r in reviews if 'sentiment' in r]
                avg_sentiments.append(np.mean(sentiments) if sentiments else 0)
                
                authenticity = [r.get('authenticity', {}).get('score', 0.5) for r in reviews if 'authenticity' in r]
                avg_authenticity.append(np.mean(authenticity) if authenticity else 0.5)
        
        if not product_names:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Rating', 'Review Count', 'Average Sentiment', 'Average Authenticity'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces
        fig.add_trace(
            go.Bar(x=product_names, y=avg_ratings, name='Avg Rating', marker_color='blue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=product_names, y=review_counts, name='Review Count', marker_color='green'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=product_names, y=avg_sentiments, name='Avg Sentiment', marker_color='orange'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=product_names, y=avg_authenticity, name='Avg Authenticity', marker_color='red'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Product Review Summary Dashboard",
            showlegend=False,
            height=600
        )
        
        # Update y-axes
        fig.update_yaxes(range=[0, 5], row=1, col=1)  # Rating 0-5
        fig.update_yaxes(range=[-1, 1], row=2, col=1)  # Sentiment -1 to 1
        fig.update_yaxes(range=[0, 1], row=2, col=2)   # Authenticity 0-1
        
        return fig
    
    def _get_normalized_spec_value(self, product: Dict[str, Any], spec_name: str) -> float:
        """Get normalized specification value (0-1 range)."""
        # This is a simplified implementation
        # In practice, you'd have more sophisticated normalization logic
        
        if spec_name == 'price_value':
            price = product.get('price', 0)
            if price > 0:
                # Inverse price value (lower price = higher value)
                return max(0, min(1, 1 - (price / 2000)))  # Normalize to $2000 max
            return 0.5
        
        elif spec_name == 'user_rating':
            reviews = product.get('reviews', [])
            if reviews:
                ratings = [r.get('rating', 0) for r in reviews]
                avg_rating = np.mean(ratings)
                return avg_rating / 5.0  # Normalize to 0-1
            return 0.5
        
        elif spec_name == 'performance':
            # Mock performance score based on specs
            specs = product.get('specifications', {})
            score = 0.5
            
            # Check for performance indicators
            processor = str(specs.get('processor', '')).lower()
            if 'i7' in processor or 'm3' in processor or 'snapdragon' in processor:
                score += 0.2
            if 'i9' in processor or 'm3 max' in processor:
                score += 0.3
            
            memory = str(specs.get('memory', '')).lower()
            if '16gb' in memory or '32gb' in memory or '36gb' in memory:
                score += 0.2
            
            return min(1.0, score)
        
        elif spec_name == 'build_quality':
            # Mock build quality based on brand and reviews
            brand = product.get('brand', '').lower()
            premium_brands = ['apple', 'sony', 'samsung']
            
            score = 0.5
            if any(brand_name in brand for brand_name in premium_brands):
                score += 0.3
            
            # Factor in review sentiment
            reviews = product.get('reviews', [])
            if reviews:
                sentiments = [r.get('sentiment', {}).get('score', 0) for r in reviews if 'sentiment' in r]
                if sentiments:
                    avg_sentiment = np.mean(sentiments)
                    score += 0.2 * max(0, avg_sentiment)
            
            return min(1.0, score)
        
        elif spec_name == 'features':
            # Count of features
            features = product.get('features', [])
            if features:
                return min(1.0, len(features) / 10.0)  # Normalize to 10 features max
            return 0.5
        
        else:
            return 0.5  # Default value

def create_visualization_dashboard(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a complete visualization dashboard."""
    viz = ProductVisualization()
    
    dashboard = {}
    
    # Create various charts
    dashboard['comparison_radar'] = viz.create_comparison_radar_chart(products[:5])  # Top 5 products
    dashboard['price_performance'] = viz.create_price_feature_scatter(products, 'performance')
    dashboard['review_summary'] = viz.create_review_summary_chart(products)
    
    # Create sentiment trends for products with reviews
    products_with_reviews = [p for p in products if p.get('reviews')]
    if products_with_reviews:
        dashboard['sentiment_trends'] = {}
        dashboard['authenticity_distributions'] = {}
        
        for product in products_with_reviews[:3]:  # Top 3 products with reviews
            product_name = product.get('name', 'Unknown')
            dashboard['sentiment_trends'][product_name] = viz.create_sentiment_trend_chart(product['reviews'])
            dashboard['authenticity_distributions'][product_name] = viz.create_authenticity_distribution(product['reviews'])
    
    return dashboard