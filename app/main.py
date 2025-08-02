"""
Main Streamlit application for the E-commerce RAG system.
"""
import streamlit as st
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config.settings import APP_CONFIG, MODEL_CONFIG, get_config
from data.ingestion import ProductDataProcessor
from data.preprocessing import preprocess_products
from models.embeddings import ProductEmbeddingManager
from models.sentiment import analyze_product_sentiment
from models.authenticity import analyze_product_authenticity
from utils.vector_store import get_vector_store
from utils.scoring import create_scoring_pipeline
from utils.visualization import create_visualization_dashboard
from app.components.search import (
    render_search_interface, render_search_results, render_product_details,
    render_search_filters_sidebar, render_search_history
)
from app.components.recommendations import (
    render_recommendation_engine, render_recommendations,
    render_recommendation_filters, render_saved_recommendations
)
from app.components.comparison import (
    render_comparison_interface, render_product_comparison
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceRAGApp:
    """Main application class for the E-commerce RAG system."""
    
    def __init__(self):
        self.setup_streamlit_config()
        self.initialize_system()
    
    def setup_streamlit_config(self):
        """Configure Streamlit settings."""
        st.set_page_config(
            page_title=APP_CONFIG.page_title,
            page_icon=APP_CONFIG.page_icon,
            layout=APP_CONFIG.layout,
            initial_sidebar_state=APP_CONFIG.sidebar_state
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #667eea;
        }
        .search-result {
            border: 1px solid #e9ecef;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            background: white;
        }
        .recommendation-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .comparison-table {
            background: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_system(self):
        """Initialize the RAG system components."""
        if 'system_initialized' not in st.session_state:
            with st.spinner("🚀 Initializing E-commerce RAG system..."):
                try:
                    # Initialize components
                    self.data_processor = ProductDataProcessor()
                    self.embedding_manager = ProductEmbeddingManager()
                    self.vector_store = get_vector_store()
                    self.ranking_system, self.personalization_engine = create_scoring_pipeline()
                    
                    # Load and process sample data
                    self.load_sample_data()
                    
                    st.session_state.system_initialized = True
                    st.session_state.app_instance = self
                    
                    logger.info("System initialized successfully")
                    
                except Exception as e:
                    st.error(f"Failed to initialize system: {e}")
                    logger.error(f"Initialization error: {e}")
                    return
        else:
            # Retrieve existing instance
            if 'app_instance' in st.session_state:
                app_instance = st.session_state.app_instance
                self.data_processor = app_instance.data_processor
                self.embedding_manager = app_instance.embedding_manager
                self.vector_store = app_instance.vector_store
                self.ranking_system = app_instance.ranking_system
                self.personalization_engine = app_instance.personalization_engine
    
    def load_sample_data(self):
        """Load and process sample product data."""
        try:
            # Load products from sample data
            products = self.data_processor.load_products('data/sample_products.json')
            
            if not products:
                logger.warning("No sample products found")
                return
            
            # Preprocess products
            processed_products = preprocess_products(products)
            
            # Create chunks for embedding
            all_chunks = []
            for product in processed_products:
                chunks = self.data_processor.process_product(product)
                all_chunks.extend(chunks)
            
            # Create embeddings if not already cached
            if 'product_embeddings' not in st.session_state:
                embedding_data = self.embedding_manager.create_product_embeddings(all_chunks)
                
                if embedding_data and 'embeddings' in embedding_data:
                    # Store embeddings in vector database
                    success = self.vector_store.add_embeddings(
                        embedding_data['embeddings'],
                        embedding_data['texts'],
                        embedding_data['metadata']
                    )
                    
                    if success:
                        st.session_state.product_embeddings = embedding_data
                        st.session_state.processed_products = processed_products
                        logger.info(f"Loaded {len(processed_products)} products with {len(all_chunks)} chunks")
                    else:
                        logger.error("Failed to store embeddings in vector database")
                else:
                    logger.error("Failed to create embeddings")
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            st.error(f"Failed to load sample data: {e}")
    
    def search_products(self, query: str, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for products based on query and preferences."""
        if not query.strip() or 'product_embeddings' not in st.session_state:
            return []
        
        try:
            # Get embeddings data
            embedding_data = st.session_state.product_embeddings
            
            # Search using embedding manager
            results = self.embedding_manager.search_similar_products(
                query=query,
                product_embeddings=embedding_data,
                top_k=preferences.get('max_results', 10),
                similarity_threshold=MODEL_CONFIG.similarity_threshold
            )
            
            # Apply filters
            filtered_results = self.apply_search_filters(results, preferences)
            
            # Rank results
            ranking_strategy = preferences.get('search_strategy', 'balanced')
            ranked_results = self.ranking_system.rank_results(
                filtered_results,
                query,
                preferences,
                ranking_strategy
            )
            
            # Apply personalization if enabled
            if preferences.get('enable_personalization', True):
                user_prefs = self.extract_user_preferences()
                ranked_results = self.personalization_engine.personalize_results(
                    ranked_results,
                    user_prefs
                )
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            st.error(f"Search failed: {e}")
            return []
    
    def apply_search_filters(self, results: List[Dict[str, Any]], preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply search filters to results."""
        filtered = results.copy()
        
        # Price filter
        max_budget = preferences.get('max_budget', float('inf'))
        if max_budget < float('inf'):
            filtered = [r for r in filtered if r.get('metadata', {}).get('price', 0) <= max_budget]
        
        # Category filter
        categories = preferences.get('categories', [])
        if categories:
            filtered = [r for r in filtered if r.get('metadata', {}).get('category', '') in categories]
        
        # Brand filter
        brands = preferences.get('preferred_brands', [])
        if brands:
            filtered = [r for r in filtered if r.get('metadata', {}).get('brand', '') in brands]
        
        # Rating filter
        min_rating = preferences.get('min_rating', 0.0)
        if min_rating > 0:
            filtered = [r for r in filtered if r.get('metadata', {}).get('rating', 0) >= min_rating]
        
        return filtered
    
    def extract_user_preferences(self) -> Dict[str, Any]:
        """Extract user preferences from session state."""
        # Get search history
        search_history = st.session_state.get('search_history', [])
        
        # Extract preferences from history
        if search_history:
            return self.personalization_engine.extract_user_preferences(
                query_history=[q for q, _ in search_history]
            )
        
        return {}
    
    def get_recommendations(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get personalized product recommendations."""
        if 'processed_products' not in st.session_state:
            return []
        
        try:
            products = st.session_state.processed_products
            
            # Convert products to recommendation format
            recommendations = []
            for product in products:
                # Create a mock recommendation based on preferences
                relevance_score = self.calculate_recommendation_relevance(product, preferences)
                
                if relevance_score > 0.3:  # Threshold for recommendations
                    rec = {
                        'metadata': {
                            'product_name': product.get('name', 'Unknown'),
                            'brand': product.get('brand', 'Unknown'),
                            'category': product.get('category', 'Unknown'),
                            'price': product.get('price', 0),
                            'rating': self.calculate_average_rating(product.get('reviews', [])),
                            'authenticity_score': self.calculate_average_authenticity(product.get('reviews', [])),
                            'sentiment_score': self.calculate_average_sentiment(product.get('reviews', []))
                        },
                        'text': product.get('description', ''),
                        'relevance_score': relevance_score,
                        'similarity': relevance_score  # Mock similarity
                    }
                    recommendations.append(rec)
            
            # Sort by relevance score
            recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return recommendations[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return []
    
    def calculate_recommendation_relevance(self, product: Dict[str, Any], preferences: Dict[str, Any]) -> float:
        """Calculate how relevant a product is to user preferences."""
        score = 0.5  # Base score
        
        # Budget alignment
        budget_pref = preferences.get('budget_preference', '')
        price = product.get('price', 0)
        
        if 'Budget-conscious' in budget_pref and price < 500:
            score += 0.2
        elif 'Premium' in budget_pref and price > 1000:
            score += 0.2
        elif 'Balanced' in budget_pref and 300 <= price <= 1200:
            score += 0.2
        
        # Feature priorities
        feature_priorities = preferences.get('feature_priorities', [])
        features = product.get('features', [])
        
        for priority in feature_priorities:
            if any(priority.lower() in feature.lower() for feature in features):
                score += 0.1
        
        # Usage context
        usage_context = preferences.get('usage_context', '')
        description = product.get('description', '').lower()
        
        context_keywords = {
            'Work/Professional': ['professional', 'business', 'productivity'],
            'Gaming': ['gaming', 'performance', 'graphics'],
            'Content Creation': ['creative', 'editing', 'design'],
            'Travel': ['portable', 'lightweight', 'compact']
        }
        
        if usage_context in context_keywords:
            keywords = context_keywords[usage_context]
            if any(keyword in description for keyword in keywords):
                score += 0.2
        
        return min(1.0, score)
    
    def calculate_average_rating(self, reviews: List[Dict[str, Any]]) -> float:
        """Calculate average rating from reviews."""
        if not reviews:
            return 0.0
        
        ratings = [r.get('rating', 0) for r in reviews]
        return sum(ratings) / len(ratings) if ratings else 0.0
    
    def calculate_average_authenticity(self, reviews: List[Dict[str, Any]]) -> float:
        """Calculate average authenticity score from reviews."""
        if not reviews:
            return 0.7  # Default authenticity
        
        # Mock authenticity calculation
        return 0.75  # Placeholder
    
    def calculate_average_sentiment(self, reviews: List[Dict[str, Any]]) -> float:
        """Calculate average sentiment from reviews."""
        if not reviews:
            return 0.0
        
        # Mock sentiment calculation
        return 0.2  # Placeholder positive sentiment
    
    def run(self):
        """Run the main application."""
        # Sidebar navigation
        st.sidebar.title("🛒 Navigation")
        
        page = st.sidebar.radio(
            "Choose a feature:",
            ["🔍 Search Products", "🎯 Recommendations", "⚖️ Compare Products", "📊 Analytics"],
            index=0
        )
        
        # Render appropriate page
        if page == "🔍 Search Products":
            self.render_search_page()
        elif page == "🎯 Recommendations":
            self.render_recommendations_page()
        elif page == "⚖️ Compare Products":
            self.render_comparison_page()
        elif page == "📊 Analytics":
            self.render_analytics_page()
        
        # Render sidebar components
        self.render_sidebar_components()
    
    def render_search_page(self):
        """Render the search page."""
        # Search interface
        search_data = render_search_interface()
        
        if search_data['search_clicked'] and search_data['query']:
            # Add to search history
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            st.session_state.search_history.append((
                search_data['query'],
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))
            
            # Perform search
            with st.spinner("🔍 Searching products..."):
                results = self.search_products(search_data['query'], search_data['preferences'])
            
            if results:
                # Render results
                selected_product = render_search_results(results, search_data['query'])
                
                # Show product details if selected
                if selected_product:
                    with st.expander("📋 Product Details", expanded=True):
                        render_product_details(selected_product)
            
            # Store results in session state
            st.session_state.last_search_results = results
    
    def render_recommendations_page(self):
        """Render the recommendations page."""
        # Recommendation interface
        rec_data = render_recommendation_engine()
        
        if rec_data['get_recommendations']:
            # Get recommendations
            with st.spinner("🎯 Generating personalized recommendations..."):
                recommendations = self.get_recommendations(rec_data['preferences'])
            
            if recommendations:
                render_recommendations(recommendations, rec_data['preferences'])
            
            # Store recommendations in session state
            st.session_state.last_recommendations = recommendations
    
    def render_comparison_page(self):
        """Render the comparison page."""
        comparison_data = render_comparison_interface()
        
        if comparison_data['has_comparisons']:
            render_product_comparison(
                comparison_data['comparison_list'],
                comparison_data['comparison_view']
            )
    
    def render_analytics_page(self):
        """Render the analytics page."""
        st.header("📊 Analytics Dashboard")
        st.markdown("System performance and insights.")
        
        # System stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'processed_products' in st.session_state:
                product_count = len(st.session_state.processed_products)
                st.metric("Products Loaded", product_count)
        
        with col2:
            if 'product_embeddings' in st.session_state:
                embedding_count = len(st.session_state.product_embeddings.get('texts', []))
                st.metric("Text Chunks", embedding_count)
        
        with col3:
            search_count = len(st.session_state.get('search_history', []))
            st.metric("Searches Performed", search_count)
        
        with col4:
            saved_count = len(st.session_state.get('saved_products', []))
            st.metric("Saved Products", saved_count)
        
        # Vector store info
        if hasattr(self, 'vector_store'):
            vector_info = self.vector_store.get_collection_info()
            if vector_info:
                st.subheader("🗄️ Vector Database Status")
                st.json(vector_info)
        
        # Configuration
        st.subheader("⚙️ System Configuration")
        config = get_config()
        st.json({
            'model_config': {
                'embedding_model': config['model'].embedding_model,
                'embedding_dimension': config['model'].embedding_dimension,
                'similarity_threshold': config['model'].similarity_threshold
            },
            'database_config': {
                'chroma_db_path': config['database'].chroma_db_path,
                'collection_name': config['database'].collection_name
            }
        })
    
    def render_sidebar_components(self):
        """Render sidebar components."""
        # Search filters
        filters = render_search_filters_sidebar()
        
        # Search history
        render_search_history()
        
        # Recommendation filters
        rec_filters = render_recommendation_filters()
        
        # Saved products
        render_saved_recommendations()
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ System Info")
        
        if st.sidebar.button("🔄 Reset System"):
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'system_initialized':
                    del st.session_state[key]
            st.experimental_rerun()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("Built with ❤️ using Streamlit")
        st.sidebar.markdown("Powered by RAG + ML")

def main():
    """Main entry point."""
    try:
        app = EcommerceRAGApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()