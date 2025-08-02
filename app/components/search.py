"""
Search components for the Streamlit app.
"""
import streamlit as st
from typing import List, Dict, Any, Optional
import logging

def render_search_interface() -> Dict[str, Any]:
    """Render the main search interface."""
    st.title("🛒 E-commerce Product Search Assistant")
    st.markdown("Ask me anything about products - I'll find the best matches and provide intelligent recommendations!")
    
    # Main search input
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g., 'Best laptop for video editing under $2000' or 'Wireless headphones with good battery life'",
        help="Describe what you're looking for in natural language"
    )
    
    # Advanced search options in expandable section
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_budget = st.number_input(
                "Maximum Budget ($)",
                min_value=0,
                max_value=10000,
                value=1000,
                step=100,
                help="Set your maximum budget for filtering results"
            )
            
            categories = st.multiselect(
                "Product Categories",
                options=["Laptops", "Smartphones", "Headphones", "Tablets", "Cameras", "Gaming"],
                help="Filter by specific product categories"
            )
        
        with col2:
            preferred_brands = st.multiselect(
                "Preferred Brands",
                options=["Apple", "Samsung", "Sony", "Dell", "HP", "Microsoft", "Google", "Lenovo"],
                help="Select brands you prefer"
            )
            
            min_rating = st.slider(
                "Minimum Rating",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5,
                help="Filter products by minimum user rating"
            )
    
    # Search strategy selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_strategy = st.selectbox(
            "Search Strategy",
            options=["balanced", "price_focused", "rating_focused", "authenticity_focused"],
            index=0,
            help="Choose how to prioritize search results"
        )
    
    with col2:
        max_results = st.slider(
            "Max Results",
            min_value=5,
            max_value=50,
            value=10,
            help="Maximum number of results to display"
        )
    
    with col3:
        enable_personalization = st.checkbox(
            "Enable Personalization",
            value=True,
            help="Use your search history to personalize results"
        )
    
    # Search button
    search_clicked = st.button("🔍 Search Products", type="primary", use_container_width=True)
    
    return {
        'query': query.strip(),
        'search_clicked': search_clicked,
        'preferences': {
            'max_budget': max_budget,
            'categories': categories,
            'preferred_brands': preferred_brands,
            'min_rating': min_rating,
            'search_strategy': search_strategy,
            'max_results': max_results,
            'enable_personalization': enable_personalization
        }
    }

def render_search_results(results: List[Dict[str, Any]], query: str) -> Optional[str]:
    """Render search results with interactive elements."""
    if not results:
        st.warning("No results found for your query. Try adjusting your search terms or filters.")
        return None
    
    st.success(f"Found {len(results)} results for: **{query}**")
    
    # Results summary
    with st.expander("📊 Results Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = sum(r.get('metadata', {}).get('price', 0) for r in results) / len(results)
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col2:
            categories = [r.get('metadata', {}).get('category', '') for r in results]
            unique_categories = len(set(categories))
            st.metric("Categories", unique_categories)
        
        with col3:
            avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)
            st.metric("Avg Similarity", f"{avg_similarity:.3f}")
        
        with col4:
            highly_rated = sum(1 for r in results if r.get('metadata', {}).get('rating', 0) >= 4.0)
            st.metric("Highly Rated", f"{highly_rated}/{len(results)}")
    
    selected_product = None
    
    # Display results
    for i, result in enumerate(results):
        metadata = result.get('metadata', {})
        product_name = metadata.get('product_name', 'Unknown Product')
        category = metadata.get('category', 'Unknown')
        brand = metadata.get('brand', 'Unknown')
        price = metadata.get('price', 0)
        similarity = result.get('similarity', 0.0)
        
        with st.container():
            st.markdown("---")
            
            # Product header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"{i+1}. {product_name}")
                st.caption(f"**{brand}** • {category}")
            
            with col2:
                st.metric("Price", f"${price:.2f}")
            
            with col3:
                st.metric("Match", f"{similarity:.1%}")
            
            # Product details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display the matched text
                content = result.get('text', '')
                if content:
                    st.markdown(f"**Relevant Information:**")
                    st.markdown(f"*{content[:300]}...*" if len(content) > 300 else f"*{content}*")
                
                # Show additional metadata if available
                if 'rating' in metadata:
                    rating = metadata['rating']
                    st.markdown(f"⭐ **Rating:** {rating}/5")
                
                if 'sentiment_score' in metadata:
                    sentiment = metadata['sentiment_score']
                    sentiment_emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😞"
                    st.markdown(f"{sentiment_emoji} **Sentiment:** {sentiment:.2f}")
                
                if 'authenticity_score' in metadata:
                    auth_score = metadata['authenticity_score']
                    auth_emoji = "✅" if auth_score > 0.7 else "⚠️" if auth_score > 0.3 else "❌"
                    st.markdown(f"{auth_emoji} **Authenticity:** {auth_score:.2f}")
            
            with col2:
                # Action buttons
                if st.button(f"View Details", key=f"details_{i}", use_container_width=True):
                    selected_product = result
                
                if st.button(f"Compare", key=f"compare_{i}", use_container_width=True):
                    if 'comparison_list' not in st.session_state:
                        st.session_state.comparison_list = []
                    
                    if result not in st.session_state.comparison_list:
                        st.session_state.comparison_list.append(result)
                        st.success(f"Added {product_name} to comparison!")
                    else:
                        st.info("Already in comparison list")
                
                if st.button(f"Save", key=f"save_{i}", use_container_width=True):
                    if 'saved_products' not in st.session_state:
                        st.session_state.saved_products = []
                    
                    if result not in st.session_state.saved_products:
                        st.session_state.saved_products.append(result)
                        st.success("Saved!")
                    else:
                        st.info("Already saved")
    
    return selected_product

def render_product_details(product: Dict[str, Any]):
    """Render detailed product information."""
    if not product:
        return
    
    metadata = product.get('metadata', {})
    product_name = metadata.get('product_name', 'Unknown Product')
    
    st.header(f"📋 Product Details: {product_name}")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Brand", metadata.get('brand', 'Unknown'))
        st.metric("Category", metadata.get('category', 'Unknown'))
    
    with col2:
        st.metric("Price", f"${metadata.get('price', 0):.2f}")
        if 'rating' in metadata:
            st.metric("Rating", f"{metadata['rating']}/5 ⭐")
    
    with col3:
        if 'sentiment_score' in metadata:
            sentiment = metadata['sentiment_score']
            st.metric("Sentiment", f"{sentiment:.2f}")
        
        if 'authenticity_score' in metadata:
            auth = metadata['authenticity_score']
            st.metric("Authenticity", f"{auth:.2f}")
    
    # Content sections
    tab1, tab2, tab3 = st.tabs(["📝 Description", "🔧 Specifications", "💬 Reviews"])
    
    with tab1:
        content = product.get('text', '')
        chunk_type = metadata.get('chunk_type', 'unknown')
        
        st.markdown(f"**Content Type:** {chunk_type.title()}")
        st.markdown(content)
    
    with tab2:
        st.markdown("**Technical Specifications:**")
        
        # Mock specifications display
        specs = {
            'Processor': 'Intel Core i7-12700H',
            'Memory': '16GB DDR4',
            'Storage': '512GB SSD',
            'Display': '15.6" Full HD',
            'Battery': 'Up to 8 hours',
            'Weight': '4.2 lbs'
        }
        
        for spec, value in specs.items():
            st.markdown(f"• **{spec}:** {value}")
    
    with tab3:
        st.markdown("**Customer Reviews:**")
        
        # Mock reviews display
        reviews = [
            {"rating": 5, "text": "Excellent product! Highly recommended.", "date": "2024-01-15"},
            {"rating": 4, "text": "Good quality but a bit expensive.", "date": "2024-01-10"},
            {"rating": 5, "text": "Perfect for my needs. Fast delivery too.", "date": "2024-01-08"}
        ]
        
        for i, review in enumerate(reviews):
            with st.container():
                st.markdown(f"**Review {i+1}** ⭐ {review['rating']}/5 • {review['date']}")
                st.markdown(f"*{review['text']}*")
                st.markdown("---")

def render_search_filters_sidebar():
    """Render search filters in the sidebar."""
    st.sidebar.header("🔍 Search Filters")
    
    # Quick filters
    st.sidebar.subheader("Quick Filters")
    
    price_range = st.sidebar.slider(
        "Price Range",
        min_value=0,
        max_value=5000,
        value=(0, 2000),
        step=100,
        format="$%d"
    )
    
    category_filter = st.sidebar.multiselect(
        "Categories",
        options=["Laptops", "Smartphones", "Headphones", "Tablets"],
        default=[]
    )
    
    brand_filter = st.sidebar.multiselect(
        "Brands",
        options=["Apple", "Samsung", "Sony", "Dell"],
        default=[]
    )
    
    # Rating filter
    min_rating_filter = st.sidebar.slider(
        "Minimum Rating",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.5
    )
    
    # Authenticity filter
    min_authenticity = st.sidebar.slider(
        "Minimum Authenticity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Filter out potentially fake reviews"
    )
    
    return {
        'price_range': price_range,
        'categories': category_filter,
        'brands': brand_filter,
        'min_rating': min_rating_filter,
        'min_authenticity': min_authenticity
    }

def render_search_history():
    """Render search history component."""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if st.session_state.search_history:
        st.sidebar.subheader("📚 Recent Searches")
        
        for i, (query, timestamp) in enumerate(st.session_state.search_history[-5:]):  # Last 5 searches
            if st.sidebar.button(f"🔍 {query[:30]}...", key=f"history_{i}"):
                st.session_state.current_query = query
                st.experimental_rerun()
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.search_history = []
        st.experimental_rerun()