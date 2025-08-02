"""
Recommendation components for the Streamlit app.
"""
import streamlit as st
from typing import List, Dict, Any, Optional
import random
import logging

def render_recommendation_engine() -> Dict[str, Any]:
    """Render the recommendation engine interface."""
    st.header("🎯 Personalized Recommendations")
    st.markdown("Get intelligent product recommendations based on your preferences and needs.")
    
    # Recommendation preferences
    with st.container():
        st.subheader("Tell us about your needs:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usage_context = st.selectbox(
                "Primary Use Case",
                options=[
                    "Work/Professional",
                    "Gaming",
                    "Content Creation",
                    "Daily Use",
                    "Travel",
                    "Students",
                    "Entertainment"
                ],
                help="What will you primarily use this product for?"
            )
            
            budget_preference = st.selectbox(
                "Budget Priority",
                options=[
                    "Budget-conscious (Best value)",
                    "Balanced (Good quality & price)",
                    "Premium (Best quality)"
                ],
                index=1
            )
        
        with col2:
            feature_priorities = st.multiselect(
                "Important Features",
                options=[
                    "Performance",
                    "Battery Life",
                    "Build Quality",
                    "Portability",
                    "Display Quality",
                    "Audio Quality",
                    "Camera Quality",
                    "Storage Capacity",
                    "Connectivity",
                    "Brand Reputation"
                ],
                default=["Performance", "Build Quality"]
            )
            
            size_preference = st.selectbox(
                "Size Preference",
                options=[
                    "Compact/Portable",
                    "Standard",
                    "Large/Desktop Replacement",
                    "No preference"
                ],
                index=3
            )
    
    # Advanced preferences
    with st.expander("🔧 Advanced Preferences"):
        col1, col2 = st.columns(2)
        
        with col1:
            experience_level = st.selectbox(
                "Technical Experience",
                options=["Beginner", "Intermediate", "Advanced", "Expert"],
                index=1
            )
            
            update_frequency = st.selectbox(
                "How often do you upgrade?",
                options=["Every year", "Every 2-3 years", "Every 4-5 years", "Until it breaks"],
                index=1
            )
        
        with col2:
            brand_loyalty = st.selectbox(
                "Brand Preference",
                options=["Strongly prefer certain brands", "Somewhat brand conscious", "Brand agnostic"],
                index=2
            )
            
            review_importance = st.slider(
                "How important are reviews?",
                min_value=1,
                max_value=5,
                value=4,
                help="1 = Not important, 5 = Very important"
            )
    
    # Get recommendations button
    get_recommendations = st.button("🎯 Get My Recommendations", type="primary", use_container_width=True)
    
    return {
        'get_recommendations': get_recommendations,
        'preferences': {
            'usage_context': usage_context,
            'budget_preference': budget_preference,
            'feature_priorities': feature_priorities,
            'size_preference': size_preference,
            'experience_level': experience_level,
            'update_frequency': update_frequency,
            'brand_loyalty': brand_loyalty,
            'review_importance': review_importance
        }
    }

def render_recommendations(recommendations: List[Dict[str, Any]], preferences: Dict[str, Any]):
    """Render recommendation results."""
    if not recommendations:
        st.warning("No recommendations found based on your preferences. Try adjusting your criteria.")
        return
    
    st.success(f"Found {len(recommendations)} personalized recommendations for you!")
    
    # Recommendation summary
    with st.container():
        st.subheader("📊 Recommendation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = sum(r.get('metadata', {}).get('price', 0) for r in recommendations) / len(recommendations)
            st.metric("Avg Price", f"${avg_price:.2f}")
        
        with col2:
            high_match = sum(1 for r in recommendations if r.get('relevance_score', 0) > 0.8)
            st.metric("High Match", f"{high_match}/{len(recommendations)}")
        
        with col3:
            categories = set(r.get('metadata', {}).get('category', '') for r in recommendations)
            st.metric("Categories", len(categories))
        
        with col4:
            avg_rating = sum(r.get('metadata', {}).get('rating', 0) for r in recommendations if r.get('metadata', {}).get('rating', 0) > 0)
            count = sum(1 for r in recommendations if r.get('metadata', {}).get('rating', 0) > 0)
            avg_rating = avg_rating / count if count > 0 else 0
            st.metric("Avg Rating", f"{avg_rating:.1f}/5")
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        render_recommendation_card(rec, i, preferences)

def render_recommendation_card(recommendation: Dict[str, Any], index: int, preferences: Dict[str, Any]):
    """Render a single recommendation card."""
    metadata = recommendation.get('metadata', {})
    product_name = metadata.get('product_name', 'Unknown Product')
    brand = metadata.get('brand', 'Unknown')
    category = metadata.get('category', 'Unknown')
    price = metadata.get('price', 0)
    
    with st.container():
        st.markdown("---")
        
        # Recommendation header with match score
        col1, col2 = st.columns([4, 1])
        
        with col1:
            relevance_score = recommendation.get('relevance_score', 0.0)
            match_level = "🔥 Excellent Match" if relevance_score > 0.8 else "✅ Good Match" if relevance_score > 0.6 else "👍 Decent Match"
            
            st.subheader(f"{index + 1}. {product_name}")
            st.markdown(f"**{brand}** • {category} • {match_level}")
        
        with col2:
            st.metric("Match Score", f"{relevance_score:.1%}")
        
        # Product details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Price", f"${price:.2f}")
            
            # Budget fit indicator
            budget_pref = preferences.get('budget_preference', '')
            if 'Budget-conscious' in budget_pref and price < 500:
                st.success("💰 Budget-friendly")
            elif 'Premium' in budget_pref and price > 1000:
                st.success("💎 Premium quality")
            elif 'Balanced' in budget_pref and 300 <= price <= 1200:
                st.success("⚖️ Balanced choice")
        
        with col2:
            if 'rating' in metadata:
                rating = metadata['rating']
                st.metric("Rating", f"{rating}/5 ⭐")
                
                stars = "⭐" * int(rating) + "☆" * (5 - int(rating))
                st.markdown(f"{stars}")
        
        with col3:
            # Authenticity indicator
            if 'authenticity_score' in metadata:
                auth_score = metadata['authenticity_score']
                if auth_score > 0.7:
                    st.success("✅ Highly Authentic")
                elif auth_score > 0.4:
                    st.warning("⚠️ Moderately Authentic")
                else:
                    st.error("❌ Low Authenticity")
        
        # Why recommended section
        with st.expander(f"🤔 Why we recommend this product"):
            reasons = generate_recommendation_reasons(recommendation, preferences)
            for reason in reasons:
                st.markdown(f"• {reason}")
        
        # Feature highlights
        st.markdown("**Key Highlights:**")
        content = recommendation.get('text', '')
        if content:
            # Extract key points from content
            highlights = extract_highlights(content)
            for highlight in highlights[:3]:  # Top 3 highlights
                st.markdown(f"✨ {highlight}")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(f"📋 Details", key=f"rec_details_{index}"):
                st.session_state.selected_recommendation = recommendation
        
        with col2:
            if st.button(f"🔍 Compare", key=f"rec_compare_{index}"):
                if 'comparison_list' not in st.session_state:
                    st.session_state.comparison_list = []
                st.session_state.comparison_list.append(recommendation)
                st.success("Added to comparison!")
        
        with col3:
            if st.button(f"💾 Save", key=f"rec_save_{index}"):
                if 'saved_products' not in st.session_state:
                    st.session_state.saved_products = []
                st.session_state.saved_products.append(recommendation)
                st.success("Saved!")
        
        with col4:
            if st.button(f"👎 Not Interested", key=f"rec_dismiss_{index}"):
                # This would typically update user preferences
                st.info("Noted! We'll improve future recommendations.")

def generate_recommendation_reasons(recommendation: Dict[str, Any], preferences: Dict[str, Any]) -> List[str]:
    """Generate reasons why this product is recommended."""
    reasons = []
    metadata = recommendation.get('metadata', {})
    
    # Budget alignment
    price = metadata.get('price', 0)
    budget_pref = preferences.get('budget_preference', '')
    
    if 'Budget-conscious' in budget_pref and price < 500:
        reasons.append("💰 **Excellent value** - Fits your budget-conscious preference")
    elif 'Premium' in budget_pref and price > 1000:
        reasons.append("💎 **Premium quality** - Matches your preference for high-end products")
    elif 'Balanced' in budget_pref:
        reasons.append("⚖️ **Good balance** - Offers quality at a reasonable price point")
    
    # Feature alignment
    feature_priorities = preferences.get('feature_priorities', [])
    if 'Performance' in feature_priorities:
        reasons.append("🚀 **High performance** - Matches your performance requirements")
    if 'Build Quality' in feature_priorities:
        reasons.append("🏗️ **Solid build quality** - Known for durability and reliability")
    if 'Battery Life' in feature_priorities:
        reasons.append("🔋 **Excellent battery life** - Perfect for long usage sessions")
    
    # Usage context alignment
    usage_context = preferences.get('usage_context', '')
    if usage_context == 'Work/Professional':
        reasons.append("💼 **Professional grade** - Ideal for business and professional use")
    elif usage_context == 'Gaming':
        reasons.append("🎮 **Gaming optimized** - Great performance for gaming")
    elif usage_context == 'Content Creation':
        reasons.append("🎨 **Creator-friendly** - Excellent for content creation workflows")
    
    # Rating-based reasons
    if 'rating' in metadata:
        rating = metadata['rating']
        if rating >= 4.5:
            reasons.append("⭐ **Highly rated** - Consistently receives excellent reviews")
        elif rating >= 4.0:
            reasons.append("👍 **Well-reviewed** - Positive feedback from users")
    
    # Authenticity-based reasons
    if 'authenticity_score' in metadata:
        auth_score = metadata['authenticity_score']
        if auth_score > 0.7:
            reasons.append("✅ **Trustworthy reviews** - High authenticity score for reviews")
    
    # Default reasons if none generated
    if not reasons:
        reasons = [
            "🎯 **Good match** - Aligns with your specified preferences",
            "📊 **Data-driven choice** - Selected based on comprehensive analysis",
            "🔍 **Relevant content** - Contains information relevant to your search"
        ]
    
    return reasons[:5]  # Return top 5 reasons

def extract_highlights(content: str) -> List[str]:
    """Extract key highlights from product content."""
    # Simple keyword-based extraction
    highlights = []
    
    positive_keywords = [
        'excellent', 'outstanding', 'amazing', 'great', 'perfect', 'best',
        'high-quality', 'premium', 'professional', 'advanced', 'innovative'
    ]
    
    feature_keywords = [
        'battery life', 'performance', 'display', 'camera', 'audio', 'design',
        'build quality', 'fast', 'lightweight', 'durable', 'reliable'
    ]
    
    sentences = content.split('.')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:  # Skip very short sentences
            continue
            
        # Check for positive keywords
        if any(keyword in sentence.lower() for keyword in positive_keywords):
            highlights.append(sentence)
        # Check for feature keywords
        elif any(keyword in sentence.lower() for keyword in feature_keywords):
            highlights.append(sentence)
    
    # If no highlights found, use first few sentences
    if not highlights and sentences:
        highlights = [s.strip() for s in sentences[:3] if s.strip()]
    
    return highlights[:5]  # Return top 5 highlights

def render_recommendation_filters():
    """Render recommendation filters in sidebar."""
    st.sidebar.header("🎯 Recommendation Filters")
    
    # Quick recommendation types
    st.sidebar.subheader("Quick Recommendations")
    
    if st.sidebar.button("💰 Best Value Products", use_container_width=True):
        st.session_state.quick_recommendation = "best_value"
    
    if st.sidebar.button("⭐ Highest Rated", use_container_width=True):
        st.session_state.quick_recommendation = "highest_rated"
    
    if st.sidebar.button("🔥 Trending Now", use_container_width=True):
        st.session_state.quick_recommendation = "trending"
    
    if st.sidebar.button("💎 Premium Picks", use_container_width=True):
        st.session_state.quick_recommendation = "premium"
    
    # Recommendation settings
    st.sidebar.subheader("Settings")
    
    diversity_level = st.sidebar.slider(
        "Result Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values show more diverse products"
    )
    
    personalization_strength = st.sidebar.slider(
        "Personalization",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values prioritize your preferences more"
    )
    
    return {
        'diversity_level': diversity_level,
        'personalization_strength': personalization_strength
    }

def render_saved_recommendations():
    """Render saved recommendations section."""
    if 'saved_products' not in st.session_state:
        st.session_state.saved_products = []
    
    saved_products = st.session_state.saved_products
    
    if saved_products:
        st.sidebar.subheader("💾 Saved Products")
        st.sidebar.markdown(f"You have {len(saved_products)} saved products")
        
        if st.sidebar.button("📋 View All Saved", use_container_width=True):
            st.session_state.show_saved = True
        
        if st.sidebar.button("🗑️ Clear Saved", use_container_width=True):
            st.session_state.saved_products = []
            st.sidebar.success("Cleared saved products!")
    else:
        st.sidebar.markdown("💾 No saved products yet")