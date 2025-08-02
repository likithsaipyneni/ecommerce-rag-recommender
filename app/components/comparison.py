"""
Product comparison components for the Streamlit app.
"""
import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

def render_comparison_interface() -> Dict[str, Any]:
    """Render the product comparison interface."""
    st.header("⚖️ Product Comparison")
    st.markdown("Compare products side-by-side to make informed decisions.")
    
    # Check if there are products to compare
    if 'comparison_list' not in st.session_state:
        st.session_state.comparison_list = []
    
    comparison_list = st.session_state.comparison_list
    
    if not comparison_list:
        st.info("🔍 **No products selected for comparison yet.**\n\nAdd products to comparison from search results or recommendations.")
        return {'has_comparisons': False}
    
    # Comparison controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Products to Compare", len(comparison_list))
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.comparison_list = []
            st.experimental_rerun()
    
    with col3:
        comparison_view = st.selectbox(
            "View Type",
            options=["Side-by-Side", "Detailed Analysis", "Specifications", "Reviews"],
            index=0
        )
    
    return {
        'has_comparisons': True,
        'comparison_list': comparison_list,
        'comparison_view': comparison_view
    }

def render_product_comparison(products: List[Dict[str, Any]], view_type: str = "Side-by-Side"):
    """Render the actual product comparison."""
    if not products:
        return
    
    if view_type == "Side-by-Side":
        render_side_by_side_comparison(products)
    elif view_type == "Detailed Analysis":
        render_detailed_analysis(products)
    elif view_type == "Specifications":
        render_specifications_comparison(products)
    elif view_type == "Reviews":
        render_reviews_comparison(products)

def render_side_by_side_comparison(products: List[Dict[str, Any]]):
    """Render side-by-side product comparison."""
    st.subheader("📊 Side-by-Side Comparison")
    
    # Create columns for each product
    cols = st.columns(len(products))
    
    for i, (col, product) in enumerate(zip(cols, products)):
        metadata = product.get('metadata', {})
        
        with col:
            # Product header
            st.markdown(f"### Product {i+1}")
            st.markdown(f"**{metadata.get('product_name', 'Unknown')}**")
            st.markdown(f"*{metadata.get('brand', 'Unknown')} • {metadata.get('category', 'Unknown')}*")
            
            # Key metrics
            price = metadata.get('price', 0)
            st.metric("Price", f"${price:.2f}")
            
            if 'rating' in metadata:
                rating = metadata['rating']
                st.metric("Rating", f"{rating}/5 ⭐")
            
            similarity = product.get('similarity', 0.0)
            st.metric("Match Score", f"{similarity:.1%}")
            
            # Quick specs
            st.markdown("**Quick Specs:**")
            specs = get_mock_specs(metadata)
            for spec, value in specs.items():
                st.markdown(f"• **{spec}:** {value}")
            
            # Pros and cons
            pros_cons = get_mock_pros_cons(metadata)
            
            with st.expander("👍 Pros"):
                for pro in pros_cons['pros']:
                    st.markdown(f"✅ {pro}")
            
            with st.expander("👎 Cons"):
                for con in pros_cons['cons']:
                    st.markdown(f"❌ {con}")
            
            # Remove button
            if st.button(f"Remove", key=f"remove_{i}", use_container_width=True):
                st.session_state.comparison_list.remove(product)
                st.experimental_rerun()

def render_detailed_analysis(products: List[Dict[str, Any]]):
    """Render detailed comparison analysis."""
    st.subheader("🔍 Detailed Analysis")
    
    # Create comparison matrix
    comparison_data = create_comparison_matrix(products)
    
    # Display as table
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Analysis insights
    st.subheader("📈 Analysis Insights")
    
    insights = generate_comparison_insights(products)
    for insight in insights:
        st.markdown(f"💡 {insight}")
    
    # Winner in each category
    st.subheader("🏆 Category Winners")
    
    winners = determine_category_winners(products)
    
    col1, col2 = st.columns(2)
    
    with col1:
        for category, winner in winners.items():
            if category in ['price', 'rating', 'performance']:
                st.markdown(f"**{category.title()}:** {winner}")
    
    with col2:
        for category, winner in winners.items():
            if category in ['value', 'features', 'reliability']:
                st.markdown(f"**{category.title()}:** {winner}")

def render_specifications_comparison(products: List[Dict[str, Any]]):
    """Render detailed specifications comparison."""
    st.subheader("🔧 Technical Specifications")
    
    # Create specifications table
    spec_data = []
    
    common_specs = ['Processor', 'Memory', 'Storage', 'Display', 'Battery', 'Weight']
    
    for spec in common_specs:
        row = {'Specification': spec}
        
        for i, product in enumerate(products):
            metadata = product.get('metadata', {})
            product_name = metadata.get('product_name', f'Product {i+1}')
            
            # Get mock spec value
            specs = get_mock_specs(metadata)
            value = specs.get(spec, 'Not specified')
            row[product_name] = value
        
        spec_data.append(row)
    
    # Display as table
    df = pd.DataFrame(spec_data)
    st.dataframe(df, use_container_width=True)
    
    # Specification highlights
    st.subheader("⭐ Specification Highlights")
    
    for i, product in enumerate(products):
        metadata = product.get('metadata', {})
        product_name = metadata.get('product_name', f'Product {i+1}')
        
        with st.expander(f"🔍 {product_name} Highlights"):
            highlights = get_specification_highlights(metadata)
            for highlight in highlights:
                st.markdown(f"• {highlight}")

def render_reviews_comparison(products: List[Dict[str, Any]]):
    """Render reviews comparison."""
    st.subheader("💬 Reviews Analysis")
    
    # Reviews summary table
    review_data = []
    
    for product in products:
        metadata = product.get('metadata', {})
        product_name = metadata.get('product_name', 'Unknown')
        
        # Mock review data
        review_summary = {
            'Product': product_name,
            'Avg Rating': metadata.get('rating', 0),
            'Total Reviews': np.random.randint(50, 500),
            'Positive %': np.random.randint(70, 95),
            'Negative %': np.random.randint(5, 30),
            'Authenticity': metadata.get('authenticity_score', 0.7)
        }
        
        review_data.append(review_summary)
    
    df = pd.DataFrame(review_data)
    st.dataframe(df, use_container_width=True)
    
    # Sentiment comparison
    st.subheader("😊 Sentiment Analysis")
    
    for i, product in enumerate(products):
        metadata = product.get('metadata', {})
        product_name = metadata.get('product_name', f'Product {i+1}')
        
        with st.expander(f"📊 {product_name} Sentiment Breakdown"):
            
            # Mock sentiment data
            sentiments = {
                'Very Positive': np.random.randint(20, 40),
                'Positive': np.random.randint(30, 50),
                'Neutral': np.random.randint(10, 20),
                'Negative': np.random.randint(5, 15),
                'Very Negative': np.random.randint(0, 5)
            }
            
            # Display as columns
            cols = st.columns(len(sentiments))
            for col, (sentiment, percentage) in zip(cols, sentiments.items()):
                with col:
                    st.metric(sentiment, f"{percentage}%")
            
            # Common themes
            st.markdown("**Common Themes:**")
            themes = get_mock_review_themes(metadata)
            for theme in themes:
                st.markdown(f"• {theme}")

def create_comparison_matrix(products: List[Dict[str, Any]]) -> Dict[str, List]:
    """Create comparison matrix data."""
    matrix = {
        'Attribute': ['Name', 'Brand', 'Category', 'Price', 'Rating', 'Match Score', 'Authenticity']
    }
    
    for i, product in enumerate(products):
        metadata = product.get('metadata', {})
        
        column_name = f'Product {i+1}'
        matrix[column_name] = [
            metadata.get('product_name', 'Unknown'),
            metadata.get('brand', 'Unknown'),
            metadata.get('category', 'Unknown'),
            f"${metadata.get('price', 0):.2f}",
            f"{metadata.get('rating', 0)}/5",
            f"{product.get('similarity', 0):.1%}",
            f"{metadata.get('authenticity_score', 0.7):.2f}"
        ]
    
    return matrix

def generate_comparison_insights(products: List[Dict[str, Any]]) -> List[str]:
    """Generate insights from product comparison."""
    insights = []
    
    if len(products) < 2:
        return insights
    
    # Price comparison
    prices = [p.get('metadata', {}).get('price', 0) for p in products]
    if prices:
        min_price = min(prices)
        max_price = max(prices)
        price_diff = max_price - min_price
        
        if price_diff > 500:
            insights.append(f"Price range varies significantly by ${price_diff:.2f} - consider your budget carefully")
        
        cheapest_idx = prices.index(min_price)
        most_expensive_idx = prices.index(max_price)
        
        cheapest_name = products[cheapest_idx].get('metadata', {}).get('product_name', 'Unknown')
        expensive_name = products[most_expensive_idx].get('metadata', {}).get('product_name', 'Unknown')
        
        insights.append(f"Most affordable: {cheapest_name} (${min_price:.2f})")
        insights.append(f"Premium option: {expensive_name} (${max_price:.2f})")
    
    # Rating comparison
    ratings = [p.get('metadata', {}).get('rating', 0) for p in products if p.get('metadata', {}).get('rating', 0) > 0]
    if ratings:
        best_rated_idx = None
        best_rating = 0
        for i, product in enumerate(products):
            rating = product.get('metadata', {}).get('rating', 0)
            if rating > best_rating:
                best_rating = rating
                best_rated_idx = i
        
        if best_rated_idx is not None:
            best_name = products[best_rated_idx].get('metadata', {}).get('product_name', 'Unknown')
            insights.append(f"Highest rated: {best_name} ({best_rating}/5 stars)")
    
    # Value proposition
    value_scores = []
    for product in products:
        price = product.get('metadata', {}).get('price', 0)
        rating = product.get('metadata', {}).get('rating', 0)
        
        if price > 0 and rating > 0:
            value_score = rating / (price / 1000)  # Rating per $1000
            value_scores.append((value_score, product))
    
    if value_scores:
        best_value = max(value_scores, key=lambda x: x[0])
        best_value_name = best_value[1].get('metadata', {}).get('product_name', 'Unknown')
        insights.append(f"Best value for money: {best_value_name}")
    
    return insights

def determine_category_winners(products: List[Dict[str, Any]]) -> Dict[str, str]:
    """Determine winner in each category."""
    winners = {}
    
    # Price winner (lowest price)
    prices = [(p.get('metadata', {}).get('price', float('inf')), p) for p in products]
    if prices:
        cheapest = min(prices, key=lambda x: x[0])
        winners['price'] = cheapest[1].get('metadata', {}).get('product_name', 'Unknown')
    
    # Rating winner (highest rating)
    ratings = [(p.get('metadata', {}).get('rating', 0), p) for p in products]
    if ratings:
        highest_rated = max(ratings, key=lambda x: x[0])
        winners['rating'] = highest_rated[1].get('metadata', {}).get('product_name', 'Unknown')
    
    # Performance winner (mock calculation)
    performance_scores = []
    for product in products:
        metadata = product.get('metadata', {})
        
        # Mock performance calculation
        score = 0.5
        if 'i7' in str(metadata.get('specifications', {})).lower():
            score += 0.2
        if '16gb' in str(metadata.get('specifications', {})).lower():
            score += 0.2
        
        performance_scores.append((score, product))
    
    if performance_scores:
        best_performance = max(performance_scores, key=lambda x: x[0])
        winners['performance'] = best_performance[1].get('metadata', {}).get('product_name', 'Unknown')
    
    return winners

def get_mock_specs(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Get mock specifications for a product."""
    category = metadata.get('category', '').lower()
    
    if 'laptop' in category:
        return {
            'Processor': 'Intel Core i7-12700H',
            'Memory': '16GB DDR4',
            'Storage': '512GB SSD',
            'Display': '15.6" Full HD',
            'Battery': 'Up to 8 hours',
            'Weight': '4.2 lbs'
        }
    elif 'phone' in category or 'smartphone' in category:
        return {
            'Processor': 'Snapdragon 8 Gen 3',
            'Memory': '12GB RAM',
            'Storage': '256GB',
            'Display': '6.7" AMOLED',
            'Battery': '5000mAh',
            'Weight': '7.8 oz'
        }
    elif 'headphone' in category:
        return {
            'Driver': '40mm dynamic',
            'Frequency': '20Hz-20kHz',
            'Battery': '30 hours ANC',
            'Weight': '8.8 oz',
            'Connectivity': 'Bluetooth 5.2',
            'Features': 'Active Noise Canceling'
        }
    else:
        return {
            'Type': 'Consumer Electronics',
            'Model': 'Latest Generation',
            'Warranty': '1 Year',
            'Color': 'Multiple Options',
            'Connectivity': 'Modern Standards',
            'Features': 'Advanced Features'
        }

def get_mock_pros_cons(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get mock pros and cons for a product."""
    category = metadata.get('category', '').lower()
    brand = metadata.get('brand', '').lower()
    
    pros = []
    cons = []
    
    # Brand-based pros/cons
    if 'apple' in brand:
        pros.extend(['Premium build quality', 'Excellent ecosystem integration', 'Great customer support'])
        cons.extend(['Higher price point', 'Limited customization'])
    elif 'samsung' in brand:
        pros.extend(['Feature-rich', 'Good value for money', 'Regular updates'])
        cons.extend(['Bloatware concerns', 'Complex interface'])
    elif 'sony' in brand:
        pros.extend(['Superior audio quality', 'Reliable performance', 'Innovative features'])
        cons.extend(['Premium pricing', 'Complex controls'])
    
    # Category-based pros/cons
    if 'laptop' in category:
        pros.extend(['Good performance', 'Portable design', 'Long battery life'])
        cons.extend(['Limited upgrade options', 'Can get warm under load'])
    elif 'phone' in category:
        pros.extend(['Great camera', 'Fast performance', 'Good display'])
        cons.extend(['Battery degradation over time', 'Screen fragility'])
    elif 'headphone' in category:
        pros.extend(['Excellent sound quality', 'Comfortable fit', 'Good noise cancellation'])
        cons.extend(['Bulky design', 'Price premium for features'])
    
    # Ensure we have at least some pros and cons
    if not pros:
        pros = ['Good build quality', 'Reliable performance', 'User-friendly design']
    if not cons:
        cons = ['Price could be lower', 'Some features may be complex', 'Competition exists']
    
    return {
        'pros': pros[:4],  # Top 4 pros
        'cons': cons[:3]   # Top 3 cons
    }

def get_specification_highlights(metadata: Dict[str, Any]) -> List[str]:
    """Get specification highlights for a product."""
    highlights = []
    
    category = metadata.get('category', '').lower()
    price = metadata.get('price', 0)
    
    if 'laptop' in category:
        highlights.extend([
            '🚀 High-performance processor for demanding tasks',
            '💾 Ample RAM for smooth multitasking',
            '⚡ Fast SSD storage for quick boot times',
            '🖥️ Crisp display with good color accuracy'
        ])
    elif 'phone' in category:
        highlights.extend([
            '📱 Latest generation mobile processor',
            '📸 Advanced camera system with AI features',
            '🔋 All-day battery life with fast charging',
            '📺 Vibrant AMOLED display technology'
        ])
    elif 'headphone' in category:
        highlights.extend([
            '🎵 Premium drivers for exceptional sound',
            '🔇 Industry-leading noise cancellation',
            '🔋 Extended battery life for long listening',
            '📱 Seamless device connectivity'
        ])
    
    # Price-based highlights
    if price < 500:
        highlights.append('💰 Excellent value for the feature set')
    elif price > 1500:
        highlights.append('💎 Premium materials and construction')
    
    return highlights[:5]

def get_mock_review_themes(metadata: Dict[str, Any]) -> List[str]:
    """Get mock review themes for a product."""
    category = metadata.get('category', '').lower()
    
    if 'laptop' in category:
        return [
            'Users praise the fast performance and build quality',
            'Battery life consistently meets expectations',
            'Some concerns about fan noise under heavy load',
            'Display quality receives positive feedback'
        ]
    elif 'phone' in category:
        return [
            'Camera quality is a standout feature',
            'Users appreciate the smooth performance',
            'Battery optimization gets mixed reviews',
            'Build quality and design are well-received'
        ]
    elif 'headphone' in category:
        return [
            'Sound quality exceeds expectations',
            'Comfort for long listening sessions',
            'Noise cancellation effectiveness varies by environment',
            'Value for money is questioned by some users'
        ]
    else:
        return [
            'Overall satisfaction with product quality',
            'Features meet advertised specifications',
            'Customer service experience varies',
            'Delivery and packaging generally positive'
        ]