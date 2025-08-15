"""
Streamlit Dashboard for FTSE 100 Quantum Machine Learning Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_collector import FTSE100DataCollector, collect_sample_data, get_market_summary
from feature_engineering import FeatureEngineer, create_sample_features
from quantum_model import VariationalQuantumClassifier, HybridQuantumClassifier
from model_trainer import ModelTrainer
from utils import create_labels, split_data, calculate_metrics, normalize_features

# Page configuration
st.set_page_config(
    page_title="FTSE 100 Quantum ML Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-up {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-down {
        color: #dc3545;
        font-weight: bold;
    }
    .quantum-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load FTSE 100 data with caching."""
    try:
        collector = FTSE100DataCollector()
        data = collector.fetch_data()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def create_features(data):
    """Create features with caching."""
    try:
        features, feature_names = create_sample_features(data)
        return features, feature_names
    except Exception as e:
        st.error(f"Error creating features: {e}")
        return None, None

@st.cache_resource
def train_models(features, labels):
    """Train models with caching."""
    try:
        trainer = ModelTrainer()
        trainer.features = features
        trainer.labels = labels
        trainer.feature_names = list(features.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(
            features.values, labels, train_ratio=0.8, random_state=42
        )
        
        results = trainer.train_models(features.values, labels)
        return trainer, results
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None, None

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 FTSE 100 Quantum ML Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predicting market direction using Quantum Machine Learning")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Data loading section
    st.sidebar.header("📊 Data")
    data_period = st.sidebar.selectbox(
        "Data Period",
        ["1y", "2y", "5y", "10y"],
        index=2
    )
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! Data will be refreshed.")
    
    # Load data
    with st.spinner("Loading FTSE 100 data..."):
        data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please check your internet connection.")
        return
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(data))
    
    with col2:
        st.metric("Date Range", f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    with col3:
        current_price = data['Close'].iloc[-1]
        st.metric("Current Price", f"£{current_price:.2f}")
    
    with col4:
        daily_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
        st.metric("Daily Change", f"{daily_change:.2f}%", delta=f"{daily_change:.2f}%")
    
    # Price chart
    st.subheader("📈 FTSE 100 Price Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='FTSE 100',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="FTSE 100 Historical Prices",
        xaxis_title="Date",
        yaxis_title="Price (£)",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature engineering
    st.subheader("🔧 Feature Engineering")
    
    with st.spinner("Creating features..."):
        features, feature_names = create_features(data)
    
    if features is None:
        st.error("Failed to create features.")
        return
    
    # Display feature information
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Features", len(feature_names))
        st.metric("Feature Categories", "5 (MA, Volatility, Momentum, Technical, Volume)")
    
    with col2:
        st.metric("Data Shape", f"{features.shape[0]} × {features.shape[1]}")
        st.metric("Missing Values", features.isnull().sum().sum())
    
    # Feature correlation heatmap
    st.subheader("🔗 Feature Correlation")
    
    # Select top features for visualization
    top_features = st.selectbox(
        "Number of features to show",
        [10, 20, 30, 50],
        index=0
    )
    
    if len(feature_names) > top_features:
        # Select features with highest variance
        variances = features.var().sort_values(ascending=False)
        selected_features = variances.head(top_features).index.tolist()
        corr_data = features[selected_features].corr()
    else:
        corr_data = features.corr()
    
    fig = px.imshow(
        corr_data,
        title=f"Feature Correlation Matrix (Top {len(corr_data.columns)})",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model training section
    st.subheader("🤖 Model Training")
    
    # Create labels
    threshold = st.slider("Prediction Threshold", -0.02, 0.02, 0.0, 0.001, 
                         help="Minimum price change to consider as 'up' movement")
    
    labels = create_labels(data['Close'], threshold=threshold)
    
    # Align features and labels
    min_length = min(len(features), len(labels))
    features_aligned = features.iloc[:min_length]
    labels_aligned = labels[:min_length]
    
    st.info(f"📊 Dataset: {len(features_aligned)} samples, {len(feature_names)} features")
    st.info(f"🎯 Labels: {np.sum(labels_aligned)} UP, {len(labels_aligned) - np.sum(labels_aligned)} DOWN")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        train_vqc = st.checkbox("Train Variational Quantum Classifier", value=True)
        train_hybrid = st.checkbox("Train Hybrid Quantum-Classical", value=True)
    
    with col2:
        train_classical = st.checkbox("Train Classical Models", value=True)
        max_iter = st.slider("Max Training Iterations", 10, 100, 50)
    
    # Train models
    if st.button("🚀 Train Models"):
        with st.spinner("Training models..."):
            trainer, results = train_models(features_aligned, labels_aligned)
        
        if trainer is None:
            st.error("Model training failed.")
            return
        
        # Display results
        st.subheader("📊 Model Performance")
        
        # Create comparison table
        comparison_data = []
        for model_name, result in results.items():
            if result is not None:
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{metrics['accuracy']:.3f}",
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1_score']:.3f}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best model
            best_model = max(results.items(), 
                           key=lambda x: x[1]['metrics']['accuracy'] if x[1] else 0)
            
            st.success(f"🏆 Best Model: {best_model[0].replace('_', ' ').title()} "
                      f"(Accuracy: {best_model[1]['metrics']['accuracy']:.3f})")
        
        # Prediction section
        st.subheader("🔮 Make Prediction")
        
        if st.button("🎯 Predict Next Day"):
            with st.spinner("Making prediction..."):
                try:
                    # Get latest features
                    latest_features = features_aligned.iloc[-1:].values
                    
                    # Use best model for prediction
                    best_model_name = best_model[0]
                    best_model_instance = best_model[1]['model']
                    
                    prediction = best_model_instance.predict(latest_features)[0]
                    probability = best_model_instance.predict_proba(latest_features)[0]
                    
                    # Display prediction
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction == 1:
                            st.markdown('<p class="prediction-up">📈 PREDICTION: UP</p>', unsafe_allow_html=True)
                        else:
                            st.markdown('<p class="prediction-down">📉 PREDICTION: DOWN</p>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{max(probability):.1%}")
                    
                    with col3:
                        st.metric("Current Price", f"£{current_price:.2f}")
                    
                    # Probability breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Probability UP", f"{probability[1]:.1%}")
                    
                    with col2:
                        st.metric("Probability DOWN", f"{probability[0]:.1%}")
                    
                    # Model info
                    st.markdown('<div class="quantum-info">', unsafe_allow_html=True)
                    st.markdown(f"**Model Used:** {best_model_name.replace('_', ' ').title()}")
                    st.markdown(f"**Prediction Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("**Disclaimer:** This is for educational purposes only. Do not use for investment decisions.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    
    # Market summary
    st.subheader("📰 Market Summary")
    
    try:
        summary = get_market_summary()
        
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"£{summary.get('current_price', 'N/A')}")
            
            with col2:
                daily_change_pct = summary.get('daily_change_pct', 0)
                st.metric("Daily Change", f"{daily_change_pct:.2f}%", delta=f"{daily_change_pct:.2f}%")
            
            with col3:
                st.metric("52-Week High", f"£{summary.get('high_52w', 'N/A')}")
            
            with col4:
                st.metric("52-Week Low", f"£{summary.get('low_52w', 'N/A')}")
    except:
        st.warning("Could not fetch live market summary.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Quantum Machine Learning for Financial Prediction</p>
        <p>⚠️ Educational purposes only - Not financial advice</p>
        <p>Built with Qiskit, Streamlit, and Python</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
