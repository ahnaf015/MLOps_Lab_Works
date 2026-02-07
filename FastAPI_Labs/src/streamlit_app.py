# /src/streamlit_app.py

import streamlit as st
import requests
import json
import plotly.graph_objects as go
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #8B0000;
        color: white;
        font-size: 16px;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #A52A2A;
    }
    .example-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 2px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'wine_data' not in st.session_state:
    st.session_state.wine_data = {
        'fixed_acidity': 7.4,
        'volatile_acidity': 0.7,
        'citric_acid': 0.0,
        'residual_sugar': 1.9,
        'chlorides': 0.076,
        'free_sulfur_dioxide': 11.0,
        'total_sulfur_dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }

# Example wine profiles
EXAMPLES = {
    "Poor Quality Wine": {
        'fixed_acidity': 10.2,
        'volatile_acidity': 1.27,
        'citric_acid': 0.0,
        'residual_sugar': 2.0,
        'chlorides': 0.12,
        'free_sulfur_dioxide': 15.0,
        'total_sulfur_dioxide': 30.0,
        'density': 1.001,
        'pH': 3.2,
        'sulphates': 0.4,
        'alcohol': 8.5
    },
    "Average Quality Wine": {
        'fixed_acidity': 7.4,
        'volatile_acidity': 0.7,
        'citric_acid': 0.0,
        'residual_sugar': 1.9,
        'chlorides': 0.076,
        'free_sulfur_dioxide': 11.0,
        'total_sulfur_dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    },
    "Good Quality Wine": {
        'fixed_acidity': 8.5,
        'volatile_acidity': 0.28,
        'citric_acid': 0.56,
        'residual_sugar': 1.8,
        'chlorides': 0.092,
        'free_sulfur_dioxide': 35.0,
        'total_sulfur_dioxide': 103.0,
        'density': 0.9969,
        'pH': 3.3,
        'sulphates': 0.75,
        'alcohol': 10.5
    }
}

def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get model metadata from API"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_wine_quality(features):
    """Make prediction via API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def create_quality_gauge(quality_score):
    """Create a gauge chart for quality score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=quality_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Wine Quality Score", 'font': {'size': 24}},
        delta={'reference': 5.5, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkred"},
            'bar': {'color': "darkred"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 4], 'color': '#ffcccc'},
                {'range': [4, 6], 'color': '#ffe6cc'},
                {'range': [6, 8], 'color': '#ccffcc'},
                {'range': [8, 10], 'color': '#99ff99'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': quality_score
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def load_example(example_name):
    """Load example data into session state"""
    st.session_state.wine_data = EXAMPLES[example_name].copy()
    st.session_state.example_loaded = example_name

def main():
    # Header
    st.title("🍷 Wine Quality Prediction System")
    st.markdown("### Predict wine quality based on physicochemical properties")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        api_healthy = check_api_health()
        
        if api_healthy:
            st.success("✅ API Connected")
            
            model_info = get_model_info()
            
            if model_info:
                st.markdown("---")
                st.subheader("🤖 Model Information")
                st.write(f"**Model:** {model_info['model_name']}")
                st.write(f"**Version:** {model_info['version']}")
                
                trained_at = datetime.fromisoformat(model_info['trained_at'])
                st.write(f"**Trained:** {trained_at.strftime('%Y-%m-%d %H:%M')}")
                
                st.markdown("---")
                st.subheader("📈 Performance Metrics")
                metrics = model_info['metrics']
                st.metric("R² Score", f"{metrics['r2']:.3f}")
                st.metric("RMSE", f"{metrics['rmse']:.3f}")
                st.metric("MAE", f"{metrics['mae']:.3f}")
                
                st.markdown("---")
                st.subheader("⚙️ Hyperparameters")
                hyperparams = model_info['hyperparameters']
                st.write(f"**Trees:** {hyperparams['n_estimators']}")
                st.write(f"**Max Depth:** {hyperparams['max_depth']}")
        else:
            st.error("❌ API Disconnected")
            st.warning("Make sure FastAPI is running on port 8000")
            st.code("uvicorn src.main:app --reload", language="bash")
    
    if not api_healthy:
        st.error("⚠️ Cannot connect to API. Please start the FastAPI backend first.")
        st.stop()
    
    # ONLY 2 TABS NOW
    tab1, tab2 = st.tabs(["🔬 Wine Quality Predictor", "ℹ️ About"])
    
    with tab1:
        # Example Selection Section - AT THE TOP
        st.markdown('<div class="example-section">', unsafe_allow_html=True)
        st.subheader("🎯 Quick Start: Load an Example")
        st.markdown("**New to wine chemistry?** Load a sample wine profile to see how it works!")
        
        ex_col1, ex_col2, ex_col3, ex_col4 = st.columns([3, 3, 3, 2])
        
        with ex_col1:
            if st.button("🔴 Poor Quality", use_container_width=True, key="poor_btn"):
                load_example("Poor Quality Wine")
                st.rerun()
        
        with ex_col2:
            if st.button("🟡 Average Quality", use_container_width=True, key="avg_btn"):
                load_example("Average Quality Wine")
                st.rerun()
        
        with ex_col3:
            if st.button("🟢 Good Quality", use_container_width=True, key="good_btn"):
                load_example("Good Quality Wine")
                st.rerun()
        
        with ex_col4:
            if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
                st.session_state.wine_data = {
                    'fixed_acidity': 7.4,
                    'volatile_acidity': 0.7,
                    'citric_acid': 0.0,
                    'residual_sugar': 1.9,
                    'chlorides': 0.076,
                    'free_sulfur_dioxide': 11.0,
                    'total_sulfur_dioxide': 34.0,
                    'density': 0.9978,
                    'pH': 3.51,
                    'sulphates': 0.56,
                    'alcohol': 9.4
                }
                if 'example_loaded' in st.session_state:
                    del st.session_state.example_loaded
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                st.rerun()
        
        # Show loaded example indicator
        if 'example_loaded' in st.session_state:
            st.success(f"✅ **Loaded: {st.session_state.example_loaded}** - Adjust values below or predict as-is!")
        else:
            st.info("💡 **Tip:** Load an example above or manually adjust the sliders below")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input Form
        st.subheader("🔬 Wine Properties")
        st.markdown("Adjust the chemical properties of your wine:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Acidity Properties")
            fixed_acidity = st.slider(
                "Fixed Acidity (g/dm³)",
                min_value=4.0, max_value=16.0, 
                value=st.session_state.wine_data['fixed_acidity'], 
                step=0.1,
                help="Tartaric acid concentration - affects taste sharpness"
            )
            
            volatile_acidity = st.slider(
                "Volatile Acidity (g/dm³)",
                min_value=0.1, max_value=1.6, 
                value=st.session_state.wine_data['volatile_acidity'], 
                step=0.01,
                help="Acetic acid (vinegar) - high values = spoiled wine"
            )
            
            citric_acid = st.slider(
                "Citric Acid (g/dm³)",
                min_value=0.0, max_value=1.0, 
                value=st.session_state.wine_data['citric_acid'], 
                step=0.01,
                help="Adds freshness and flavor"
            )
            
            pH = st.slider(
                "pH Level",
                min_value=2.7, max_value=4.0, 
                value=st.session_state.wine_data['pH'], 
                step=0.01,
                help="Acidity level: 2.7-3.5 (very acidic), 3.5-4.0 (less acidic)"
            )
            
            st.markdown("#### Sugar & Salt")
            residual_sugar = st.slider(
                "Residual Sugar (g/dm³)",
                min_value=0.9, max_value=15.0, 
                value=st.session_state.wine_data['residual_sugar'], 
                step=0.1,
                help="Sugar remaining after fermentation - affects sweetness"
            )
            
            chlorides = st.slider(
                "Chlorides (g/dm³)",
                min_value=0.01, max_value=0.6, 
                value=st.session_state.wine_data['chlorides'], 
                step=0.001,
                help="Salt content - high values = salty taste"
            )
        
        with col2:
            st.markdown("#### Sulfur Dioxide")
            free_sulfur_dioxide = st.slider(
                "Free SO₂ (mg/dm³)",
                min_value=1.0, max_value=70.0, 
                value=st.session_state.wine_data['free_sulfur_dioxide'], 
                step=1.0,
                help="Prevents oxidation and microbial growth"
            )
            
            total_sulfur_dioxide = st.slider(
                "Total SO₂ (mg/dm³)",
                min_value=6.0, max_value=290.0, 
                value=st.session_state.wine_data['total_sulfur_dioxide'], 
                step=1.0,
                help="Free + bound SO₂ - preservative"
            )
            
            st.markdown("#### Other Properties")
            density = st.slider(
                "Density (g/cm³)",
                min_value=0.990, max_value=1.004, 
                value=st.session_state.wine_data['density'], 
                step=0.0001,
                help="Depends on alcohol and sugar content"
            )
            
            sulphates = st.slider(
                "Sulphates (g/dm³)",
                min_value=0.3, max_value=2.0, 
                value=st.session_state.wine_data['sulphates'], 
                step=0.01,
                help="Wine additive - antimicrobial and antioxidant"
            )
            
            alcohol = st.slider(
                "Alcohol (%)",
                min_value=8.0, max_value=15.0, 
                value=st.session_state.wine_data['alcohol'], 
                step=0.1,
                help="Alcohol by volume - higher = stronger wine"
            )
        
        # Update session state with current slider values
        st.session_state.wine_data = {
            'fixed_acidity': fixed_acidity,
            'volatile_acidity': volatile_acidity,
            'citric_acid': citric_acid,
            'residual_sugar': residual_sugar,
            'chlorides': chlorides,
            'free_sulfur_dioxide': free_sulfur_dioxide,
            'total_sulfur_dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        
        # Predict button
        st.markdown("---")
        
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        
        with predict_col2:
            if st.button("🔮 Predict Wine Quality", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing wine properties..."):
                    result = predict_wine_quality(st.session_state.wine_data)
                
                if result:
                    st.session_state.prediction_result = result
                    st.rerun()
        
        # Display results
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                fig = create_quality_gauge(result['quality_score'])
                st.plotly_chart(fig, use_container_width=True)
            
            with res_col2:
                st.markdown("### Results")
                
                score = result['quality_score']
                category = result['quality_category']
                
                if score < 5:
                    color = "🔴"
                elif score < 6:
                    color = "🟡"
                elif score < 7:
                    color = "🟢"
                else:
                    color = "💚"
                
                st.markdown(f"### {color} Quality Score: **{score:.2f}/10**")
                st.markdown(f"### Category: **{category}**")
                st.markdown(f"**Confidence:** {result['confidence']}")
                
                st.markdown("---")
                st.markdown("#### 📝 Interpretation")
                
                if category == "Poor":
                    st.warning("⚠️ This wine has below-average quality. Consider adjusting acidity, sulfur levels, or alcohol content.")
                elif category == "Average":
                    st.info("ℹ️ This wine has average quality. Good for everyday consumption.")
                elif category == "Good":
                    st.success("✅ This wine has good quality. Suitable for special occasions.")
                else:
                    st.success("🌟 This wine has excellent quality! Premium grade.")
    
    with tab2:
        st.subheader("About This Application")
        
        st.markdown("""
        ### 🍷 Wine Quality Prediction System
        
        This application predicts wine quality based on physicochemical properties using machine learning.
        
        #### 📊 How It Works
        
        1. **Load Example** (Optional): Click a quality button to auto-fill the form with sample data
        2. **Adjust Values**: Fine-tune the chemical properties using the sliders
        3. **Predict**: Click the prediction button to get quality score
        4. **Review Results**: See visual gauge and detailed interpretation
        
        #### 🤖 Model Details
        
        - **Algorithm:** Random Forest Regressor (Ensemble of 100 decision trees)
        - **Features:** 11 physicochemical properties of red wine
        - **Dataset:** UCI Red Wine Quality Dataset (1,599 samples)
        - **Performance:** 
          - R² Score: 0.51 (explains 51% of quality variance)
          - RMSE: 0.56 (predictions within ±0.5 quality points)
          - MAE: 0.44 (average error of 0.44 points)
        
        #### 🏗️ Tech Stack
        
        - **Frontend:** Streamlit + Plotly (Interactive UI & Visualizations)
        - **Backend:** FastAPI (REST API server)
        - **ML Model:** Scikit-learn (Training & Inference)
        - **Data Processing:** Pandas, NumPy
        
        #### 📚 Wine Quality Scale
        
        - **0-4:** 🔴 Poor quality (avoid)
        - **5-6:** 🟡 Average quality (everyday wine)
        - **6-7:** 🟢 Good quality (special occasions)
        - **8-10:** 💚 Excellent quality (premium/rare)
        
        #### 🔬 Key Wine Properties
        
        **Acidity:**
        - High fixed acidity → Sharp, tart taste
        - High volatile acidity → Vinegar smell (spoilage)
        - Citric acid → Fresh, citrus notes
        
        **Sulfur Dioxide (SO₂):**
        - Preservative to prevent oxidation
        - Too much → Chemical taste
        - Too little → Wine spoils quickly
        
        **Alcohol:**
        - Higher alcohol → Fuller body, warming sensation
        - Lower alcohol → Lighter, more refreshing
        
        #### 🔗 Links
        
        - [API Documentation](http://localhost:8000/docs) - Interactive API testing
        - [Model Metadata](http://localhost:8000/model/info) - Performance metrics
        
        """)

if __name__ == "__main__":
    main()