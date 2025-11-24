# app.py

import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# --- Configuration & Styling (Vibrant & Professional) ---
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the NEW high-contrast, professional theme
st.markdown("""
<style>
    /* Main Content Styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray/Off-white background */
        color: #1a1a1a; /* Dark text for high contrast */
    }
    h1 {
        color: #0057b7; /* Deep Blue Accent for the Title */
        font-weight: 800;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    h3 {
        color: #1a1a1a; /* Dark text for section headers */
    }
    
    /* Button Styling (Yellow/Orange Highlight) */
    .stButton>button {
        background-color: #ffd700; /* Yellow/Orange Accent */
        color: #1a1a1a; /* Dark text on button */
        font-weight: bold;
        border-radius: 8px;
        border: 2px solid #0057b7; /* Deep Blue border */
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #e0bb00; /* Slightly darker hover color */
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Metric/Result Styling */
    [data-testid="stMetricValue"] {
        color: #0057b7; /* Deep Blue for the final prediction value */
        font-size: 2.5rem;
    }
    [data-testid="stMetricLabel"] {
        color: #4a4a4a; /* Dark gray for metric label */
    }

    /* Info Boxes/Sidebar */
    .stAlert {
        background-color: #e0f7fa; /* Very light blue for info boxes */
        border-left: 5px solid #0057b7;
    }
    .stSidebar {
        background-color: #ffffff; /* White background for the sidebar */
        border-right: 1px solid #e0e0e0;
    }

</style>
""", unsafe_allow_html=True)


# --- Model Loading & Caching (The Core) ---
# Use st.cache_resource to load the model only once.
@st.cache_resource
def load_model():
    """Load the serialized machine learning model."""
    try:
        model_path = Path("sal prediction.pkl")
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model. Check scikit-learn version in requirements.txt. Details: {e}")
        st.stop()

model = load_model()

# --- Feature Mapping (Based on Model Snippet) ---
EDUCATION_MAP = {
    "High School": 1,
    "Bachelor's Degree": 2,
    "Master's Degree": 3,
    "PhD": 4
}


# --- Prediction Function ---
def predict_salary(experience, education_level_num, age):
    """Makes a prediction using the loaded model."""
    features = np.array([[experience, education_level_num, age]])
    prediction = model.predict(features)[0]
    return prediction

# --- Main App Interface ---

# Header Section
st.title("💸 AI-Powered Salary Estimator")
st.markdown("---")
st.markdown("## Input your professional profile to estimate your annual compensation.")

# 1. User Input Section
st.markdown("### Profile Details")
with st.container():
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input 1: Years of Experience
        experience = st.slider(
            "💼 **Years of Professional Experience**",
            min_value=0.0, 
            max_value=30.0, 
            value=5.0, 
            step=0.5,
            format="%.1f"
        )
        st.info("Drag the slider to accurately reflect your experience.")

    with col2:
        # Input 2: Education Level 
        education_label = st.selectbox(
            "🎓 **Highest Education Level**",
            options=list(EDUCATION_MAP.keys()),
            index=2 
        )
        education_level_num = EDUCATION_MAP[education_label] 

        # Input 3: Age
        age = st.number_input(
            "🎂 **Age (Years)**",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

st.markdown("---")

# 2. Prediction Trigger
if st.button("🚀 Predict Salary Estimate", key="predict_button"):
    
    # 3. Prediction Logic
    try:
        predicted_salary = predict_salary(experience, education_level_num, age)
        
        # Display the result attractively using st.metric
        st.balloons() 
        
        st.markdown("## 💰 Your Predicted Annual Salary Estimate:")
        st.metric(
            label="Based on your inputs:", 
            value=f"${predicted_salary:,.2f}", 
            delta="Estimate"
        )
        
        st.success("The estimation is complete. This is the projected salary based on the trained model.")

    except Exception as e:
        st.error(f"Prediction Error. Please check your inputs and model compatibility. {e}")

# 4. Sidebar Information
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1593640428585-7b79626b9a2b", width=300) 
    st.title("About This App")
    st.info("This is an end-to-end Machine Learning deployment demo built with **Streamlit**.")
    st.markdown("""
    ---
    ### Model Details
    - **Type**: Linear Regression
    - **Features**: Experience, Education Level, Age
    - **Source**: `sal prediction.pkl`
    """)
