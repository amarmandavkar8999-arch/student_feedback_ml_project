import streamlit as st
import pickle
import numpy as np

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    with open("Student_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("Student Performance Prediction App by Amar M")

st.write("Enter the details below to get prediction")

# Change names based on your dataset
col1, col2 = st.columns(2)

math_score = col1.number_input("Math Score", 0, 100, 50)
reading_score = col2.number_input("Reading Score", 0, 100, 50)
writing_score = st.number_input("Writing Score", 0, 100, 50)

# Missing 4th input added here
extra_feature = st.number_input("Extra Feature (4th Column)", 0, 100, 50)

if st.button("Predict"):
    try:
        # Now sending 4 features instead of 3
        input_data = np.array([[math_score, reading_score, writing_score, extra_feature]])
        prediction = model.predict(input_data)
        st.success(f"Predicted Value: {prediction[0]}")
    except Exception as e:
        st.error(f"Error while predicting: {e}")
