import streamlit as st
import pandas as pd
import pickle

# ----- LOAD MODEL -----
MODEL_PATH = "Student_model.pkl"
DATA_PATH = "employee_clean_data_01 (1).csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

# ----- STREAMLIT UI -----
st.title("ML Prediction App")

st.write("This app uses a trained model to make predictions.")

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.write(data)

# Automatically create input widgets for each numeric column
st.subheader("Enter Input Values")

inputs = {}
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    val = st.number_input(col, value=float(data[col].mean()))
    inputs[col] = val

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
