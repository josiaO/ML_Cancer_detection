import streamlit as st
import numpy as np
import joblib

# Load the trained model
clf = joblib.load("model.pkl")

st.title("Breast Cancer Prediction App")
st.subheader("Input the 30 features used for classification")

# Define the 30 feature inputs
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Collect input values from user
input_values = []
for feature in features:
    val = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
    input_values.append(val)

# Predict and display result
if st.button("Predict"):
    input_data = np.array([input_values])
    prediction = clf.predict(input_data)[0]
    result = "Malignant" if prediction == 0 else "Benign"
    st.success(f"The tumor is likely: {result}")
