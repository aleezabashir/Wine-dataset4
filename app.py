import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and label encoder
try:
    rf_model = joblib.load("wine_quality_model.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ Model files not found. Run train_model.py first.")
    st.stop()
except ModuleNotFoundError:
    st.error("❌ joblib not installed. Make sure requirements.txt includes joblib.")
    st.stop()

# Feature names
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

st.set_page_config(page_title="🍷 Wine Quality Classifier", layout="wide")
st.markdown("<h1 style='text-align: center; color: darkred;'>🍷 Wine Quality Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict wine quality category based on chemical features</p>", unsafe_allow_html=True)

# Input sliders
st.subheader("🔬 Input Wine Features")
cols = st.columns(3)
inputs = []
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.slider(f"{feature}", 0.0, 20.0, 5.0, 0.01)
        inputs.append(val)

# Predict button
if st.button("🌟 Predict Quality Class 🌟"):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    pred_encoded = rf_model.predict(input_df)
    pred_class = le.inverse_transform(pred_encoded)[0]
    st.success(f"✅ Predicted Wine Quality Class: **{pred_class}**")
    st.subheader("Your Input Measurements")
    st.table(input_df)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit & Random Forest</p>", unsafe_allow_html=True)
