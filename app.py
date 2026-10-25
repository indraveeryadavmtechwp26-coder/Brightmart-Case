import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set up app title and description
st.title("Sales Prediction App")
st.write("""
This app predicts **Sales** based on advertising budgets allocated to **TV**, **Radio**, and **Newspaper** channels 
using a trained Linear Regression model.
""")

# Load the saved model
try:
    model = joblib.load("LR.sav")
except FileNotFoundError:
    st.error("Model file 'LR.sav' not found. Please ensure it is present in the same directory as this script.")
    st.stop()

# Create input fields for user to input budget values
st.sidebar.header("Input Advertising Budgets")
tv_budget = st.sidebar.number_input("TV Budget ($ thousands)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
radio_budget = st.sidebar.number_input("Radio Budget ($ thousands)", min_value=0.0, max_value=1000.0, value=20.0, step=1.0)
newspaper_budget = st.sidebar.number_input("Newspaper Budget ($ thousands)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)

# Prepare input data matching feature names of the training set
input_data = pd.DataFrame({
    'TV': [tv_budget],
    'Radio': [radio_budget],
    'Newspaper': [newspaper_budget]
})

st.subheader("Your Input Parameters:")
st.write(input_data)

# Make prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales: **{prediction[0]:.2f}** units")
