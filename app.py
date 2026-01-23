import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")

input_data = {}

# Split features into numeric & binary (one-hot)
binary_cols = [c for c in features if "_" in c]
num_cols = [c for c in features if c not in binary_cols]

st.subheader("Numeric Inputs")
for c in num_cols:
    input_data[c] = st.number_input(c, value=0.0)

st.subheader("Categorical Flags (Select if true)")
for c in binary_cols:
    input_data[c] = 1 if st.checkbox(c) else 0

# Create DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Churn Probability: {prob:.2f}")

    # Business threshold = 0.4
    if prob >= 0.4:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")
