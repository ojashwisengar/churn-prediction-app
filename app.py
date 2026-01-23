import streamlit as st
import pandas as pd
import joblib
from collections import defaultdict

# Load model & features
model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")

input_data = {}

# --------- Identify groups ----------
groups = defaultdict(list)
numeric_cols = []

for f in features:
    if "_" in f:
        base = f.split("_")[0]
        groups[base].append(f)
    else:
        numeric_cols.append(f)

# --------- Numeric Inputs ----------
st.subheader("Numeric Inputs")
for col in numeric_cols:
    input_data[col] = st.number_input(col, value=0.0)

# --------- Categorical Inputs ----------
st.subheader("Categorical Inputs")

for base, cols in groups.items():
    if len(cols) == 1:
        # binary flag
        input_data[cols[0]] = 1 if st.checkbox(cols[0]) else 0
    else:
        # multi-class group → radio
        options = ["None"] + [c.replace(base + "_", "") for c in cols]
        choice = st.radio(base, options, index=0)

        for c in cols:
            input_data[c] = 1 if c.endswith(choice) else 0

# --------- Predict ----------
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader(f"Churn Probability: {prob:.2f}")

    if prob >= 0.4:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")
