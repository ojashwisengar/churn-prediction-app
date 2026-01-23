import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

st.title("Customer Churn Prediction")
st.write("Enter Customer details to predict churn risk")

input_data = {}

for f in features:
  input_data[f] = st.number_input(f, value = 0.0)

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
  prob = model.predict_proba(input_df)[0][1]
  st.subheader(f"Churn Probability: {prob: .2f}")
  if prob >= 0.4:
    st.error("High risk of churn")
  else:
    st.success("Low risk of churn")
