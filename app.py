import streamlit as st
import pandas as pd
import joblib

# Page Title

st.title("📊 Customer Churn Analytics Platform")

st.markdown("""
Predict customer churn risk using machine learning.

### Features

* Churn Prediction
* Risk Scoring
* Customer Segmentation
* Business Analytics Dashboard
* Download Predictions
  """)

# Load Model

model = joblib.load("churn_model.pkl")

# Upload CSV

uploaded_file = st.file_uploader(
"Upload CSV File",
type=["csv"]
)

# Risk Category Function

def risk_category(prob):
  if prob >= 70:
    return "High"
  elif prob >= 40:
    return "Medium"
  else:
    return "Low"

# Main App

if uploaded_file is not None:

```
# Read uploaded file
data = pd.read_csv(uploaded_file)

# Predictions
predictions = model.predict(data)
probabilities = model.predict_proba(data)

# Add prediction columns
data["Churn_Probability"] = probabilities[:, 1] * 100
data["Risk_Level"] = data["Churn_Probability"].apply(risk_category)

data["Predicted_Churn"] = predictions
data["Predicted_Churn"] = data["Predicted_Churn"].map({
    0: "No",
    1: "Yes"
})

# Dashboard Metrics
total_customers = len(data)
high_risk = len(data[data["Risk_Level"] == "High"])
medium_risk = len(data[data["Risk_Level"] == "Medium"])
low_risk = len(data[data["Risk_Level"] == "Low"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", total_customers)
col2.metric("High Risk", high_risk)
col3.metric("Medium Risk", medium_risk)
col4.metric("Low Risk", low_risk)

# Risk Chart
st.subheader("Risk Distribution")

risk_counts = data["Risk_Level"].value_counts()

st.bar_chart(risk_counts)

# Download Button
st.download_button(
    label="📥 Download Predictions",
    data=data.to_csv(index=False),
    file_name="churn_predictions.csv",
    mime="text/csv"
)

# Results Table
st.subheader("Prediction Results")

st.write(data)
```
