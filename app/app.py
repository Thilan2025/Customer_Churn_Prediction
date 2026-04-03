import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow app to import from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from predict import predict  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("""
This interactive dashboard predicts customer churn using a machine learning model trained on telecom data.

**Key features:**
- Predict churn probability
- Identify customer risk level
- Support business decision-making
""")

st.write(
    "This application predicts whether a customer is likely to churn "
    "using the best-performing model from the dissertation project."
)

st.markdown("---")
st.markdown("### Customer Information")

gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=0.1)


def get_risk_label(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.60:
        return "Medium Risk"
    return "High Risk"


if st.button("Predict Churn"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    prediction, probability = predict(input_data)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 High Risk: The customer is likely to churn.")
    else:
        st.success("✅ Low Risk: The customer is not likely to churn.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{probability:.2%}")

    with col2:
        st.metric("Risk Level", get_risk_label(probability))

    st.write("**Model Used:** Logistic Regression")

    st.markdown("### Interpretation")
    st.write(
        "This prediction is based on the trained Logistic Regression model. "
        "Higher churn probability suggests that the customer may be at greater risk "
        "of leaving the service."
    )

st.markdown("---")
st.subheader("📊 Model Performance Summary")

results = pd.read_csv(PROJECT_ROOT / "models" / "model_results.csv")
st.dataframe(results)

st.markdown("---")
st.caption("Developed as part of MSc Data Analytics Dissertation Project")