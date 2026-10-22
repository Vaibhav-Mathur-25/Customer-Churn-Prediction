import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------------------------------------------------------------------------
# Train the model once, cached — exact same pipeline as Churn_prediction.ipynb
# ---------------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # TotalCharges cleanup: blanks -> 0 (all correspond to tenure=0, new customers)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop("customerID", axis=1)

    # Binary Yes/No + gender mapping
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
    for col in yes_no_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Collapse "No internet/phone service" into "No"
    service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in service_cols:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode remaining categoricals (drop_first=True, same as notebook)
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, X.columns.tolist(), numeric_cols, report


model, scaler, feature_columns, numeric_cols, report = load_and_train()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📉 Customer Churn Predictor")
st.caption("Logistic Regression baseline — telecom churn, trained on the Telco Customer Churn dataset")

with st.expander("ℹ️ Model performance on held-out test set"):
    churn_metrics = report["1"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision (Churn)", f"{churn_metrics['precision']:.2f}")
    c2.metric("Recall (Churn)", f"{churn_metrics['recall']:.2f}")
    c3.metric("F1 (Churn)", f"{churn_metrics['f1-score']:.2f}")

st.divider()
st.subheader("Enter customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, step=10.0)

if st.button("Predict Churn", type="primary", use_container_width=True):
    raw = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": 0 if multiple_lines in ("No", "No phone service") else 1,
        "OnlineSecurity": 0 if online_security in ("No", "No internet service") else 1,
        "OnlineBackup": 0 if online_backup in ("No", "No internet service") else 1,
        "DeviceProtection": 0 if device_protection in ("No", "No internet service") else 1,
        "TechSupport": 0 if tech_support in ("No", "No internet service") else 1,
        "StreamingTV": 0 if streaming_tv in ("No", "No internet service") else 1,
        "StreamingMovies": 0 if streaming_movies in ("No", "No internet service") else 1,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    }

    row = pd.DataFrame([raw])
    # Align to training column order; fill anything missing with 0
    row = row.reindex(columns=feature_columns, fill_value=0)
    row[numeric_cols] = scaler.transform(row[numeric_cols])

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Likely to CHURN — probability: {prob:.1%}")
    else:
        st.success(f"✅ Likely to STAY — churn probability: {prob:.1%}")

    st.progress(float(prob))

st.divider()
st.caption("Model: Logistic Regression · Dataset: IBM Telco Customer Churn (7,043 customers) · Best performer among LR, Random Forest, XGBoost on F1")
