import streamlit as st
import pandas as pd
import pickle

# ---------------------------------
# Load Model, Scaler, Feature Columns
# ---------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ---------------------------------
# App Title
# ---------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Upload patient data (CSV) to predict heart disease risk.")

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data)

    # ---------------------------------
    # Handle patient_id
    # ---------------------------------
    if "patient_id" not in data.columns:
        st.error("❌ CSV must contain a 'patient_id' column.")
        st.stop()

    patient_ids = data["patient_id"]

    # Drop patient_id before preprocessing
    data = data.drop(columns=["patient_id"])

    # ---------------------------------
    # One-hot encode categorical feature
    # ---------------------------------
    data_encoded = pd.get_dummies(
        data,
        columns=["thal"],
        drop_first=True
    )

    # ---------------------------------
    # Ensure all training columns exist
    # ---------------------------------
    for col in feature_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Reorder columns to match training
    data_encoded = data_encoded[feature_columns]

    # ---------------------------------
    # Scale numerical features
    # ---------------------------------
    numerical_features = [
        "age",
        "resting_blood_pressure",
        "serum_cholesterol_mg_per_dl",
        "max_heart_rate_achieved",
        "oldpeak_eq_st_depression"
    ]

    data_encoded[numerical_features] = scaler.transform(
        data_encoded[numerical_features]
    )

    # ---------------------------------
    # Prediction
    # ---------------------------------
    predictions = model.predict(data_encoded)
    probabilities = model.predict_proba(data_encoded)[:, 1]

    # ---------------------------------
    # Results
    # ---------------------------------
    result_df = data.copy()
    result_df.insert(0, "patient_id", patient_ids)
    result_df["Heart Disease Prediction"] = predictions
    result_df["Risk Probability"] = probabilities.round(3)

    st.subheader("Prediction Results")
    st.dataframe(result_df)