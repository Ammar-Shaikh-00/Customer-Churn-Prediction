import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")

# Define the preprocessing function
def preprocess_input(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                      MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
                      TechSupport, StreamingTV, StreamingMovies, PaperlessBilling,
                      MonthlyCharges, TotalCharges, InternetService, Contract, PaymentMethod):
    
    # Convert categorical inputs into numerical format
    categorical_mapping = {
        "gender": {"Male": 0, "Female": 1},
        "YesNo": {"Yes": 1, "No": 0},
        "MultipleLines": {"No phone service": 0, "No": 0, "Yes": 1},
        "InternetService": {"DSL": [1, 0, 0], "Fiber optic": [0, 1, 0], "No": [0, 0, 1]},
        "Contract": {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]},
        "PaymentMethod": {
            "Bank transfer (automatic)": [1, 0, 0, 0],
            "Credit card (automatic)": [0, 1, 0, 0],
            "Electronic check": [0, 0, 1, 0],
            "Mailed check": [0, 0, 0, 1]
        }
    }
    
    # Convert categorical inputs
    gender = categorical_mapping["gender"][gender]
    SeniorCitizen = int(SeniorCitizen)
    Partner = categorical_mapping["YesNo"][Partner]
    Dependents = categorical_mapping["YesNo"][Dependents]
    PhoneService = categorical_mapping["YesNo"][PhoneService]
    MultipleLines = categorical_mapping["MultipleLines"][MultipleLines]
    OnlineSecurity = categorical_mapping["YesNo"][OnlineSecurity]
    OnlineBackup = categorical_mapping["YesNo"][OnlineBackup]
    DeviceProtection = categorical_mapping["YesNo"][DeviceProtection]
    TechSupport = categorical_mapping["YesNo"][TechSupport]
    StreamingTV = categorical_mapping["YesNo"][StreamingTV]
    StreamingMovies = categorical_mapping["YesNo"][StreamingMovies]
    PaperlessBilling = categorical_mapping["YesNo"][PaperlessBilling]
    
    # One-hot encode multi-category columns
    InternetService_values = categorical_mapping["InternetService"][InternetService]
    Contract_values = categorical_mapping["Contract"][Contract]
    PaymentMethod_values = categorical_mapping["PaymentMethod"][PaymentMethod]
    
    # Construct DataFrame with the same feature order used in training
    feature_values = [
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        PaperlessBilling, MonthlyCharges, TotalCharges,
        *InternetService_values, *Contract_values, *PaymentMethod_values
    ]
    
    feature_names = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    # Create DataFrame
    input_df = pd.DataFrame([feature_values], columns=feature_names)

    # Ensure input_data matches model's expected features
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match model’s expected input
    input_df = input_df[model.feature_names_in_]

    return input_df

# Streamlit UI
st.title("Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", ["0", "1"])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0)
TotalCharges = st.number_input("Total Charges ($)", min_value=0.0)
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

if st.button("Predict"):
    # Preprocess input
    input_data = preprocess_input(
        gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
        MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, PaperlessBilling,
        MonthlyCharges, TotalCharges, InternetService, Contract, PaymentMethod
    )
    
    # Debugging: Print missing columns
    model_features = set(model.feature_names_in_)
    input_features = set(input_data.columns)
    missing_features = model_features - input_features
    extra_features = input_features - model_features
    print(f"Missing features: {missing_features}")  # Debugging
    print(f"Extra features: {extra_features}")  # Debugging

    # Make prediction
    prediction = model.predict(input_data)
    
    st.success(f"Churn Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")  
