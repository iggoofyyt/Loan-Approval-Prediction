
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('loan_approval_model.pkl')

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)


import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_approval_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# Heading
st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant information to predict loan approval.")

st.divider()

# Applicant form
with st.form("loan_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        married = st.selectbox("Married", ["Yes", "No"])

    with col3:
        dependents = st.selectbox(
            "Dependents", ["0", "1", "2", "3+"]
        )

    col4, col5 = st.columns(2)

    with col4:
        education = st.selectbox(
            "Education", ["Graduate", "Not Graduate"]
        )

    with col5:
        self_employed = st.selectbox(
            "Self Employed", ["No", "Yes"]
        )

    st.divider()
    st.subheader("Financial Information")

    col6, col7 = st.columns(2)

    with col6:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

    with col7:
        coapplicant_income = st.number_input(
            "Co-applicant Income",
            min_value=0,
            value=0
        )

    col8, col9 = st.columns(2)

    with col8:
        loan_amount = st.number_input(
            "Loan Amount (dataset units)",
            min_value=1,
            value=150
        )

    with col9:
        loan_term = st.selectbox(
            "Loan Amount Term (months)",
            [360, 300, 240, 180, 120, 60, 36, 12]
        )

    st.divider()
    st.subheader("Credit and Property Information")

    col10, col11 = st.columns(2)

    with col10:
        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x: "Good (1)" if x == 1.0 else "Poor (0)"
        )

    with col11:
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    submitted = st.form_submit_button(
        "Predict Loan Status",
        use_container_width=True
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan is predicted to be APPROVED!")
    else:
        st.error("❌ Loan is predicted to be REJECTED!")

    st.caption(
        "This is an ML model prediction, not an actual bank decision."
    )

