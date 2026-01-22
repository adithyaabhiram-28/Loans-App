import streamlit as st
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# App Title & Description
# -------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("🏦 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict loan approval "
    "based on applicant financial and credit details."
)

# -------------------------------
# Load & Prepare Data
# -------------------------------
@st.cache_data
@st.cache_data
def load_and_train(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    from sklearn.preprocessing import LabelEncoder
    le_self = LabelEncoder()
    df['Self_Employed'] = le_self.fit_transform(df['Self_Employed'])

    le_target = LabelEncoder()
    df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])

    X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Self_Employed']]
    y = df['Loan_Status']

    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train, scaler
    
uploaded_file = st.file_uploader("Upload loans.csv", type=["csv"])

if uploaded_file:
    X_train, y_train, scaler = load_and_train(uploaded_file)
else:
    st.stop()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🧾 Applicant Details")

income = st.sidebar.number_input(
    "Applicant Income", min_value=0, value=5000, step=500
)

loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=0, value=150, step=10
)

credit_history = st.sidebar.radio(
    "Credit History", ["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status", ["Self Employed", "Not Self Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# Encode inputs
credit_val = 1 if credit_history == "Yes" else 0
employment_val = 1 if employment_status == "Self Employed" else 0

input_data = np.array([[income, loan_amount, credit_val, employment_val]])
input_scaled = scaler.transform(input_data)

# -------------------------------
# Model Selection
# -------------------------------
st.subheader("⚙️ Select SVM Kernel")

kernel_choice = st.radio(
    "Choose Kernel Type",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel_choice == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", gamma="scale", probability=True)

model.fit(X_train, y_train)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("✅ Check Loan Eligibility"):
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max() * 100

    st.markdown("---")

    # -------------------------------
    # Output Section
    # -------------------------------
    if prediction == 1:
        st.success("✅ **Loan Approved**")
        explanation = (
            "Based on strong credit history and income pattern, "
            "the applicant is likely to repay the loan."
        )
    else:
        st.error("❌ **Loan Rejected**")
        explanation = (
            "Based on credit history and income pattern, "
            "the applicant is unlikely to repay the loan."
        )

    st.write(f"**Confidence Score:** {confidence:.2f}%")
    st.write(f"**Kernel Used:** {kernel_choice}")

    # -------------------------------
    # Business Explanation
    # -------------------------------
    st.info(explanation)
