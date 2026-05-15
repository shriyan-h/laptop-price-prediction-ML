import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

# Load the model and dataset
@st.cache_resource
def load_model():
    return joblib.load('Linear_Model_laptop.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('laptop_price - dataset.csv')

model = load_model()
df = load_data()
feature_names = model.feature_names_in_

# Title and Description
st.title("💻 Laptop Price Predictor")
st.markdown("Enter the specifications of the laptop to estimate its price in Euro.")

# Create Layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", sorted(df['Company'].unique()))
    type_name = st.selectbox("Type", sorted(df['TypeName'].unique()))
    inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    screen_res = st.selectbox("Screen Resolution", sorted(df['ScreenResolution'].unique()))

with col2:
    cpu_company = st.selectbox("CPU Brand", sorted(df['CPU_Company'].unique()))
    cpu_type = st.selectbox("CPU Model", sorted(df['CPU_Type'].unique()))
    cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
    ram = st.selectbox("RAM (GB)", sorted(df['RAM (GB)'].unique()))

st.divider()

col3, col4 = st.columns(2)
with col3:
    memory = st.selectbox("Memory/Storage", sorted(df['Memory'].unique()))
    gpu_company = st.selectbox("GPU Brand", sorted(df['GPU_Company'].unique()))
    
with col4:
    gpu_type = st.selectbox("GPU Model", sorted(df['GPU_Type'].unique()))
    opsys = st.selectbox("Operating System", sorted(df['OpSys'].unique()))

# Prediction Logic
if st.button("Predict Price", type="primary", use_container_width=True):
    # Initialize input vector with zeros
    input_data = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    
    # 1. Fill Numerical Values
    input_data['Inches'] = inches
    input_data['CPU_Frequency (GHz)'] = cpu_freq
    input_data['RAM (GB)'] = ram
    
    # 2. Map Categorical Values to One-Hot Columns
    def set_category(col_name, value):
        dummy_col = f"{col_name}_{value}"
        if dummy_col in input_data.columns:
            input_data[dummy_col] = 1

    set_category('Company', company)
    set_category('TypeName', type_name)
    set_category('ScreenResolution', screen_res)
    set_category('CPU_Company', cpu_company)
    set_category('CPU_Type', cpu_type)
    set_category('Memory', memory)
    set_category('GPU_Company', gpu_company)
    set_category('GPU_Type', gpu_type)
    set_category('OpSys', opsys)
    
    # Perform Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.balloons()
    st.success(f"### Estimated Price: €{prediction:,.2f}")
    
    # Conversion logic (Optional)
    st.info(f"Approximate Price in USD: ${prediction * 1.08:,.2f}")

st.markdown("---")
st.caption("Note: This model uses Linear Regression and performance depends on the provided dataset features.")