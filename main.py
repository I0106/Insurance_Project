import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load("insurance.pkl")

# App title and styling
st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
st.title("💰 Insurance Cost Prediction App")

# Custom CSS to set background
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://cdn.download.ams.birds.cornell.edu/api/v1/asset/308065631/1800");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# File upload
file = st.file_uploader("📂 Upload a CSV file to visualize", type=['csv'])

# Display dataframe and plot if file is uploaded
if file is not None:
    try:
        df = pd.read_csv(file)
        st.markdown("### 📊 Uploaded Data")
        st.dataframe(df)

        # Plot first two numeric columns if available
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) >= 2:
            fig, ax = plt.subplots()
            ax.scatter(df[numeric_columns[0]], df[numeric_columns[1]])
            ax.set_xlabel(numeric_columns[0])
            ax.set_ylabel(numeric_columns[1])
            ax.set_title("Scatter Plot")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

# App description
st.markdown("""
Welcome to the **Insurance Cost Predictor**!  
This app predicts the cost of health insurance based on your personal information like **age**, **BMI**, **region**, and **smoking habits**.
""")

st.markdown("---")

# Input section
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('🔢 Age', min_value=1, max_value=100, value=39)
    bmi = st.number_input('⚖️ BMI', min_value=15.0, max_value=53.0, value=30.0)

with col2:
    children = st.number_input('👶 Number of Children', min_value=0, max_value=15, value=1)
    sex = st.selectbox('🧑 Sex', options=['female', 'male'])

with col3:
    smoker = st.selectbox('🚬 Smoker', options=['yes', 'no'])
    region = st.selectbox('🌍 Region', options=['southwest', 'southeast', 'northwest', 'northeast'])

# Encode categorical variables
sex_code = 1 if sex == 'male' else 0
smoker_code = 1 if smoker == 'yes' else 0
region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
region_code = region_mapping[region]

# Display input summary
st.markdown("### 📋 Your Information")
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.write(f"**Age:** {age}")
    st.write(f"**Sex:** {sex}")
    st.write(f"**BMI:** {bmi}")
with info_col2:
    st.write(f"**Children:** {children}")
    st.write(f"**Smoker:** {smoker}")
    st.write(f"**Region:** {region.capitalize()}")

# Prediction
if st.button("🔮 Predict Insurance Cost"):
    input_data = np.array([[age, sex_code, bmi, children, smoker_code, region_code]])
    try:
        predicted_cost = model.predict(input_data)[0]
        st.success(f"💸 **Predicted Insurance Cost: ${predicted_cost:,.2f}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
