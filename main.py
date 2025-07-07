import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("insurance.pkl")

# Set page config
st.set_page_config(page_title="Insurance Cost Predictor", layout="wide", page_icon="💰")

# Custom CSS for background and styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1588776814546-ec7ee82baad7?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #FFFFFF;
    }
    .main > div {
        background-color: rgba(0,0,0,0.65);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin: 1rem;
    }
    h1, h2, h3 {
        color: #FFD700;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        padding-top: 2rem;
        color: #DDDDDD;
    }
    </style>
""", unsafe_allow_html=True)

# Header with title
st.markdown("<h1 style='text-align: center;'>🤖 Insurance Cost Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Estimate your health insurance cost using AI & machine learning</h4>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2977/2977292.png", width=100)
    st.markdown("## 📁 Upload CSV to Visualize Data")
    file = st.file_uploader("Upload a CSV file", type=['csv'])

    st.markdown("---")
    st.markdown("👨‍💻 **Created by:** Irenee Murhula")
    st.markdown("🔗 [LinkedIn Profile](https://linkedin.com)")
    st.markdown("🧠 Project: AI Insurance Cost Prediction")

# Main container
st.markdown('<div class="main">', unsafe_allow_html=True)

# Dataset visualization
if file is not None:
    try:
        df = pd.read_csv(file)
        st.subheader("📊 Uploaded Data")
        st.dataframe(df)

        # Plot options
        st.markdown("### 📈 Select a Plot to Display")
        plot_type = st.selectbox("Choose a plot type", [
            "Scatter Plot",
            "Bar Chart",
            "Line Chart",
            "Histogram",
            "Correlation Heatmap"
        ])

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for plotting.")
        else:
            if plot_type == "Scatter Plot":
                x_axis = st.selectbox("X-Axis", numeric_cols, key="scatter_x")
                y_axis = st.selectbox("Y-Axis", numeric_cols, key="scatter_y")
                fig, ax = plt.subplots()
                ax.scatter(df[x_axis], df[y_axis], color='orange')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{x_axis} vs {y_axis}")
                st.pyplot(fig)

            elif plot_type == "Bar Chart":
                bar_col = st.selectbox("Column for Bar Chart (categorical or numeric)", df.columns, key="bar")
                bar_data = df[bar_col].value_counts().head(20)
                fig, ax = plt.subplots()
                bar_data.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f"Bar Chart: {bar_col}")
                st.pyplot(fig)

            elif plot_type == "Line Chart":
                x_axis = st.selectbox("X-Axis", numeric_cols, key="line_x")
                y_axis = st.selectbox("Y-Axis", numeric_cols, key="line_y")
                fig, ax = plt.subplots()
                ax.plot(df[x_axis], df[y_axis], color='lime')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"Line Chart: {x_axis} vs {y_axis}")
                st.pyplot(fig)

            elif plot_type == "Histogram":
                hist_col = st.selectbox("Column for Histogram", numeric_cols, key="hist")
                bins = st.slider("Number of Bins", 5, 100, 20)
                fig, ax = plt.subplots()
                ax.hist(df[hist_col], bins=bins, color='purple')
                ax.set_title(f"Histogram of {hist_col}")
                st.pyplot(fig)

            elif plot_type == "Correlation Heatmap":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGnBu", ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

st.markdown("---")
st.subheader("📝 Fill in Your Information")

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=100, value=39)
    bmi = st.number_input('BMI', min_value=15.0, max_value=53.0, value=30.0)

with col2:
    children = st.number_input('Number of Children', min_value=0, max_value=15, value=1)
    sex = st.selectbox('Sex', options=['female', 'male'])

with col3:
    smoker = st.selectbox('Smoker', options=['yes', 'no'])
    region = st.selectbox('Region', options=['southwest', 'southeast', 'northwest', 'northeast'])

# Encode
sex_code = 1 if sex == 'male' else 0
smoker_code = 1 if smoker == 'yes' else 0
region_code = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}[region]

# Input summary
with st.expander("📋 Your Input Summary"):
    st.markdown(f"- **Age:** {age}")
    st.markdown(f"- **Sex:** {sex}")
    st.markdown(f"- **BMI:** {bmi}")
    st.markdown(f"- **Children:** {children}")
    st.markdown(f"- **Smoker:** {smoker}")
    st.markdown(f"- **Region:** {region.capitalize()}")

# Predict
if st.button("🔮 Predict Insurance Cost"):
    try:
        input_data = np.array([[age, sex_code, bmi, children, smoker_code, region_code]])
        prediction = model.predict(input_data)[0]
        st.success(f"💸 **Estimated Insurance Cost: ${prediction:,.2f}**")
        st.balloons()
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown('<div class="footer">Powered by AI & Streamlit • © 2025 Irenee Murhula</div>', unsafe_allow_html=True)
