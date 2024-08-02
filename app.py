import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model
model_path = os.path.join('model', 'strokePredict.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data_path = os.path.join('Dataset', 'healthcare-stroke-dataset.csv')
df = pd.read_csv(data_path)

# Set page title
st.set_page_config(page_title="Stroke Analysis and Prediction", layout="wide")

# Title
st.title("Stroke Analysis and Prediction")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Data Exploration", "Prediction"])

if page == "Data Exploration":
    st.header("Data Exploration")

    # Display basic statistics
    st.subheader("Dataset Overview")
    st.write(df.describe())

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Display distribution of stroke cases
    st.subheader("Distribution of Stroke Cases")
    fig, ax = plt.subplots()
    df['stroke'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

elif page == "Prediction":
    st.header("Stroke Prediction")

    # Create input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    # Create a dictionary of inputs
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input data (you may need to adjust this based on your model's requirements)
    # For example, convert categorical variables to numerical
    input_df['gender'] = input_df['gender'].map({'Male': 0, 'Female': 1})
    input_df['ever_married'] = input_df['ever_married'].map({'No': 0, 'Yes': 1})
    input_df = pd.get_dummies(input_df, columns=['work_type', 'Residence_type', 'smoking_status'])

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.write("The model predicts: No stroke")
        else:
            st.write("The model predicts: Stroke")

        st.write(f"Probability of stroke: {probability[0][1]:.2%}")

# Add a note about the model and data source
st.sidebar.markdown("---")
st.sidebar.write("Note: This is for Academic purposes. Always consult with a healthcare professional for medical advice.")