import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model
model_path = os.path.join('model', 'strokePrediction.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the dataset
data_path = os.path.join('dataset', 'healthcare-dataset-stroke-data.csv')
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

    # Preprocess data for correlation
    df_corr = df.copy()
    
    # Convert binary categorical variables
    binary_columns = ['gender', 'ever_married', 'Residence_type']
    for col in binary_columns:
        df_corr[col] = pd.factorize(df_corr[col])[0]
    
    # Convert multi-category variables
    multi_cat_columns = ['work_type', 'smoking_status']
    df_corr = pd.get_dummies(df_corr, columns=multi_cat_columns)
    
    # Select only numeric columns
    numeric_columns = df_corr.select_dtypes(include=[np.number]).columns
    df_corr = df_corr[numeric_columns]

    # Display correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df_corr.corr()
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

    # Preprocess input data
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    input_df['ever_married'] = input_df['ever_married'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode categorical variables
    categorical_columns = ['work_type', 'Residence_type', 'smoking_status']
    input_df = pd.get_dummies(input_df, columns=categorical_columns)

    # Make prediction
    if st.button("Predict"):
        try:
            # Convert all column names to strings
            input_df.columns = input_df.columns.astype(str)

            # Get the feature names that the model expects
            model_features = [str(feature) for feature in model.feature_names_in_]

            # Ensure all expected features are present
            for feature in model_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0

            # Select only the features that the model expects
            input_df_model = input_df[model_features]

            prediction = model.predict(input_df_model)
            probability = model.predict_proba(input_df_model)

            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.write("The model predicts: No stroke")
            else:
                st.write("The model predicts: Stroke")

            st.write(f"Probability of stroke: {probability[0][1]:.2%}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Model features:", model.feature_names_in_)
            st.write("Input features:", input_df.columns.tolist())

# Add a note about the model and data source
st.sidebar.markdown("---")
st.sidebar.write("Note: This is for Academic purposes. Always consult with a healthcare professional for medical advice.")
