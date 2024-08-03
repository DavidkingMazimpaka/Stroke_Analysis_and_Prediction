# Stroke Analysis and Prediction

## Summary

This project aims to predict whether a patient is likely to have a stroke based on various input parameters such as gender, age, medical conditions, and smoking status. The dataset contains information about patients, including their unique identifier, demographic details, medical history, and whether they have experienced a stroke.

The main objective of this project is to explore different classification algorithms and evaluate their performance in predicting stroke risk. The notebook `stroke_prediction.ipynb` contains the implementation and analysis of the stroke prediction model.

### Model File

The `strokePrediction.pkl` file located in `model folder` which contains the trained stroke prediction model. This file can be used to make predictions on new patient data without having to retrain the model.

### stroke_prediction.ipynb

The `stroke_prediction.ipynb` Jupyter Notebook file contains the following:

1. **Data Exploration and Preprocessing**:
   - Load the dataset
   - Explore the data
   - Handle missing values
   - Encode categorical variables
   - Split the data into training and testing sets

2. **Model Training and Evaluation**:
   - Try different classification algorithms, such as Logistic Regression, Decision Tree, Random Forest, and others
   - Evaluate the performance of each model using metrics like accuracy, precision, recall, and F1-score
   - Select the best performing model and save it as `model.pkl`

3. **Model Deployment**:
   - Provide instructions on how to use the saved model (`model.pkl`) to make predictions on new patient data

4. **Conclusion**:
   - Summarize the findings
   - Discuss the limitations of the current approach
   - Suggest potential improvements or future work

To run the `stroke_prediction.ipynb` notebook, you will need to have the following dependencies installed:

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn

Make sure to follow the instructions in the notebook to execute the code and explore the stroke prediction model.
