from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

app = Flask(__name__)

# Load the model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'strokePredict.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()
# Load test data (assuming you have a CSV file with test data)
test_data = pd.read_csv('/data/healthcare-dataset-stroke-data.csv')
X_test = test_data.drop('stroke', axis=1)
y_test = test_data['stroke']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input features from the form
        features = [float(request.form[f'feature{i}']) for i in range(1, len(X_test.columns) + 1)]
        # Make prediction
        prediction = model.predict([features])[0]
        return render_template('/front-end/result.html', prediction=prediction)
    return render_template('/front-end/index.html', feature_names=X_test.columns)

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json['features']
    prediction = model.predict([features])[0]
    return jsonify({'prediction': int(prediction)})

@app.route('/evaluate', methods=['GET'])
def evaluate():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return render_template('/front-end/evaluation.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1)

@app.route('/retrain', methods=['POST'])
def retrain():
    # fetching new data here
    # I just use the existing test data
    new_model = type(model)()  # Create a new instance of the same model type
    new_model.fit(X_test, y_test)
    
    # Save the new model
    with open('strokePredict.pkl', 'wb') as f:
        pickle.dump(new_model, f)
    
    model = new_model
    
    return jsonify({'message': 'Model retrained successfully'})

if __name__ == '__main__':
    app.run(debug=True)