from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset for preprocessing structure
data = pd.read_csv('synthetic_disease_dataset.csv')

# Preprocessing: Encode categorical variables
categorical_columns = ['Gender', 'Exercise_Level', 'Smoking_Habit', 'Diet_Type', 'Chronic_Conditions', 'Disease']
encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = encoders[col].fit_transform(data[col])

# Scale numeric features
numeric_columns = ['Age', 'BMI', 'Sleep_Hours', 'Stress_Level']
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Separate features and target for disease prediction
X = data.drop(columns=['Disease'])
y = data['Disease']

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save the trained model and preprocessing tools
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('encoders.pkl', 'wb') as encoder_file:
    pickle.dump(encoders, encoder_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model and encoders
        model = pickle.load(open('model.pkl', 'rb'))
        encoders = pickle.load(open('encoders.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        # Get input data from the request
        data = request.json

        # Preprocess input data
        features = [
            data['Age'],
            encoders['Gender'].transform([data['Gender']])[0],
            data['BMI'],
            encoders['Exercise_Level'].transform([data['Exercise_Level']])[0],
            encoders['Smoking_Habit'].transform([data['Smoking_Habit']])[0],
            encoders['Breakfast_Skipped'].transform([data['Breakfast_Skipped']])[0],
            data['Sleep_Hours'],
            encoders['Diet_Type'].transform([data['Diet_Type']])[0],
            data['Stress_Level'],
            encoders['Chronic_Conditions'].transform([data['Chronic_Conditions']])[0]
        ]

        # Scale numeric features
        numeric_indices = [0, 2, 6, 8]  # Indices of numeric features
        features = np.array(features).reshape(1, -1)
        features[:, numeric_indices] = scaler.transform(features[:, numeric_indices])

        # Make prediction
        prediction = model.predict(features)[0]
        predicted_disease = encoders['Disease'].inverse_transform([prediction])[0]

        # Return the prediction as a response
        return jsonify({'prediction': predicted_disease})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
