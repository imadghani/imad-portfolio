from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
from titanic_ml_model import TitanicSurvivalPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model_predictor = None
model_data = None

def load_model():
    """Load the trained model"""
    global model_predictor, model_data
    
    # Try multiple paths for the model file
    model_paths = [
        'titanic_model.pkl',
        'apps/ml_model/titanic_model.pkl',
        os.path.join(os.path.dirname(__file__), 'titanic_model.pkl')
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        logger.error(f"Model file not found in any of: {model_paths}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model_predictor = model_data['model']
        logger.info(f"Model loaded successfully from: {model_path}")
        logger.info(f"Model: {model_data['model_name']}")
        logger.info(f"Performance: Accuracy={model_data['accuracy']:.4f}, AUC={model_data['auc']:.4f}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_predictor is not None
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': model_data['model_name'],
        'features': model_data.get('feature_names', []),
        'accuracy': model_data['accuracy'],
        'auc_score': model_data['auc'],
        'timestamp': model_data.get('timestamp', 'Not available')
    })

@app.route('/predict', methods=['POST'])
def predict_survival():
    """Predict passenger survival"""
    if model_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get passenger data from request
        passenger_data = request.json
        
        # Validate required fields
        required_fields = [
            'gender', 'age', 'class_number', 'fare_amount', 'family_size',
            'port_name', 'has_cabin', 'ticket_type'
        ]
        
        for field in required_fields:
            if field not in passenger_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Preprocess and predict
        processed_data = preprocess_prediction_input(passenger_data)
        prediction_result = make_prediction(processed_data)
        
        return jsonify({
            'prediction': prediction_result,
            'input_data': passenger_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def preprocess_prediction_input(passenger_data):
    """Preprocess passenger data for prediction"""
    # Create a DataFrame with the input data
    df = pd.DataFrame([passenger_data])
    
    # Fill missing values with defaults
    df['age'] = df['age'].fillna(df['age'].median()) if 'age' in df.columns else 30
    df['fare_amount'] = df['fare_amount'].fillna(df['fare_amount'].median()) if 'fare_amount' in df.columns else 32
    
    # Handle default values for missing fields
    defaults = {
        'is_alone_flag': 1 if passenger_data.get('family_size', 1) == 1 else 0,
        'is_group_ticket': 0,
        'passengers_on_ticket': 1,
        'age_group': 'Adult',
        'title': 'Mr' if passenger_data.get('gender') == 'male' else 'Miss',
        'deck_name': 'Unknown Deck'
    }
    
    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value
    
    # Create derived features
    df['age_fare_ratio'] = df['age'] / (df['fare_amount'] + 1)
    df['fare_per_person'] = df['fare_amount'] / df['passengers_on_ticket']
    
    # Encode categorical variables using saved encoders
    categorical_columns = ['gender', 'age_group', 'title', 'class_name', 'port_name', 'deck_name', 'ticket_type']
    
    for col in categorical_columns:
        if col in df.columns and col in model_data['encoders']:
            try:
                df[col + '_encoded'] = model_data['encoders'][col].transform(df[col].astype(str))
            except ValueError:
                # Handle unknown categories
                df[col + '_encoded'] = 0
        else:
            df[col + '_encoded'] = 0
    
    # Convert class_name to class_number if needed
    if 'class_name' in df.columns and 'class_number' not in df.columns:
        class_mapping = {'First Class': 1, 'Second Class': 2, 'Third Class': 3}
        df['class_number'] = df['class_name'].map(class_mapping).fillna(3)
    
    # Select features in the same order as training
    feature_columns = [
        'fare_amount', 'age', 'family_size', 'is_alone_flag', 'class_number',
        'has_cabin', 'is_group_ticket', 'passengers_on_ticket', 'age_fare_ratio', 'fare_per_person'
    ] + [col + '_encoded' for col in categorical_columns]
    
    # Ensure all features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_columns].iloc[0].values

def make_prediction(processed_data):
    """Make prediction using the loaded model"""
    # Reshape for single prediction
    processed_data = processed_data.reshape(1, -1)
    
    # Make prediction
    if model_data['model_name'] in ['Logistic Regression', 'SVM']:
        processed_data_scaled = model_data['scaler'].transform(processed_data)
        prediction = model_data['model'].predict(processed_data_scaled)[0]
        probability = model_data['model'].predict_proba(processed_data_scaled)[0, 1]
    else:
        prediction = model_data['model'].predict(processed_data)[0]
        probability = model_data['model'].predict_proba(processed_data)[0, 1]
    
    return {
        'survived': bool(prediction),
        'survival_probability': float(probability),
        'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'
    }

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict survival for multiple passengers"""
    if model_predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get batch data from request
        batch_data = request.json
        
        if not isinstance(batch_data, list):
            return jsonify({'error': 'Input must be a list of passenger data'}), 400
        
        predictions = []
        for passenger_data in batch_data:
            try:
                processed_data = preprocess_prediction_input(passenger_data)
                prediction_result = make_prediction(processed_data)
                predictions.append({
                    'input': passenger_data,
                    'prediction': prediction_result
                })
            except Exception as e:
                predictions.append({
                    'input': passenger_data,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'total_count': len(batch_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/predict/example', methods=['GET'])
def prediction_example():
    """Get example passenger data for testing"""
    examples = [
        {
            "name": "Upper Class Lady",
            "gender": "female",
            "age": 35,
            "class_number": 1,
            "fare_amount": 75.25,
            "family_size": 2,
            "port_name": "Southampton",
            "has_cabin": True,
            "ticket_type": "Prefix Ticket"
        },
        {
            "name": "Working Class Man",
            "gender": "male",
            "age": 28,
            "class_number": 3,
            "fare_amount": 7.75,
            "family_size": 1,
            "port_name": "Southampton",
            "has_cabin": False,
            "ticket_type": "Numeric Ticket"
        },
        {
            "name": "Middle Class Child",
            "gender": "female",
            "age": 12,
            "class_number": 2,
            "fare_amount": 30.0,
            "family_size": 4,
            "port_name": "Cherbourg",
            "has_cabin": True,
            "ticket_type": "Prefix Ticket"
        }
    ]
    
    return jsonify({
        'examples': examples,
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'content_type': 'application/json'
        }
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting Titanic Survival Prediction API")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("Failed to load model. Please train a model first.")
        print("Run 'python titanic_ml_model.py' to train a model first.") 