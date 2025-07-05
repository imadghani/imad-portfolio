#!/usr/bin/env python3

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
model = None
model_info = None

def load_model():
    """Load the trained model"""
    global model, model_info
    
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
        model_data = joblib.load(model_path)
        
        model = model_data['model']
        model_info = model_data
        logger.info(f"✅ Model loaded successfully from: {model_path}")
        logger.info(f"📊 Model: {model_data['model_name']}")
        logger.info(f"🎯 Performance: Accuracy={model_data['metrics']['accuracy']:.4f}, AUC={model_data['metrics']['auc_score']:.4f}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/model/info', methods=['GET'])
def model_info_endpoint():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': model_info['model_name'],
        'accuracy': model_info['metrics']['accuracy'],
        'auc_score': model_info['metrics']['auc_score'],
        'timestamp': model_info.get('timestamp', 'Not available')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Simple prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get passenger data from request
        data = request.json
        
        # Simple validation - check for basic fields
        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create simple feature array (matching training data structure)
        features = [
            data['Pclass'],
            1 if data['Sex'] == 'male' else 0,  # Sex_male
            data['Age'],
            data['SibSp'],
            data['Parch'],
            data['Fare'],
            1 if data['Embarked'] == 'Q' else 0,  # Embarked_Q
            1 if data['Embarked'] == 'S' else 0,  # Embarked_S
        ]
        
        # Make prediction
        X = np.array(features).reshape(1, -1)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0, 1]
        
        return jsonify({
            'prediction': {
                'survived': bool(prediction),
                'survival_probability': float(probability),
                'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'
            },
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/example', methods=['GET'])
def prediction_example():
    """Get example passenger data for testing"""
    examples = [
        {
            "name": "Upper Class Lady",
            "Pclass": 1,
            "Sex": "female",
            "Age": 35,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 75.25,
            "Embarked": "S"
        },
        {
            "name": "Working Class Man",
            "Pclass": 3,
            "Sex": "male",
            "Age": 28,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.75,
            "Embarked": "S"
        },
        {
            "name": "Middle Class Child",
            "Pclass": 2,
            "Sex": "female",
            "Age": 12,
            "SibSp": 1,
            "Parch": 2,
            "Fare": 30.0,
            "Embarked": "C"
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
    print("🚀 Starting Titanic Survival Prediction API...")
    
    # Load model on startup
    if load_model():
        print("🎯 API will be available at: http://localhost:5001")
        print("📊 Available endpoints:")
        print("  • GET  /health - Health check")
        print("  • GET  /model/info - Model information")
        print("  • POST /predict - Single prediction")
        print("  • GET  /predict/example - Example data")
        print()
        print("🔧 Example API test:")
        print("curl -X POST http://localhost:5001/predict \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"Pclass\": 1, \"Sex\": \"female\", \"Age\": 25, \"SibSp\": 0, \"Parch\": 0, \"Fare\": 50, \"Embarked\": \"S\"}'")
        print()
        app.run(host='0.0.0.0', port=5001, debug=False)
    else:
        print("❌ Failed to load model. Please train a model first.")
        print("Run: python titanic_ml_model.py")
        sys.exit(1) 