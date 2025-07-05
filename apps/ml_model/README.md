# 🤖 Titanic Survival Prediction Model

A comprehensive machine learning system for predicting passenger survival on the RMS Titanic using data from BigQuery.

## Features

- **Multiple ML Models**: Trains and compares Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Hyperparameter Tuning**: Automated parameter optimization using GridSearchCV
- **Feature Engineering**: Creates derived features and handles categorical encoding
- **Model Persistence**: Saves and loads trained models
- **REST API**: Flask-based API for making predictions
- **Batch Processing**: Supports both single and batch predictions
- **Performance Metrics**: Comprehensive evaluation with accuracy, AUC, and visualizations

## Setup

### 1. Prerequisites
- Python virtual environment activated
- BigQuery data tables available (run dbt models first)
- Required dependencies installed

### 2. Install Dependencies
```bash
pip install scikit-learn joblib flask
```

### 3. Train the Model
```bash
python apps/ml_model/titanic_ml_model.py
```

This will:
- Load data from BigQuery
- Train multiple ML models
- Perform hyperparameter tuning
- Save the best model as `titanic_model.pkl`
- Generate performance visualizations

### 4. Start the API Server
```bash
python apps/ml_model/prediction_api.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model/info
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "gender": "female",
  "age": 35,
  "class_number": 1,
  "fare_amount": 75.25,
  "family_size": 2,
  "port_name": "Southampton",
  "has_cabin": true,
  "ticket_type": "Prefix Ticket"
}
```

### Batch Predictions
```bash
POST /predict/batch
Content-Type: application/json

[
  {
    "gender": "female",
    "age": 35,
    "class_number": 1,
    "fare_amount": 75.25,
    "family_size": 2,
    "port_name": "Southampton",
    "has_cabin": true,
    "ticket_type": "Prefix Ticket"
  },
  {
    "gender": "male",
    "age": 28,
    "class_number": 3,
    "fare_amount": 7.75,
    "family_size": 1,
    "port_name": "Southampton",
    "has_cabin": false,
    "ticket_type": "Numeric Ticket"
  }
]
```

### Example Data
```bash
GET /predict/example
```

## API Response Format

### Single Prediction Response
```json
{
  "prediction": {
    "survived": true,
    "survival_probability": 0.85,
    "confidence": "High"
  },
  "input_data": { ... },
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Prediction Response
```json
{
  "predictions": [
    {
      "input": { ... },
      "prediction": { ... }
    }
  ],
  "total_count": 2,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Model Features

The model uses the following features for prediction:

### Numerical Features
- `fare_amount`: Ticket fare
- `age`: Passenger age
- `family_size`: Number of family members
- `class_number`: Passenger class (1, 2, 3)
- `age_fare_ratio`: Age divided by fare amount
- `fare_per_person`: Fare divided by passengers on ticket

### Categorical Features (Encoded)
- `gender`: Male/Female
- `age_group`: Child/Adult/Senior
- `title`: Mr, Mrs, Miss, etc.
- `class_name`: First/Second/Third Class
- `port_name`: Southampton/Cherbourg/Queenstown
- `deck_name`: Ship deck (A-G)
- `ticket_type`: Prefix/Numeric/Slash Format

### Boolean Features
- `is_alone_flag`: Whether passenger is traveling alone
- `has_cabin`: Whether cabin information is available
- `is_group_ticket`: Whether ticket is shared

## Model Performance

The system trains multiple models and selects the best performing one:

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Gradient boosted trees
- **Logistic Regression**: Linear classification
- **SVM**: Support Vector Machine

Performance metrics include:
- **Accuracy**: Overall classification accuracy
- **AUC Score**: Area Under the ROC Curve
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Precision, recall, and F1-score

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model/info

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "age": 35,
    "class_number": 1,
    "fare_amount": 75.25,
    "family_size": 2,
    "port_name": "Southampton",
    "has_cabin": true,
    "ticket_type": "Prefix Ticket"
  }'
```

### Using Python
```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', json={
    "gender": "female",
    "age": 35,
    "class_number": 1,
    "fare_amount": 75.25,
    "family_size": 2,
    "port_name": "Southampton",
    "has_cabin": True,
    "ticket_type": "Prefix Ticket"
})

print(response.json())
```

## Files Structure

```
apps/ml_model/
├── titanic_ml_model.py      # Main ML pipeline
├── prediction_api.py        # Flask API server
├── titanic_model.pkl        # Trained model (generated)
├── feature_importance.html  # Feature importance plot
├── roc_curve.html          # ROC curve plot
└── README.md               # This file
```

## Troubleshooting

1. **Model not found**: Run `python titanic_ml_model.py` to train the model first
2. **BigQuery connection**: Ensure GCP credentials are configured
3. **Missing data**: Verify dbt models have been run and tables exist
4. **API errors**: Check the console logs for detailed error messages
5. **Performance issues**: Consider reducing hyperparameter search space

## Model Retraining

To retrain the model with new data:

1. Update the data in BigQuery
2. Run the training script: `python titanic_ml_model.py`
3. Restart the API server to load the new model

## Next Steps

- Add model versioning
- Implement A/B testing for model comparison
- Add monitoring and logging
- Create automated retraining pipeline
- Add model explainability features 