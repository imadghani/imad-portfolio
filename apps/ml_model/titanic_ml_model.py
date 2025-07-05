import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
from google.cloud import bigquery
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class TitanicSurvivalPredictor:
    """
    A comprehensive machine learning model for predicting Titanic passenger survival
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        self.model_metrics = {}
        
        # Set up BigQuery client with proper credentials path
        # Get project root (go up 3 levels: ml_model -> apps -> imad-portfolio)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        credentials_path = os.path.join(project_root, "secrets", "bigquery-service-account.json")
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        self.bigquery_client = bigquery.Client(project=os.getenv("GCP_PROJECT"))
        
    def load_data_from_bigquery(self):
        """Load training data from BigQuery"""
        print("📊 Loading data from BigQuery...")
        
        query = """
        SELECT 
            f.survived_flag,
            f.fare_amount,
            f.age,
            f.family_size,
            f.is_alone_flag,
            dp.gender,
            dp.age_group,
            dp.title,
            dpc.class_name,
            dpc.class_number,
            de.port_name,
            de.port_code,
            dc.deck_name,
            dc.has_cabin,
            dt.ticket_type,
            dt.is_group_ticket,
            dt.passengers_on_ticket
        FROM `{project}.prototype_data.fact_passenger_journey` f
        LEFT JOIN `{project}.prototype_data.dim_passenger` dp ON f.passenger_key = dp.passenger_key
        LEFT JOIN `{project}.prototype_data.dim_passenger_class` dpc ON f.passenger_class_key = dpc.passenger_class_key
        LEFT JOIN `{project}.prototype_data.dim_embarkation` de ON f.embarkation_key = de.embarkation_key
        LEFT JOIN `{project}.prototype_data.dim_cabin` dc ON f.cabin_key = dc.cabin_key
        LEFT JOIN `{project}.prototype_data.dim_ticket` dt ON f.ticket_key = dt.ticket_key
        WHERE dpc.class_name IS NOT NULL
        """.format(project=os.getenv("GCP_PROJECT"))
        
        # Use standard BigQuery API to avoid storage permissions issue
        query_job = self.bigquery_client.query(query)
        results = query_job.result()
        
        # Convert to pandas DataFrame manually
        rows = [dict(row) for row in results]
        df = pd.DataFrame(rows)
        print(f"✅ Loaded {len(df)} records from BigQuery")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        print("🔧 Preprocessing data...")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Handle missing values
        data['age'] = data['age'].fillna(data['age'].median())
        data['fare_amount'] = data['fare_amount'].fillna(data['fare_amount'].median())
        
        # Create derived features
        data['age_fare_ratio'] = data['age'] / (data['fare_amount'] + 1)
        data['fare_per_person'] = data['fare_amount'] / data['passengers_on_ticket']
        
        # Encode categorical variables
        categorical_columns = ['gender', 'age_group', 'title', 'class_name', 'port_name', 'deck_name', 'ticket_type']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                data[col + '_encoded'] = self.encoders[col].fit_transform(data[col].astype(str))
            else:
                data[col + '_encoded'] = self.encoders[col].transform(data[col].astype(str))
        
        # Select features for modeling
        feature_columns = [
            'fare_amount', 'age', 'family_size', 'is_alone_flag', 'class_number',
            'has_cabin', 'is_group_ticket', 'passengers_on_ticket', 'age_fare_ratio', 'fare_per_person'
        ] + [col + '_encoded' for col in categorical_columns]
        
        X = data[feature_columns]
        y = data['survived_flag']
        
        # Store feature names for later use
        self.feature_names = feature_columns
        
        print(f"✅ Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        print("🤖 Training machine learning models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled features for Logistic Regression and SVM
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Store model and metrics
            self.models[name] = model
            self.model_metrics[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  ✅ {name}: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}")
        
        # Select best model
        best_auc = max(self.model_metrics.values(), key=lambda x: x['auc_score'])['auc_score']
        self.best_model_name = [name for name, metrics in self.model_metrics.items() if metrics['auc_score'] == best_auc][0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"🏆 Best model: {self.best_model_name} (AUC = {best_auc:.4f})")
        
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning on the best model"""
        print(f"🔍 Performing hyperparameter tuning for {self.best_model_name}...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Use scaled features for certain models
            if self.best_model_name in ['Logistic Regression', 'SVM']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                grid_search = GridSearchCV(self.best_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
            else:
                grid_search = GridSearchCV(self.best_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
                grid_search.fit(X_train, y_train)
            
            self.best_model = grid_search.best_estimator_
            self.models[self.best_model_name] = self.best_model
            
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV AUC: {grid_search.best_score_:.4f}")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model and create visualizations"""
        print("📈 Evaluating model performance...")
        
        # Make predictions
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.best_model.predict(X_test_scaled)
            y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = self.best_model.predict(X_test)
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"🎯 Final Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC Score: {auc_score:.4f}")
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔍 Confusion Matrix:")
        print(cm)
        
        return y_pred, y_pred_proba
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Create plotly figure
            fig = px.bar(
                importance_df.head(15), 
                x='importance', 
                y='feature',
                title=f'Top 15 Feature Importances - {self.best_model_name}',
                labels={'importance': 'Importance', 'feature': 'Features'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            return fig
        else:
            print("Feature importance not available for this model type")
            return None
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{self.best_model_name} (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        return fig
    
    def save_model(self, filepath='titanic_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'metrics': self.model_metrics[self.best_model_name],
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath='titanic_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.encoders = model_data['encoders']
        self.feature_names = model_data['feature_names']
        print(f"📥 Model loaded from {filepath}")
        return model_data
    
    def predict_survival(self, passenger_data):
        """Predict survival for new passenger data"""
        if self.best_model is None:
            raise ValueError("No model trained. Please train a model first.")
        
        # Preprocess the input data
        processed_data = self.preprocess_single_prediction(passenger_data)
        
        # Make prediction
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            processed_data_scaled = self.scaler.transform([processed_data])
            prediction = self.best_model.predict(processed_data_scaled)[0]
            probability = self.best_model.predict_proba(processed_data_scaled)[0, 1]
        else:
            prediction = self.best_model.predict([processed_data])[0]
            probability = self.best_model.predict_proba([processed_data])[0, 1]
        
        return {
            'survived': bool(prediction),
            'survival_probability': float(probability),
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium' if abs(probability - 0.5) > 0.1 else 'Low'
        }
    
    def preprocess_single_prediction(self, passenger_data):
        """Preprocess single passenger data for prediction"""
        # This would need to be implemented based on the specific input format
        # For now, assuming the data comes in the same format as training
        pass
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 Starting Titanic Survival Prediction Pipeline")
        print("=" * 50)
        
        # Load data
        df = self.load_data_from_bigquery()
        
        # Preprocess
        X, y = self.preprocess_data(df)
        
        # Train models
        X_train, X_test, y_train, y_test = self.train_models(X, y)
        
        # Hyperparameter tuning
        self.hyperparameter_tuning(X_train, y_train)
        
        # Final evaluation
        y_pred, y_pred_proba = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model('titanic_model.pkl')
        
        print("\n🎉 Pipeline completed successfully!")
        
        return {
            'feature_importance_plot': self.plot_feature_importance(),
            'roc_curve_plot': self.plot_roc_curve(y_test, y_pred_proba),
            'model_name': self.best_model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba)
        }

def main():
    """Main function to run the ML pipeline"""
    predictor = TitanicSurvivalPredictor()
    results = predictor.run_complete_pipeline()
    
    print(f"\n📊 Final Results:")
    print(f"   Best Model: {results['model_name']}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   AUC Score: {results['auc_score']:.4f}")
    
    # Save plots
    if results['feature_importance_plot']:
        results['feature_importance_plot'].write_html('feature_importance.html')
        print("📊 Feature importance plot saved to feature_importance.html")
    
    results['roc_curve_plot'].write_html('roc_curve.html')
    print("📊 ROC curve plot saved to roc_curve.html")

if __name__ == "__main__":
    main() 