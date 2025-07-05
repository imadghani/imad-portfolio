#!/bin/bash

# Titanic ML Model Runner Script
# Usage: ./run_ml_model.sh [train|api|both]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}🤖 Titanic ML Model - Data Engineering Portfolio${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Function to check if virtual environment is activated
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not detected."
        if [[ -d "../../python_venv" ]]; then
            print_status "Activating python_venv..."
            source ../../python_venv/bin/activate
        else
            print_error "Virtual environment not found. Please run from project root: python -m venv python_venv && source python_venv/bin/activate"
            exit 1
        fi
    else
        print_status "Virtual environment active: $VIRTUAL_ENV"
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing ML model dependencies..."
    pip install -q setuptools wheel
    pip install -q flask==3.0.0 scikit-learn==1.5.0 pandas==2.1.4 numpy==1.26.0
    pip install -q google-cloud-bigquery==3.14.1 joblib==1.3.2
    print_status "Dependencies installed successfully!"
}

# Function to train the model
train_model() {
    print_header
    print_status "🔧 Training Titanic ML Model..."
    echo ""
    
    check_venv
    install_dependencies
    
    print_status "Starting model training..."
    python titanic_ml_model.py
    
    if [[ $? -eq 0 ]]; then
        print_status "✅ Model training completed successfully!"
        print_status "📊 Model saved to: titanic_model.pkl"
        print_status "📈 Performance plots saved to current directory"
    else
        print_error "❌ Model training failed!"
        exit 1
    fi
}

# Function to start the prediction API
start_api() {
    print_header
    print_status "🚀 Starting Titanic Prediction API..."
    echo ""
    
    check_venv
    install_dependencies
    
    # Check if model exists
    if [[ ! -f "titanic_model.pkl" ]]; then
        print_warning "Model not found. Training model first..."
        python titanic_ml_model.py
    fi
    
    print_status "Starting Flask API server..."
    print_status "API will be available at: http://localhost:5001"
    print_status ""
    print_status "Available endpoints:"
    print_status "  • POST /predict - Single prediction"
    print_status "  • GET /model/info - Model information"
    print_status "  • GET /health - Health check"
    print_status "  • GET /predict/example - Example data"
    print_status ""
    print_status "Press Ctrl+C to stop the server"
    echo ""
    
    python simple_api.py
}

# Function to run both training and API
run_both() {
    print_header
    print_status "🎯 Running Full ML Pipeline (Train + Serve)..."
    echo ""
    
    # Train the model
    train_model
    
    echo ""
    print_status "⏳ Waiting 3 seconds before starting API..."
    sleep 3
    
    # Start the API
    start_api
}

# Function to show usage
show_usage() {
    print_header
    echo ""
    echo "Usage: ./run_ml_model.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  train    - Train the Titanic ML model"
    echo "  api      - Start the prediction API server"
    echo "  both     - Train model then start API (default)"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_ml_model.sh train    # Just train the model"
    echo "  ./run_ml_model.sh api      # Just start the API"
    echo "  ./run_ml_model.sh both     # Train then serve"
    echo "  ./run_ml_model.sh          # Same as 'both'"
    echo ""
    echo "API Testing:"
    echo "  curl -X POST http://localhost:5001/predict \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"Pclass\": 1, \"Sex\": \"female\", \"Age\": 25, \"SibSp\": 0, \"Parch\": 0, \"Fare\": 50, \"Embarked\": \"S\"}'"
    echo ""
}

# Main execution
case "${1:-both}" in
    "train")
        train_model
        ;;
    "api")
        start_api
        ;;
    "both")
        run_both
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 