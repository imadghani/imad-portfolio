#!/bin/bash
# Source this script to set up the Airflow environment
# Usage: source setup_env.sh

# Project paths
PROJECT_ROOT="/Users/imadghani/GitHub/imad-portfolio"
AIRFLOW_HOME="$PROJECT_ROOT/airflow"
VENV_PATH="$PROJECT_ROOT/python_venv"

# Set environment variables
export AIRFLOW_HOME="$AIRFLOW_HOME"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

echo "✅ Environment setup complete!"
echo "🏠 AIRFLOW_HOME: $AIRFLOW_HOME"
echo "🐍 Virtual environment activated"
echo "📁 Python path: $PYTHONPATH"
echo ""
echo "You can now run Airflow commands like:"
echo "  airflow dags list"
echo "  airflow standalone"
echo "  airflow webserver --port 8080" 