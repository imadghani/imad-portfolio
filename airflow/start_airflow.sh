#!/bin/bash

# Airflow startup script for local development
# This script sets up the environment and starts Airflow services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="/Users/imadghani/GitHub/imad-portfolio"
AIRFLOW_HOME="$PROJECT_ROOT/airflow"
VENV_PATH="$PROJECT_ROOT/python_venv"

echo -e "${BLUE}🚀 Starting Airflow Local Development Environment${NC}"
echo "========================================================"

# Set environment variables
export AIRFLOW_HOME="$AIRFLOW_HOME"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Ensure AIRFLOW_HOME is set correctly for all commands
export AIRFLOW_HOME="$AIRFLOW_HOME"
echo -e "${BLUE}🏠 AIRFLOW_HOME set to: $AIRFLOW_HOME${NC}"

# Check if database is initialized
if [ ! -f "$AIRFLOW_HOME/airflow.db" ]; then
    echo -e "${YELLOW}🔧 Initializing Airflow database...${NC}"
    airflow db init
fi

# For Airflow 2.10.5, use standalone mode for development
# This automatically creates admin user and handles auth

echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""
echo "Available commands:"
echo "  1. Start standalone (recommended): airflow standalone"
echo "  2. Start webserver: airflow webserver --port 8080"
echo "  3. Start scheduler: airflow scheduler"
echo "  4. List DAGs: airflow dags list"
echo "  5. Test DAG: airflow dags test titanic_dbt_pipeline"
echo ""
echo -e "${BLUE}🌐 Airflow UI will be available at: http://localhost:8080${NC}"
echo -e "${BLUE}🔐 Login credentials: admin / admin${NC}"
echo ""

# Function to start services
start_services() {
    echo -e "${YELLOW}🚀 Starting Airflow services...${NC}"
    
    # Start scheduler in background
    echo -e "${YELLOW}📅 Starting scheduler...${NC}"
    airflow scheduler &
    SCHEDULER_PID=$!
    
    # Start webserver in background
    echo -e "${YELLOW}🌐 Starting webserver...${NC}"
    airflow webserver --port 8080 &
    WEBSERVER_PID=$!
    
    # Function to handle cleanup on exit
    cleanup() {
        echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
        kill $SCHEDULER_PID $WEBSERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✅ Services stopped${NC}"
        exit 0
    }
    
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    echo -e "${GREEN}✅ Services started successfully!${NC}"
    echo -e "${BLUE}🌐 Airflow UI: http://localhost:8080${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
    
    # Wait for services
    wait
}

# Function to start standalone mode (recommended for development)
start_standalone() {
    echo -e "${YELLOW}🚀 Starting Airflow in standalone mode...${NC}"
    echo -e "${YELLOW}📝 This automatically creates admin user and handles configuration${NC}"
    echo -e "${BLUE}🌐 Airflow UI will be available at: http://localhost:8080${NC}"
    echo -e "${BLUE}🔐 Login credentials will be displayed below${NC}"
    echo ""
    airflow standalone
}

# Check command line arguments
if [ "$1" = "start" ]; then
    start_services
elif [ "$1" = "standalone" ]; then
    start_standalone
elif [ "$1" = "webserver" ]; then
    echo -e "${YELLOW}🌐 Starting webserver only...${NC}"
    airflow webserver --port 8080
elif [ "$1" = "scheduler" ]; then
    echo -e "${YELLOW}📅 Starting scheduler only...${NC}"
    airflow scheduler
elif [ "$1" = "test" ]; then
    echo -e "${YELLOW}🧪 Testing DAG...${NC}"
    airflow dags test titanic_dbt_pipeline
else
    echo "Usage: $0 [standalone|start|webserver|scheduler|test]"
    echo ""
    echo "Commands:"
    echo "  standalone - Start Airflow in standalone mode (recommended for development)"
    echo "  start      - Start both webserver and scheduler"
    echo "  webserver  - Start webserver only"
    echo "  scheduler  - Start scheduler only"
    echo "  test       - Test the DAG"
    echo ""
    echo "For development, use: ./start_airflow.sh standalone"
    echo "Or run individual commands manually:"
    echo "  airflow standalone"
    echo "  airflow webserver --port 8080"
    echo "  airflow scheduler"
fi 