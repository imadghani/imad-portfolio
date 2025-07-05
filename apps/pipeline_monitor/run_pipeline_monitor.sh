#!/bin/bash

# Pipeline Monitor Dashboard Runner Script
# Usage: ./run_pipeline_monitor.sh [port]

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
    echo -e "${BLUE}🔧 Pipeline Monitor - Data Engineering Portfolio${NC}"
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
    print_status "Installing pipeline monitor dependencies..."
    pip install -q streamlit==1.41.0 pandas==2.1.4 numpy==1.24.3
    print_status "Dependencies installed successfully!"
}

# Function to check Airflow database
check_airflow_db() {
    local airflow_db="../../airflow/airflow.db"
    if [[ ! -f "$airflow_db" ]]; then
        print_warning "Airflow database not found at $airflow_db"
        print_warning "Make sure Airflow has been initialized and run at least once"
        print_warning "You can start Airflow with: cd airflow && ./start_airflow.sh"
        return 1
    else
        print_status "Airflow database found: $airflow_db"
        return 0
    fi
}

# Function to kill existing streamlit processes
kill_existing_streamlit() {
    print_status "Checking for existing Streamlit processes..."
    pkill -f "streamlit run pipeline_dashboard.py" 2>/dev/null || true
    sleep 2
}

# Function to start the pipeline monitor
start_pipeline_monitor() {
    local port=${1:-8502}
    
    print_header
    print_status "🚀 Starting Pipeline Monitor Dashboard..."
    echo ""
    
    check_venv
    install_dependencies
    
    # Check if Airflow database exists
    if ! check_airflow_db; then
        print_error "Cannot start pipeline monitor without Airflow database"
        exit 1
    fi
    
    kill_existing_streamlit
    
    print_status "Starting Pipeline Monitor on port $port..."
    print_status "Dashboard will be available at: http://localhost:$port"
    print_status ""
    print_status "Monitor features:"
    print_status "  • 📊 Real-time DAG run monitoring"
    print_status "  • 📈 Task success/failure rates"
    print_status "  • ⏱️  Task duration analysis"
    print_status "  • 📋 Task instance details"
    print_status "  • 🔍 Log analysis and filtering"
    print_status ""
    print_status "Press Ctrl+C to stop the monitor"
    echo ""
    
    streamlit run pipeline_dashboard.py --server.port $port --server.headless true
}

# Function to show usage
show_usage() {
    print_header
    echo ""
    echo "Usage: ./run_pipeline_monitor.sh [PORT]"
    echo ""
    echo "Arguments:"
    echo "  PORT     - Port number (default: 8502)"
    echo ""
    echo "Examples:"
    echo "  ./run_pipeline_monitor.sh         # Start on default port 8502"
    echo "  ./run_pipeline_monitor.sh 8503    # Start on port 8503"
    echo ""
    echo "Prerequisites:"
    echo "  • Airflow must be initialized and have run at least once"
    echo "  • Airflow database should exist at: ../../airflow/airflow.db"
    echo ""
    echo "Monitor Features:"
    echo "  • Real-time pipeline monitoring"
    echo "  • DAG run success/failure tracking"
    echo "  • Task performance analysis"
    echo "  • Historical pipeline trends"
    echo "  • Task log analysis"
    echo ""
}

# Main execution
case "${1:-start}" in
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        # Check if argument is a number (port)
        if [[ $1 =~ ^[0-9]+$ ]]; then
            start_pipeline_monitor $1
        else
            start_pipeline_monitor
        fi
        ;;
esac 