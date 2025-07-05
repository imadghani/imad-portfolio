#!/bin/bash

# Titanic Dashboard Runner Script
# Usage: ./run_dashboard.sh [port]

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
    echo -e "${BLUE}📊 Titanic Dashboard - Data Engineering Portfolio${NC}"
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
    print_status "Installing dashboard dependencies..."
    pip install -q streamlit==1.41.0 plotly==5.24.1 seaborn==0.13.2 matplotlib==3.10.3
    pip install -q google-cloud-bigquery==3.14.1 pandas==2.1.4 numpy==1.24.3
    print_status "Dependencies installed successfully!"
}

# Function to kill existing streamlit processes
kill_existing_streamlit() {
    print_status "Checking for existing Streamlit processes..."
    pkill -f "streamlit run titanic_dashboard.py" 2>/dev/null || true
    sleep 2
}

# Function to start the dashboard
start_dashboard() {
    local port=${1:-8501}
    
    print_header
    print_status "🚀 Starting Titanic Data Dashboard..."
    echo ""
    
    check_venv
    install_dependencies
    kill_existing_streamlit
    
    print_status "Starting Streamlit dashboard on port $port..."
    print_status "Dashboard will be available at: http://localhost:$port"
    print_status ""
    print_status "Dashboard features:"
    print_status "  • 📊 Interactive survival analysis charts"
    print_status "  • 🔍 Dynamic passenger demographic filters"
    print_status "  • 📈 Fare analysis and geographic insights"
    print_status "  • 📥 Data export functionality"
    print_status ""
    print_status "Press Ctrl+C to stop the dashboard"
    echo ""
    
    streamlit run titanic_dashboard.py --server.port $port --server.headless true
}

# Function to show usage
show_usage() {
    print_header
    echo ""
    echo "Usage: ./run_dashboard.sh [PORT]"
    echo ""
    echo "Arguments:"
    echo "  PORT     - Port number (default: 8501)"
    echo ""
    echo "Examples:"
    echo "  ./run_dashboard.sh         # Start on default port 8501"
    echo "  ./run_dashboard.sh 8502    # Start on port 8502"
    echo ""
    echo "Dashboard Features:"
    echo "  • Interactive survival analysis"
    echo "  • Passenger demographics breakdown"
    echo "  • Fare analysis and insights"
    echo "  • Geographic embarkation analysis"
    echo "  • Data filtering and export"
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
            start_dashboard $1
        else
            start_dashboard
        fi
        ;;
esac 