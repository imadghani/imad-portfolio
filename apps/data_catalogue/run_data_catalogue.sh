#!/bin/bash

# Data Catalogue Launcher
# Runs the Streamlit app for data discovery and documentation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📚 Starting Data Catalogue${NC}"
echo "============================"

# Project paths
PROJECT_ROOT="/Users/imadghani/GitHub/imad-portfolio"
VENV_PATH="$PROJECT_ROOT/python_venv"

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if required packages are installed
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
if ! python -c "import streamlit, networkx, plotly" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing additional dependencies...${NC}"
    pip install networkx
fi

# Kill any existing Streamlit processes on port 8504
echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
lsof -ti:8504 | xargs kill -9 2>/dev/null || true

# Start the Streamlit app
echo -e "${GREEN}🚀 Starting Data Catalogue...${NC}"
echo -e "${BLUE}📍 URL: http://localhost:8504${NC}"
echo -e "${YELLOW}📚 Explore all tables, columns, tests, and lineage${NC}"
echo -e "${YELLOW}🔍 Search functionality and interactive data discovery${NC}"
echo ""
echo -e "${GREEN}✅ Data Catalogue is starting...${NC}"

# Navigate to the app directory and run
cd "$(dirname "$0")"
streamlit run data_catalogue.py --server.port 8504 --server.address localhost --browser.gatherUsageStats false 