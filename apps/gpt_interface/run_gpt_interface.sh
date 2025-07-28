#!/bin/bash

# Titanic GPT Interface Launcher
# Runs the Streamlit app for natural language queries

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚢 Starting Titanic GPT Interface${NC}"
echo "======================================"

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
if ! python -c "import streamlit, openai, google.cloud.bigquery" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing additional dependencies...${NC}"
    pip install openai
fi

# Kill any existing Streamlit processes on port 8503
echo -e "${YELLOW}🧹 Cleaning up existing processes...${NC}"
lsof -ti:8503 | xargs kill -9 2>/dev/null || true

# Start the Streamlit app
echo -e "${GREEN}🚀 Starting GPT Interface...${NC}"
echo -e "${BLUE}📍 URL: http://localhost:8503${NC}"
echo -e "${YELLOW}🔑 Note: You'll need an OpenAI API key to use this interface${NC}"
echo -e "${YELLOW}💡 Tip: You can get an API key from https://platform.openai.com/api-keys${NC}"
echo ""
echo -e "${GREEN}✅ GPT Interface is starting...${NC}"

# Navigate to the app directory and run
cd "$(dirname "$0")"
streamlit run titanic_gpt.py --server.port 8503 --server.address localhost --browser.gatherUsageStats false 