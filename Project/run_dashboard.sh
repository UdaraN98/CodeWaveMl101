#!/bin/bash

# SHAP & LIME Streamlit Dashboard Launcher
# This script launches the Streamlit dashboard for SHAP and LIME visualizations

echo "🔍 Starting SHAP & LIME Explainability Dashboard..."
echo "==============================================="

# Navigate to the Project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found. Please set up the Python environment first."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking required packages..."
../.venv/bin/python -c "import streamlit, shap, lime, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some packages might be missing. Installing required packages..."
    ../.venv/bin/pip install streamlit plotly shap lime
fi

# Launch the Streamlit app
echo "🚀 Launching dashboard..."
echo "📍 The app will be available at: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the application"
echo "==============================================="

../.venv/bin/python -m streamlit run streamlit_app.py --server.port 8501

echo "👋 Dashboard stopped. Thank you for using the SHAP & LIME Explainability Dashboard!"