#!/bin/bash
# Description: Setup script to initialize the project environment using a virtual environment,
# install dependencies, and ensure the dataset is properly downloaded.

set -e

# Change directory to the workspace root
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

echo "=== System Initialization: CLARITY Environment ==="

# Check for .env file
if [ ! -f ".env" ]; then
    echo "[INFO] .env file not found. Creating a default one..."
    echo "HF_TOKEN=\"\"" > .env
    echo "Please add your Hugging Face token to the .env file if required."
else
    echo "[INFO] .env file found."
fi

# Locate Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python not found. Please install Python 3."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# Create virtual environment if it doesn't exist
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ">> 1. Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo ">> 1. Virtual environment $VENV_DIR already exists."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install dependencies
echo ">> 2. Installing dependencies from requirements.txt..."
pip install --upgrade pip > /dev/null 2>&1
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found. Skipping dependency installation."
fi

# Download dataset
echo ">> 3. Checking and downloading datasets..."
if [ -f "scripts/download_dataset.py" ]; then
    python scripts/download_dataset.py
else
    echo "[WARNING] scripts/download_dataset.py not found. Skipping dataset download."
fi

echo "=== Setup Complete ==="
echo "You can now activate the virtual environment using:"
echo "source $VENV_DIR/bin/activate"
