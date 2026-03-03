#!/bin/bash
# Description: Setup script to initialize and open the devcontainer, 
# and ensure the dataset is properly downloaded.

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

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] Docker is not running or you do not have permission to access it."
    echo "Please start Docker and try running this script again."
    exit 1
fi

echo "[INFO] Docker is running."

# Check if devcontainer CLI is accessible
if ! command -v devcontainer &> /dev/null; then
    echo "[WARNING] 'devcontainer' CLI not found on host."
    echo "You can install it with: npm install -g @devcontainers/cli"
    echo "Or just open VS Code in this directory and let it rebuild the container."
    
    # Fallback to local python dataset downloader
    echo "[INFO] Attempting to download dataset and models locally as fallback..."
    if command -v python3 &> /dev/null || command -v python &> /dev/null; then
        PYTHON_CMD=$(command -v python3 || command -v python)
        
        # Check if datasets is installed, if not, create a temporary venv
        if ! $PYTHON_CMD -c "import datasets; import huggingface_hub" &> /dev/null; then
            echo "[INFO] Required packages not found. Creating a temporary virtual environment..."
            $PYTHON_CMD -m venv .tmp_venv
            source .tmp_venv/bin/activate
            # Upgrade pip and install datasets
            pip install --upgrade pip > /dev/null 2>&1
            pip install datasets huggingface_hub > /dev/null 2>&1
            python scripts/download_dataset.py
            deactivate
            rm -rf .tmp_venv
        else
            $PYTHON_CMD scripts/download_dataset.py
        fi
    else
        echo "[ERROR] Python not found. Could not download dataset locally."
    fi
    exit 0
fi

echo ">> 1. Building and starting the development container..."
echo "This might take a while if the container is not yet built."
devcontainer up --workspace-folder .

echo ">> 2. Checking and downloading datasets inside the container..."
# Run dataset download script natively inside the devcontainer using its Python environment
devcontainer exec --workspace-folder . python scripts/download_dataset.py

echo "=== Setup Complete ==="
echo "You can now connect to the container using VS Code! (e.g. 'code .')"
