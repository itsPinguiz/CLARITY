#!/bin/bash
# Description: Smart setup script for CLARITY. Adapts to Local vs Colab environments.

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

# ==========================================
# GESTIONE AMBIENTE: Colab vs Locale
# ==========================================

# Rileva se ci troviamo dentro Google Colab controllando la directory /content
if [ -d "/content" ] && [[ "$PWD" == *"/content"* ]]; then
    echo "[INFO] Ambiente Google Colab rilevato."
    echo "[INFO] Salto la creazione del venv (non necessario in Colab)."
    
    echo ">> Installazione delle dipendenze nell'ambiente globale di Colab..."
    if [ -f "requirements.txt" ]; then
        REQ_HASH=$(md5sum requirements.txt | awk '{print $1}')
        if [ -f ".requirements_hash" ] && [ "$REQ_HASH" == "$(cat .requirements_hash)" ]; then
            echo "[INFO] Dipendenze già aggiornate rispetto a requirements.txt."
        else
            echo "[INFO] Installazione/aggiornamento dipendenze in corso..."
            pip install -r requirements.txt
            echo "$REQ_HASH" > .requirements_hash
        fi
    else
        echo "[WARNING] requirements.txt non trovato."
    fi

else
    # Comportamento normale per PC Locali
    echo "[INFO] Ambiente locale rilevato."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    echo "[INFO] Using Python: $($PYTHON_CMD --version)"

    VENV_DIR=".venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo ">> 1. Creating virtual environment in $VENV_DIR..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    else
        echo ">> 1. Virtual environment $VENV_DIR already exists."
    fi

    echo "[INFO] Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo ">> 2. Installing dependencies from requirements.txt..."
    pip install --upgrade pip > /dev/null 2>&1
    if [ -f "requirements.txt" ]; then
        REQ_HASH=$(md5sum requirements.txt | awk '{print $1}')
        if [ -f "$VENV_DIR/.requirements_hash" ] && [ "$REQ_HASH" == "$(cat "$VENV_DIR/.requirements_hash")" ]; then
            echo "[INFO] Dependencies are practically up to date."
        else
            echo "[INFO] Requirements changed or first run. Installing dependencies..."
            pip install -r requirements.txt
            echo "$REQ_HASH" > "$VENV_DIR/.requirements_hash"
        fi
    fi
fi