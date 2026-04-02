#!/bin/bash
# Description: Smart setup script for CLARITY. Adapts to Local vs Colab environments.
# Uses 'uv' for extremely fast package management.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

echo "=== System Initialization: CLARITY Environment (UV Version) ==="

if [ ! -f ".env" ]; then
    echo "[INFO] .env file not found. Creating a default one..."
    echo "HF_TOKEN=\"\"" > .env
else
    echo "[INFO] .env file found."
fi

ensure_uv() {
    if ! command -v uv &> /dev/null; then
        echo "[INFO] 'uv' non trovato. Installazione tramite script ufficiale in corso..."
        # Utilizzo dello script ufficiale di Astral per aggirare il blocco PEP 668 di Ubuntu
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
}

# ==========================================
# GESTIONE AMBIENTE: Colab vs Locale
# ==========================================

if [ -d "/content" ] && [[ "$PWD" == *"/content"* ]]; then
    echo "[INFO] Ambiente Google Colab rilevato."
    ensure_uv

    if [ -f "requirements.txt" ]; then
        REQ_HASH=$(md5sum requirements.txt | awk '{print $1}')
        if [ -f ".requirements_hash" ] && [ "$REQ_HASH" == "$(cat .requirements_hash)" ]; then
            echo "[INFO] Dipendenze già aggiornate."
        else
            echo "[INFO] Installazione dipendenze tramite uv in corso..."
            # In Colab forziamo --system perché usiamo il container root
            uv pip install --system --index-strategy unsafe-best-match -r requirements.txt
            echo "$REQ_HASH" > .requirements_hash
        fi
    fi
else
    echo "[INFO] Ambiente locale rilevato."
    ensure_uv
    export PATH="$HOME/.cargo/bin:$PATH"

    VENV_DIR=".venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo ">> 1. Creating virtual environment using uv in $VENV_DIR..."
        uv venv "$VENV_DIR"
    else
        echo ">> 1. Virtual environment $VENV_DIR already exists."
    fi

    echo "[INFO] Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo ">> 2. Installing dependencies using uv from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        REQ_HASH=$(md5sum requirements.txt | awk '{print $1}')
        if [ -f "$VENV_DIR/.requirements_hash" ] && [ "$REQ_HASH" == "$(cat "$VENV_DIR/.requirements_hash")" ]; then
            echo "[INFO] Dependencies are practically up to date."
        else
            echo "[INFO] Installing dependencies via uv..."
            # In locale NON usiamo --system, ma installiamo nel .venv
            uv pip install --index-strategy unsafe-best-match -r requirements.txt
            echo "$REQ_HASH" > "$VENV_DIR/.requirements_hash"
        fi
    fi
fi

# ==========================================
# DOWNLOAD DATASET
# ==========================================

echo ">> 3. Checking and downloading datasets..."
if [ -f "scripts/download_dataset.py" ]; then
    python scripts/download_dataset.py
fi

echo "=== Setup Complete ==="