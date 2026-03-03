# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# SemEval-2026 CLARITY Task Environment
# Base: NVIDIA CUDA 12.6.3 Devel on Ubuntu 24.04 LTS
# Target: PyTorch 2.9.1, NumPy 2.3.5
# -----------------------------------------------------------------------------

FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

# -----------------------------------------------------------------------------
# 1. Environment Configuration
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# -----------------------------------------------------------------------------
# 2. System Dependency Installation
# -----------------------------------------------------------------------------
# Install Python 3.12 (standard in Ubuntu 24.04) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    libsndfile1 \
    pkg-config \
    software-properties-common \
    unzip \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 3. User Setup (DevContainer Standard)
# -----------------------------------------------------------------------------
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN userdel -r ubuntu || true \
    && groupdel ubuntu || true \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Grant sudo privileges for flexibility during development
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# -----------------------------------------------------------------------------
# 4. Python Environment Construction
# -----------------------------------------------------------------------------
USER $USERNAME
WORKDIR /home/$USERNAME

# Create Virtual Environment
ENV VIRTUAL_ENV=/home/$USERNAME/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip to handle modern wheels and pyproject.toml builds
RUN pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# 5. Application Dependency Installation
# -----------------------------------------------------------------------------
# Copy requirements first to leverage Docker layer caching
COPY --chown=$USERNAME:$USERNAME requirements.txt .

# Install dependencies
# Note: We rely on the index-url in requirements.txt or pip's default resolution
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# 6. Runtime Metadata
# -----------------------------------------------------------------------------
ENV SHELL=/bin/bash
CMD ["/bin/bash"]