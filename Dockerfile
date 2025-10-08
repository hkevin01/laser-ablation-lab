# Multi-stage Docker build for laser ablation lab
# Stage 1: Development environment with full toolchain
FROM python:3.11-slim as development

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libhdf5-dev \
    libnetcdf-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ablation
USER ablation
WORKDIR /home/ablation

# Create virtual environment
RUN python -m venv /home/ablation/venv
ENV PATH="/home/ablation/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY --chown=ablation:ablation requirements*.txt ./
RUN pip install -r requirements.txt
RUN if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; fi

# Copy project files
COPY --chown=ablation:ablation . .

# Install the package in development mode
RUN pip install -e .

# Set up Jupyter configuration
RUN mkdir -p /home/ablation/.jupyter
RUN echo "c.NotebookApp.ip = '0.0.0.0'" > /home/ablation/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /home/ablation/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /home/ablation/.jupyter/jupyter_notebook_config.py

# Expose ports for Jupyter and documentation
EXPOSE 8888 8000

# Default command
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888"]

# Stage 2: Production environment (minimal)
FROM python:3.11-slim as production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libnetcdf19 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash ablation
USER ablation
WORKDIR /home/ablation

# Create virtual environment
RUN python -m venv /home/ablation/venv
ENV PATH="/home/ablation/venv/bin:$PATH"

# Copy only production requirements and install
COPY --chown=ablation:ablation requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only the source code
COPY --chown=ablation:ablation src/ ./src/
COPY --chown=ablation:ablation pyproject.toml ./
COPY --chown=ablation:ablation README.md ./

# Install the package
RUN pip install .

# Default command for production
CMD ["python", "-c", "import ablab; print('Laser Ablation Lab installed successfully')"]
