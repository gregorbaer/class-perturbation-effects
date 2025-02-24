# Use Ubuntu as base image for better compatibility
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="/root/.local/bin:${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
ENV PYTHON_VERSION=3.10.13

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Install Python with pyenv
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}

# Install UV
RUN curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh && \
    which uv  # Verify UV is in PATH

# Create working directory
WORKDIR /app

# Copy local files
COPY . .

# Create and activate virtual environment, then attempt installation
SHELL ["/bin/bash", "-c"]
RUN which uv && \
    uv venv && \
    . .venv/bin/activate && \
    if [ -f requirements.txt ]; then \
        echo "Attempting installation from requirements.txt..." && \
        (uv pip install -r requirements.txt || \
        (echo "requirements.txt installation failed, falling back to direct installation..." && \
        # install problem packages first
        uv pip install TSInterpret && \
        uv pip install tsai)) && \
        # install local package
        uv pip install -e . || \
        (echo "Direct installation failed as well. Check package compatibility." && exit 1); \
    else \
        echo "No requirements.txt found, using direct installation..." && \
        # install problem packages first
        uv pip install TSInterpret && \
        uv pip install tsai && \
        # install local package
        uv pip install -e .; \
    fi && \
    # Verify package installation
    python -c "import tsxai" && \
    echo "✅ Package installation successful: tsxai package imported successfully" || \
    (echo "❌ Package installation failed: could not import tsxai" && exit 1)

# Set up entrypoint to activate virtual environment
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["python"]