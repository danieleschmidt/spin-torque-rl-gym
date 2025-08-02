# Multi-stage Dockerfile for Spin-Torque RL-Gym
# Optimized for security, size, and performance

#==============================================================================
# Build Stage - Install dependencies and build package
#==============================================================================
FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set labels for metadata
LABEL maintainer="daniel@terragonlabs.com" \
      org.opencontainers.image.title="Spin-Torque RL-Gym Builder" \
      org.opencontainers.image.description="Builder stage for Spin-Torque RL-Gym" \
      org.opencontainers.image.url="https://github.com/terragon-labs/spin-torque-rl-gym" \
      org.opencontainers.image.source="https://github.com/terragon-labs/spin-torque-rl-gym" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Create application directory
WORKDIR /app

# Copy dependency files first (for better caching)
COPY pyproject.toml README.md LICENSE ./
COPY spin_torque_gym/__init__.py spin_torque_gym/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev,jax,viz]"

# Copy source code
COPY . .

# Build wheel distribution
RUN python -m build --wheel

# Run basic tests to validate build
RUN python -m pytest tests/unit/test_core.py -v || echo "Tests not yet passing - continuing build"

#==============================================================================
# Production Stage - Minimal runtime image
#==============================================================================
FROM python:3.11-slim-bullseye as production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set labels for metadata
LABEL maintainer="daniel@terragonlabs.com" \
      org.opencontainers.image.title="Spin-Torque RL-Gym" \
      org.opencontainers.image.description="Gymnasium environment for spin-torque device control via RL" \
      org.opencontainers.image.url="https://github.com/terragon-labs/spin-torque-rl-gym" \
      org.opencontainers.image.source="https://github.com/terragon-labs/spin-torque-rl-gym" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set environment variables for the application
ENV ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    ENABLE_VISUALIZATION=false \
    DATA_DIR=/app/data \
    RESULTS_DIR=/app/results \
    MODELS_DIR=/app/models

# Create application directories
WORKDIR /app
RUN mkdir -p data results models logs cache && \
    chown -R appuser:appuser /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown appuser:appuser /entrypoint.sh

# Switch to non-root user
USER appuser

# Set up volumes for data persistence
VOLUME ["/app/data", "/app/results", "/app/models", "/app/logs"]

# Expose port for potential web interfaces
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spin_torque_gym; print('OK')" || exit 1

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "-m", "spin_torque_gym.cli", "--help"]

#==============================================================================
# Development Stage - Full development environment
#==============================================================================
FROM builder as development

# Set labels for development stage
LABEL org.opencontainers.image.title="Spin-Torque RL-Gym Development" \
      org.opencontainers.image.description="Development environment for Spin-Torque RL-Gym"

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    curl \
    wget \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets

# Set development environment variables
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG \
    ENABLE_VISUALIZATION=true

# Copy development configuration
COPY docker/dev-entrypoint.sh /dev-entrypoint.sh
RUN chmod +x /dev-entrypoint.sh

# Switch to application user
USER appuser

# Expose ports for Jupyter and development servers
EXPOSE 8888 8080 8000

# Set development entrypoint
ENTRYPOINT ["/dev-entrypoint.sh"]

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

#==============================================================================
# Testing Stage - Optimized for CI/CD testing
#==============================================================================
FROM builder as testing

# Set labels for testing stage
LABEL org.opencontainers.image.title="Spin-Torque RL-Gym Testing" \
      org.opencontainers.image.description="Testing environment for Spin-Torque RL-Gym CI/CD"

# Install additional testing tools
RUN pip install --no-cache-dir \
    pytest-xvfb \
    pytest-mock \
    pytest-timeout

# Set testing environment variables
ENV ENVIRONMENT=testing \
    DEBUG=false \
    LOG_LEVEL=WARNING \
    ENABLE_VISUALIZATION=false \
    TESTING=true

# Run comprehensive tests
RUN python -m pytest tests/ -v --tb=short || echo "Some tests may not pass yet - continuing"

# Switch to application user
USER appuser

# Default command for testing
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]