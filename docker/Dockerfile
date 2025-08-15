# Production Dockerfile for Spin Torque RL-Gym
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
COPY spin_torque_gym/ ./spin_torque_gym/

# Install package in build mode
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.12-slim as production

# Create non-root user for security
RUN groupadd -r spinuser && useradd -r -g spinuser spinuser

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/spin_torque_gym ./spin_torque_gym

# Copy production files
COPY docker/entrypoint.sh /entrypoint.sh
COPY examples/ ./examples/
COPY README.md ./

# Set proper permissions
RUN chmod +x /entrypoint.sh && \
    chown -R spinuser:spinuser /app

# Switch to non-root user
USER spinuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import spin_torque_gym; import gymnasium as gym; env = gym.make('SpinTorque-v0'); env.reset(); env.close()" || exit 1

# Expose ports
EXPOSE 8080 8888

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Set default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]