#!/bin/bash
set -e

# Development entrypoint script for Spin-Torque RL-Gym
# Provides additional development tools and convenience features

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [DEV-ENTRYPOINT]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1"
}

info() {
    echo -e "${PURPLE}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1"
}

# Development environment setup
setup_dev_environment() {
    log "Setting up development environment..."
    
    # Install package in development mode if not already installed
    if [ -f "/app/pyproject.toml" ]; then
        log "Installing package in development mode"
        pip install -e "/app[dev,jax,viz]" --quiet || warn "Failed to install package in dev mode"
    fi
    
    # Set up pre-commit hooks if available
    if [ -f "/app/.pre-commit-config.yaml" ]; then
        log "Installing pre-commit hooks"
        pre-commit install --install-hooks >/dev/null 2>&1 || warn "Failed to install pre-commit hooks"
    fi
    
    # Create development directories
    for dir in /app/experiments /app/notebooks /app/scripts; do
        if [ ! -d "$dir" ]; then
            log "Creating development directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Set up Jupyter configuration
    if [ ! -d "/home/appuser/.jupyter" ]; then
        log "Setting up Jupyter configuration"
        mkdir -p /home/appuser/.jupyter
        cat > /home/appuser/.jupyter/jupyter_lab_config.py << EOF
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = '${JUPYTER_TOKEN:-spin-torque-dev}'
c.ServerApp.password = ''
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True
c.ServerApp.notebook_dir = '/app'
EOF
    fi
    
    success "Development environment setup completed"
}

# Display helpful information
show_dev_info() {
    echo ""
    info "=== Spin-Torque RL-Gym Development Container ==="
    info "Available commands:"
    info "  jupyter    - Start Jupyter Lab server"
    info "  notebook   - Start Jupyter Notebook server"
    info "  test       - Run test suite"
    info "  test-watch - Run tests in watch mode"
    info "  lint       - Run code linting"
    info "  format     - Format code"
    info "  shell      - Start interactive shell"
    info "  python     - Start Python interpreter"
    info ""
    info "Jupyter Lab will be available at:"
    info "  http://localhost:8888 (token: ${JUPYTER_TOKEN:-spin-torque-dev})"
    info ""
    info "Development directories:"
    info "  /app/experiments - For experiment scripts"
    info "  /app/notebooks   - For Jupyter notebooks"
    info "  /app/scripts     - For utility scripts"
    info ""
    info "Useful development commands:"
    info "  make test        - Run tests"
    info "  make lint        - Run linting"
    info "  make format      - Format code"
    info "  make dev-setup   - Set up development environment"
    echo ""
}

# Watch mode for tests
run_test_watch() {
    log "Starting test watch mode..."
    if command -v pytest-watch >/dev/null 2>&1; then
        exec ptw -- tests/ -v
    else
        warn "pytest-watch not available, running tests once"
        exec python -m pytest tests/ -v
    fi
}

# Code formatting
run_format() {
    log "Formatting code..."
    if [ -f "/app/pyproject.toml" ]; then
        black /app/spin_torque_gym /app/tests --config /app/pyproject.toml
        isort /app/spin_torque_gym /app/tests --settings-path /app/pyproject.toml
        success "Code formatting completed"
    else
        error "pyproject.toml not found"
        exit 1
    fi
}

# Code linting
run_lint() {
    log "Running code linting..."
    if [ -f "/app/pyproject.toml" ]; then
        ruff check /app/spin_torque_gym /app/tests --config /app/pyproject.toml
        mypy /app/spin_torque_gym --config-file /app/pyproject.toml
        success "Linting completed"
    else
        error "pyproject.toml not found"
        exit 1
    fi
}

# Install additional development packages
install_dev_packages() {
    log "Installing additional development packages..."
    pip install --quiet \
        jupyter-widgets \
        ipywidgets \
        plotly \
        seaborn \
        pytest-watch \
        jupyter-lab-git \
        jupyterlab-lsp \
        python-lsp-server \
        || warn "Some development packages failed to install"
}

# Main execution
main() {
    log "Starting Spin-Torque RL-Gym development container"
    
    # Set up development environment
    setup_dev_environment
    install_dev_packages
    
    # Show development information
    show_dev_info
    
    # Handle different execution modes
    if [ $# -eq 0 ]; then
        # No arguments provided, start Jupyter Lab
        log "No command provided, starting Jupyter Lab"
        exec jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --ServerApp.token="${JUPYTER_TOKEN:-spin-torque-dev}" \
            --ServerApp.password='' \
            --ServerApp.allow_origin='*' \
            --ServerApp.allow_remote_access=True \
            --ServerApp.notebook_dir='/app'
    elif [ "$1" = "jupyter" ]; then
        # Start Jupyter Lab
        log "Starting Jupyter Lab"
        exec jupyter lab \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --ServerApp.token="${JUPYTER_TOKEN:-spin-torque-dev}" \
            --ServerApp.password='' \
            --ServerApp.allow_origin='*' \
            --ServerApp.allow_remote_access=True \
            --ServerApp.notebook_dir='/app' \
            "${@:2}"
    elif [ "$1" = "notebook" ]; then
        # Start Jupyter Notebook
        log "Starting Jupyter Notebook"
        exec jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token="${JUPYTER_TOKEN:-spin-torque-dev}" \
            --NotebookApp.password='' \
            --NotebookApp.allow_origin='*' \
            --NotebookApp.allow_remote_access=True \
            --notebook-dir='/app' \
            "${@:2}"
    elif [ "$1" = "test" ]; then
        # Run tests
        log "Running tests"
        exec python -m pytest tests/ -v "${@:2}"
    elif [ "$1" = "test-watch" ]; then
        # Run tests in watch mode
        run_test_watch
    elif [ "$1" = "format" ]; then
        # Format code
        run_format
    elif [ "$1" = "lint" ]; then
        # Run linting
        run_lint
    elif [ "$1" = "shell" ] || [ "$1" = "bash" ]; then
        # Interactive shell
        log "Starting interactive shell"
        exec /bin/bash "${@:2}"
    elif [ "$1" = "python" ]; then
        # Python interpreter
        log "Starting Python interpreter"
        exec python "${@:2}"
    else
        # Execute provided command
        log "Executing command: $*"
        exec "$@"
    fi
}

# Run main function with all arguments
main "$@"