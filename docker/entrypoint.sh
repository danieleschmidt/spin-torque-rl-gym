#!/bin/bash
set -e

# Entrypoint script for Spin-Torque RL-Gym production container
# Handles initialization, health checks, and graceful shutdown

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [ENTRYPOINT]${NC} $1"
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

# Signal handlers for graceful shutdown
shutdown() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill any background processes
    if [ -n "$BG_PID" ]; then
        log "Stopping background process (PID: $BG_PID)"
        kill -TERM "$BG_PID" 2>/dev/null || true
        wait "$BG_PID" 2>/dev/null || true
    fi
    
    # Clean up temporary files
    if [ -d "/tmp/spin_torque_gym" ]; then
        log "Cleaning up temporary files"
        rm -rf /tmp/spin_torque_gym
    fi
    
    success "Cleanup completed"
    exit 0
}

# Trap signals for graceful shutdown
trap shutdown SIGTERM SIGINT SIGQUIT

# Environment validation
validate_environment() {
    log "Validating environment..."
    
    # Check required directories exist
    for dir in /app/data /app/results /app/models /app/logs; do
        if [ ! -d "$dir" ]; then
            log "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Check Python installation
    if ! python --version >/dev/null 2>&1; then
        error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Check package installation
    if ! python -c "import spin_torque_gym" >/dev/null 2>&1; then
        error "spin_torque_gym package is not installed"
        exit 1
    fi
    
    # Validate environment variables
    if [ -z "$ENVIRONMENT" ]; then
        warn "ENVIRONMENT not set, defaulting to 'production'"
        export ENVIRONMENT="production"
    fi
    
    if [ -z "$LOG_LEVEL" ]; then
        warn "LOG_LEVEL not set, defaulting to 'INFO'"
        export LOG_LEVEL="INFO"
    fi
    
    success "Environment validation completed"
}

# System health check
health_check() {
    log "Performing health check..."
    
    # Check disk space
    DISK_USAGE=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        warn "Disk usage is at ${DISK_USAGE}% - consider cleaning up"
    fi
    
    # Check memory usage
    if command -v free >/dev/null 2>&1; then
        MEMORY_USAGE=$(free | grep '^Mem:' | awk '{printf "%.0f", $3/$2 * 100.0}')
        if [ "$MEMORY_USAGE" -gt 90 ]; then
            warn "Memory usage is at ${MEMORY_USAGE}%"
        fi
    fi
    
    # Test Python imports
    if ! python -c "import numpy, scipy, matplotlib" >/dev/null 2>&1; then
        error "Failed to import required scientific packages"
        exit 1
    fi
    
    # Test JAX if enabled
    if [ "$ENABLE_JAX" = "true" ]; then
        if ! python -c "import jax; print(f'JAX devices: {jax.devices()}')" >/dev/null 2>&1; then
            warn "JAX is enabled but not working properly"
        else
            log "JAX is available and working"
        fi
    fi
    
    success "Health check completed"
}

# Initialize application
initialize_app() {
    log "Initializing application..."
    
    # Set up logging directory
    LOG_DIR="/app/logs"
    mkdir -p "$LOG_DIR"
    
    # Create application log file
    touch "$LOG_DIR/app.log"
    
    # Set up cache directory
    CACHE_DIR="/app/cache"
    mkdir -p "$CACHE_DIR"
    
    # Initialize configuration if not exists
    if [ ! -f "/app/.env" ] && [ -f "/app/.env.example" ]; then
        log "Creating .env file from example"
        cp /app/.env.example /app/.env
    fi
    
    # Run any database migrations or setup if needed
    if [ "$ENVIRONMENT" = "production" ]; then
        log "Running production initialization"
        # Add any production-specific initialization here
    fi
    
    success "Application initialization completed"
}

# Main execution
main() {
    log "Starting Spin-Torque RL-Gym container"
    log "Environment: $ENVIRONMENT"
    log "Debug: ${DEBUG:-false}"
    log "Log Level: $LOG_LEVEL"
    
    # Run initialization steps
    validate_environment
    health_check
    initialize_app
    
    # Handle different execution modes
    if [ $# -eq 0 ]; then
        # No arguments provided, run default command
        log "No command provided, running default application"
        exec python -m spin_torque_gym.cli --help
    elif [ "$1" = "server" ]; then
        # Run as server
        log "Starting server mode"
        exec python -m spin_torque_gym.server "${@:2}"
    elif [ "$1" = "train" ]; then
        # Run training
        log "Starting training mode"
        exec python -m spin_torque_gym.train "${@:2}"
    elif [ "$1" = "evaluate" ]; then
        # Run evaluation
        log "Starting evaluation mode"
        exec python -m spin_torque_gym.evaluate "${@:2}"
    elif [ "$1" = "benchmark" ]; then
        # Run benchmarks
        log "Starting benchmark mode"
        exec python -m spin_torque_gym.benchmark "${@:2}"
    elif [ "$1" = "shell" ] || [ "$1" = "bash" ]; then
        # Interactive shell
        log "Starting interactive shell"
        exec /bin/bash "${@:2}"
    elif [ "$1" = "python" ]; then
        # Python interpreter
        log "Starting Python interpreter"
        exec python "${@:2}"
    elif [ "$1" = "jupyter" ]; then
        # Jupyter notebook (if available)
        log "Starting Jupyter notebook"
        if command -v jupyter >/dev/null 2>&1; then
            exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root "${@:2}"
        else
            error "Jupyter is not installed"
            exit 1
        fi
    elif [ "$1" = "test" ]; then
        # Run tests
        log "Running tests"
        exec python -m pytest tests/ -v "${@:2}"
    else
        # Execute provided command
        log "Executing command: $*"
        exec "$@"
    fi
}

# Run main function with all arguments
main "$@"