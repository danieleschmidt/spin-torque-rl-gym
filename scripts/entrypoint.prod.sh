#!/bin/bash
# Production entrypoint script for Spin-Torque RL-Gym

set -e

echo "🚀 Starting Spin-Torque RL-Gym in production mode..."

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Production environment validation
validate_environment() {
    log "Validating production environment..."
    
    # Check required environment variables
    required_vars=(
        "SPIN_TORQUE_ENV"
        "PYTHONPATH"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log "ERROR: Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate Python installation
    if ! python -c "import spin_torque_gym; print('Import successful')" > /dev/null 2>&1; then
        log "ERROR: Failed to import spin_torque_gym"
        exit 1
    fi
    
    log "Environment validation passed ✅"
}

# Initialize performance optimization
initialize_optimization() {
    log "Initializing performance optimization..."
    
    python -c "
from spin_torque_gym.utils.performance_optimization import initialize_performance_optimization, PerformanceConfig, OptimizationLevel
import os

# Production optimization configuration
config = PerformanceConfig(
    optimization_level=OptimizationLevel.AGGRESSIVE,
    enable_caching=True,
    enable_parallelization=True,
    enable_memory_pooling=True,
    max_cache_size=int(os.getenv('SPIN_TORQUE_CACHE_SIZE', 5000)),
    max_workers=int(os.getenv('SPIN_TORQUE_MAX_WORKERS', 8))
)

# Initialize global optimizer
optimizer = initialize_performance_optimization(config)
print('Performance optimization initialized')
"
    
    log "Performance optimization initialized ✅"
}

# Setup logging
setup_logging() {
    log "Setting up production logging..."
    
    # Create log directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set log file permissions
    touch /app/logs/app.log
    touch /app/logs/error.log
    touch /app/logs/performance.log
    
    log "Logging setup completed ✅"
}

# Health check function
health_check() {
    log "Performing startup health check..."
    
    if python /healthcheck.py; then
        log "Health check passed ✅"
        return 0
    else
        log "Health check failed ❌"
        return 1
    fi
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Clean up temporary files
    find /tmp -name "spin_torque_*" -mtime +1 -delete 2>/dev/null || true
    
    # Clear old cache files if they exist
    if [[ -d "/app/cache" ]]; then
        find /app/cache -name "*.cache" -mtime +7 -delete 2>/dev/null || true
    fi
    
    log "Cleanup completed ✅"
}

# Trap signals for graceful shutdown
trap 'log "Received termination signal, shutting down gracefully..."; cleanup; exit 0' SIGTERM SIGINT

# Main execution
main() {
    log "=== Spin-Torque RL-Gym Production Startup ==="
    
    # Validate environment
    validate_environment
    
    # Setup logging
    setup_logging
    
    # Initialize optimization
    initialize_optimization
    
    # Perform cleanup
    cleanup
    
    # Health check
    if ! health_check; then
        log "Startup health check failed, exiting..."
        exit 1
    fi
    
    log "Production startup completed successfully ✅"
    log "Starting application..."
    
    # Execute the main command
    exec "$@"
}

# Run main function
main "$@"