#!/bin/bash
# Production deployment script for Spin Torque RL-Gym
# Handles build, test, deploy, and monitoring setup

set -euo pipefail

# Configuration
PROJECT_NAME="spin-torque-gym"
REGISTRY="${REGISTRY:-localhost:5000}"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOY_MODE="${DEPLOY_MODE:-docker-compose}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] [INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] [SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] [WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] [ERROR]${NC} $1" >&2
}

# Print usage
show_usage() {
    cat << EOF
Spin Torque RL-Gym Deployment Script

USAGE:
    ./deploy.sh [COMMAND] [OPTIONS]

COMMANDS:
    build       Build Docker images
    test        Run comprehensive tests
    deploy      Deploy to production
    scale       Scale services
    status      Check deployment status
    logs        Show service logs
    cleanup     Clean up resources
    backup      Backup data and models
    restore     Restore from backup
    help        Show this help message

OPTIONS:
    --environment ENV    Set environment (production, staging, development)
    --version VERSION    Set image version tag
    --registry REGISTRY  Set Docker registry
    --replicas N         Number of replicas (for scale command)
    --service NAME       Target specific service
    --follow            Follow logs (for logs command)
    --force             Force operation without confirmation

EXAMPLES:
    ./deploy.sh build
    ./deploy.sh test --environment staging
    ./deploy.sh deploy --version v1.2.3
    ./deploy.sh scale --service spin-torque-gym --replicas 3
    ./deploy.sh logs --service spin-torque-gym --follow

EOF
}

# Parse command line arguments
parse_args() {
    COMMAND=""
    REPLICAS=1
    SERVICE=""
    FOLLOW_LOGS=false
    FORCE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            build|test|deploy|scale|status|logs|cleanup|backup|restore|help)
                COMMAND="$1"
                shift
                ;;
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --replicas)
                REPLICAS="$2"
                shift 2
                ;;
            --service)
                SERVICE="$2"
                shift 2
                ;;
            --follow)
                FOLLOW_LOGS=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "$COMMAND" ]]; then
        error "No command specified"
        show_usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    REQUIRED_SPACE=1048576  # 1GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        error "Insufficient disk space. Required: 1GB, Available: $(($AVAILABLE_SPACE / 1024))MB"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Build production image
    docker build \
        -f docker/Dockerfile \
        --target production \
        --tag "${PROJECT_NAME}:${VERSION}" \
        --tag "${PROJECT_NAME}:latest" \
        .
    
    # Tag for registry if specified
    if [[ "$REGISTRY" != "localhost:5000" ]]; then
        docker tag "${PROJECT_NAME}:${VERSION}" "${REGISTRY}/${PROJECT_NAME}:${VERSION}"
        docker tag "${PROJECT_NAME}:latest" "${REGISTRY}/${PROJECT_NAME}:latest"
    fi
    
    success "Docker images built successfully"
}

# Run comprehensive tests
run_tests() {
    log "Running comprehensive tests..."
    
    # Create test container
    docker run --rm \
        --name "${PROJECT_NAME}-test" \
        -v "$(pwd):/app" \
        "${PROJECT_NAME}:${VERSION}" \
        test
    
    success "All tests passed"
}

# Deploy services
deploy_services() {
    log "Deploying services to $ENVIRONMENT environment..."
    
    # Ensure directories exist
    mkdir -p data results logs models
    
    # Deploy based on mode
    case "$DEPLOY_MODE" in
        "docker-compose")
            # Use Docker Compose
            export COMPOSE_PROJECT_NAME="$PROJECT_NAME"
            export IMAGE_VERSION="$VERSION"
            
            docker-compose -f production-docker-compose.yml down --remove-orphans
            docker-compose -f production-docker-compose.yml up -d
            ;;
        "kubernetes")
            # Deploy to Kubernetes (if k8s manifests exist)
            if [[ -d "k8s" ]]; then
                kubectl apply -f k8s/
            else
                error "Kubernetes manifests not found"
                exit 1
            fi
            ;;
        *)
            error "Unsupported deploy mode: $DEPLOY_MODE"
            exit 1
            ;;
    esac
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    timeout 120 bash -c 'until docker-compose -f production-docker-compose.yml ps | grep -q "Up (healthy)"; do sleep 2; done'
    
    success "Deployment completed successfully"
}

# Scale services
scale_services() {
    log "Scaling services..."
    
    if [[ -z "$SERVICE" ]]; then
        error "Service name required for scaling. Use --service option"
        exit 1
    fi
    
    docker-compose -f production-docker-compose.yml up -d --scale "$SERVICE=$REPLICAS"
    
    success "Scaled $SERVICE to $REPLICAS replicas"
}

# Check deployment status
check_status() {
    log "Checking deployment status..."
    
    echo "=== Docker Compose Services ==="
    docker-compose -f production-docker-compose.yml ps
    
    echo ""
    echo "=== Service Health ==="
    docker-compose -f production-docker-compose.yml exec spin-torque-gym python -c "
import requests
import json

try:
    response = requests.get('http://localhost:8080/health', timeout=5)
    health_data = response.json()
    print(f'Health Status: {health_data.get(\"status\", \"unknown\")}')
    print(f'Timestamp: {health_data.get(\"timestamp\", \"unknown\")}')
except Exception as e:
    print(f'Health check failed: {e}')
"
    
    echo ""
    echo "=== Resource Usage ==="
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Show service logs
show_logs() {
    log "Showing service logs..."
    
    if [[ -n "$SERVICE" ]]; then
        if [[ "$FOLLOW_LOGS" == "true" ]]; then
            docker-compose -f production-docker-compose.yml logs -f "$SERVICE"
        else
            docker-compose -f production-docker-compose.yml logs --tail=50 "$SERVICE"
        fi
    else
        if [[ "$FOLLOW_LOGS" == "true" ]]; then
            docker-compose -f production-docker-compose.yml logs -f
        else
            docker-compose -f production-docker-compose.yml logs --tail=50
        fi
    fi
}

# Cleanup resources
cleanup_resources() {
    log "Cleaning up resources..."
    
    if [[ "$FORCE" == "false" ]]; then
        read -p "This will remove all containers and volumes. Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Cleanup cancelled"
            return
        fi
    fi
    
    # Stop and remove containers
    docker-compose -f production-docker-compose.yml down --volumes --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    success "Cleanup completed"
}

# Backup data and models
backup_data() {
    log "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup data directory
    if [[ -d "data" ]]; then
        tar -czf "$BACKUP_DIR/data.tar.gz" data/
    fi
    
    # Backup models directory
    if [[ -d "models" ]]; then
        tar -czf "$BACKUP_DIR/models.tar.gz" models/
    fi
    
    # Backup configuration
    cp -r docker/ "$BACKUP_DIR/"
    cp production-docker-compose.yml "$BACKUP_DIR/"
    
    success "Backup created at $BACKUP_DIR"
}

# Restore from backup
restore_data() {
    log "Restoring from backup..."
    
    if [[ ! -d "backups" ]]; then
        error "No backups directory found"
        exit 1
    fi
    
    # List available backups
    echo "Available backups:"
    ls -la backups/
    
    read -p "Enter backup directory name: " BACKUP_NAME
    BACKUP_PATH="backups/$BACKUP_NAME"
    
    if [[ ! -d "$BACKUP_PATH" ]]; then
        error "Backup not found: $BACKUP_PATH"
        exit 1
    fi
    
    # Restore data
    if [[ -f "$BACKUP_PATH/data.tar.gz" ]]; then
        tar -xzf "$BACKUP_PATH/data.tar.gz"
    fi
    
    # Restore models
    if [[ -f "$BACKUP_PATH/models.tar.gz" ]]; then
        tar -xzf "$BACKUP_PATH/models.tar.gz"
    fi
    
    success "Restore completed from $BACKUP_PATH"
}

# Main execution
main() {
    log "Starting Spin Torque RL-Gym deployment script"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Command: $COMMAND"
    
    check_prerequisites
    
    case "$COMMAND" in
        "build")
            build_images
            ;;
        "test")
            build_images
            run_tests
            ;;
        "deploy")
            build_images
            run_tests
            deploy_services
            ;;
        "scale")
            scale_services
            ;;
        "status")
            check_status
            ;;
        "logs")
            show_logs
            ;;
        "cleanup")
            cleanup_resources
            ;;
        "backup")
            backup_data
            ;;
        "restore")
            restore_data
            ;;
        "help")
            show_usage
            ;;
        *)
            error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
    
    success "Operation completed successfully"
}

# Parse arguments and run main function
parse_args "$@"
main