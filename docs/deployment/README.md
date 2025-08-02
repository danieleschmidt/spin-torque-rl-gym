# Deployment Guide for Spin-Torque RL-Gym

This guide covers deployment strategies and configurations for different environments.

## Quick Start

### Development Environment

```bash
# Start development environment with Jupyter Lab
docker-compose up dev

# Access Jupyter Lab at http://localhost:8888
# Token: spin-torque-dev (configurable via JUPYTER_TOKEN)
```

### Production Environment

```bash
# Build and start production services
docker-compose -f docker/docker-compose.prod.yml up -d

# Access application at http://localhost:8080
# Access monitoring at http://localhost:3000 (Grafana)
```

## Docker Images

### Available Targets

The multi-stage Dockerfile provides several targets:

- **`production`**: Minimal runtime image for production
- **`development`**: Full development environment with tools
- **`testing`**: Optimized for CI/CD testing

### Building Images

```bash
# Build production image
docker build --target production -t spin-torque-rl-gym:latest .

# Build development image
docker build --target development -t spin-torque-rl-gym:dev .

# Build with version tags
docker build --target production \
  --build-arg VERSION=0.1.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  -t spin-torque-rl-gym:0.1.0 .
```

## Environment Configurations

### Development (`docker-compose.yml`)

Default configuration for development:

- Single application container
- Jupyter Lab interface
- Volume mounting for live code changes
- Development tools and utilities

```bash
# Start development environment
docker-compose up dev

# Run tests
docker-compose run --rm test

# Interactive shell
docker-compose run --rm dev shell
```

### Production (`docker/docker-compose.prod.yml`)

Production-ready configuration:

- Load balancer (Nginx)
- Database (PostgreSQL)
- Caching (Redis)
- Experiment tracking (MLflow)
- Monitoring (Prometheus + Grafana)
- Log aggregation (Fluentd)

```bash
# Start production stack
docker-compose -f docker/docker-compose.prod.yml up -d

# Scale application
docker-compose -f docker/docker-compose.prod.yml up -d --scale app=3

# View logs
docker-compose -f docker/docker-compose.prod.yml logs -f app
```

## Service Profiles

Use Docker Compose profiles to run specific service combinations:

```bash
# Development only
docker-compose --profile development up

# Production with monitoring
docker-compose --profile production --profile monitoring up

# Full stack (all services)
docker-compose --profile full up

# GPU-accelerated training
docker-compose --profile gpu up
```

## Environment Variables

### Core Application

```bash
# Application settings
ENVIRONMENT=production|development|testing
DEBUG=true|false
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
RANDOM_SEED=42

# Performance settings
NUM_CORES=0  # 0 = auto-detect
GPU_DEVICE_ID=-1  # -1 = CPU only
ENABLE_JAX=false
JAX_PLATFORM=cpu|gpu

# Visualization
ENABLE_VISUALIZATION=false
VISUALIZATION_FPS=30
```

### Database Configuration

```bash
# PostgreSQL
POSTGRES_DB=spin_torque_db
POSTGRES_USER=spin_torque_user
POSTGRES_PASSWORD=spin_torque_pass
POSTGRES_PORT=5432

# Connection URLs
POSTGRES_URL=postgresql://user:pass@host:port/db
```

### Caching and Queues

```bash
# Redis
REDIS_PASSWORD=spin_torque_redis
REDIS_PORT=6379
REDIS_URL=redis://:password@host:port
```

### Monitoring and Tracking

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_PORT=5000

# Grafana
GRAFANA_PASSWORD=admin
GRAFANA_PORT=3000

# Prometheus
PROMETHEUS_PORT=9090
```

### Development Tools

```bash
# Jupyter
JUPYTER_TOKEN=spin-torque-dev
JUPYTER_PORT=8888

# Application ports
APP_PORT=8080
HTTP_PORT=80
HTTPS_PORT=443
```

## Volume Management

### Development Volumes

- **Source code**: Live mounting for development
- **Data**: Persistent data storage
- **Results**: Experiment results
- **Models**: Trained model storage

### Production Volumes

Configured for persistent storage:

```bash
# Create volume directories
mkdir -p volumes/{data,results,models,logs,postgres,redis,mlflow,grafana,prometheus}

# Set proper permissions
chown -R 1000:1000 volumes/
```

### Backup Strategy

```bash
# Backup database
docker-compose exec postgres pg_dump -U spin_torque_user spin_torque_db > backup.sql

# Backup volumes
tar -czf volumes-backup.tar.gz volumes/

# Restore database
docker-compose exec -T postgres psql -U spin_torque_user spin_torque_db < backup.sql
```

## Security Considerations

### Container Security

- Non-root user execution
- Read-only filesystem where possible
- Resource limits and constraints
- Security scanning with `docker scan`

### Network Security

- Internal Docker networks
- Exposed ports only where necessary
- TLS/SSL termination at load balancer
- Database access restricted to internal network

### Secrets Management

Use Docker secrets or external secret management:

```bash
# Using environment files
echo "POSTGRES_PASSWORD=secure_password" > .env.prod
docker-compose --env-file .env.prod up
```

For production, consider:
- AWS Secrets Manager
- HashiCorp Vault
- Kubernetes Secrets
- Azure Key Vault

## Performance Optimization

### Resource Allocation

```yaml
# Docker Compose resource limits
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

### Database Optimization

PostgreSQL configuration (`docker/postgres/postgresql.conf`):

```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

# Connection settings
max_connections = 100
```

### Application Optimization

- Use JAX for GPU acceleration
- Configure parallel processing
- Optimize batch sizes
- Enable caching layers

## Health Checks and Monitoring

### Application Health

```bash
# Health check endpoint
curl http://localhost:8080/health

# Container health status
docker ps --filter "health=healthy"
```

### Monitoring Stack

Access monitoring interfaces:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000

### Log Management

```bash
# View application logs
docker-compose logs -f app

# View all service logs
docker-compose logs -f

# Log rotation configuration
echo '{"log-driver":"json-file","log-opts":{"max-size":"100m","max-file":"5"}}' > /etc/docker/daemon.json
```

## Scaling and Load Balancing

### Horizontal Scaling

```bash
# Scale application containers
docker-compose up -d --scale app=3

# With load balancer
docker-compose -f docker/docker-compose.prod.yml up -d --scale app=5
```

### Load Balancer Configuration

Nginx configuration for load balancing:

```nginx
upstream app_servers {
    server app:8080;
    # Additional servers added automatically by Docker Compose
}

server {
    listen 80;
    location / {
        proxy_pass http://app_servers;
    }
}
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build and test
  run: |
    docker build --target testing .
    docker-compose run --rm test

- name: Deploy to production
  run: |
    docker-compose -f docker/docker-compose.prod.yml up -d
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'docker build --target production -t spin-torque-rl-gym:${BUILD_NUMBER} .'
            }
        }
        stage('Test') {
            steps {
                sh 'docker-compose run --rm test'
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker-compose -f docker/docker-compose.prod.yml up -d'
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Container fails to start**
   ```bash
   # Check logs
   docker-compose logs app
   
   # Check container status
   docker ps -a
   ```

2. **Database connection issues**
   ```bash
   # Test database connection
   docker-compose exec app python -c "
   import psycopg2
   conn = psycopg2.connect('postgresql://user:pass@postgres:5432/db')
   print('Connection successful')
   "
   ```

3. **Memory issues**
   ```bash
   # Check resource usage
   docker stats
   
   # Increase memory limits
   docker-compose up --scale app=1 --memory=8g
   ```

### Debug Mode

```bash
# Run with debug logging
ENVIRONMENT=development DEBUG=true LOG_LEVEL=DEBUG docker-compose up

# Interactive debugging
docker-compose run --rm app python -c "
import pdb; pdb.set_trace()
import spin_torque_gym
"
```

## Migration and Upgrades

### Database Migrations

```bash
# Run migrations
docker-compose exec app python -m spin_torque_gym.db.migrate

# Backup before upgrade
docker-compose exec postgres pg_dump -U user db > backup_$(date +%Y%m%d).sql
```

### Application Updates

```bash
# Pull latest images
docker-compose pull

# Recreate containers with new images
docker-compose up -d --force-recreate

# Verify deployment
docker-compose ps
docker-compose logs -f app
```

## Best Practices

### Development

- Use volume mounts for live code reloading
- Keep development and production environments similar
- Use multi-stage builds for optimization
- Implement proper logging and monitoring

### Production

- Use specific image tags, not `latest`
- Implement health checks for all services
- Set up automated backups
- Monitor resource usage and performance
- Use secrets management for sensitive data
- Implement proper log rotation
- Set up alerting for critical issues

### Security

- Regular security updates
- Vulnerability scanning
- Principle of least privilege
- Network segmentation
- Secure secrets management
- Regular security audits