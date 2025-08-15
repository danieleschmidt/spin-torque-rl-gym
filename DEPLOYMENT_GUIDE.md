# Spin Torque RL-Gym - Production Deployment Guide

## 🚀 Quick Start

Deploy Spin Torque RL-Gym to production in 3 steps:

```bash
# 1. Build and test
./deploy.sh build
./deploy.sh test

# 2. Deploy to production
./deploy.sh deploy --environment production

# 3. Verify deployment
./deploy.sh status
```

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: 2+ cores (4+ cores recommended)
- **Memory**: 4GB RAM minimum (8GB+ recommended)
- **Storage**: 10GB free space minimum
- **Network**: Internet access for package downloads

### Software Dependencies
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Python**: 3.12+ (for development/testing)
- **Git**: Latest version

### Installation Commands
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

## 🏗️ Deployment Architecture

### Production Stack
```
┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Application   │
│    (Optional)   │    │   Container     │
└─────────────────┘    └─────────────────┘
                              │
┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Data Storage  │
│  (Prometheus)   │    │   (Volumes)     │
└─────────────────┘    └─────────────────┘
```

### Services Overview
- **spin-torque-gym**: Main application service
- **jupyter**: Development notebook interface (optional)
- **benchmark**: Performance testing service (optional)
- **monitoring**: Metrics collection (optional)

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Production Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Performance Settings
ENABLE_MONITORING=true
CACHE_SIZE=1000
MAX_WORKERS=4

# Resource Limits
MEMORY_LIMIT=4G
CPU_LIMIT=2.0

# Security Settings
ENABLE_SECURITY_HEADERS=true
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Optional: Advanced Settings
ENABLE_JAX=false
SOLVER_TIMEOUT=5.0
RETRY_ATTEMPTS=3
```

### Port Configuration
Default ports used by the application:
- **8080**: Main HTTP API and demo server
- **8888**: Jupyter notebook interface
- **9090**: Prometheus monitoring (if enabled)

## 🚀 Deployment Methods

### Method 1: Docker Compose (Recommended)

1. **Clone and prepare**:
```bash
git clone <repository-url>
cd spin-torque-rl-gym
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. **Deploy**:
```bash
./deploy.sh deploy --environment production
```

4. **Verify deployment**:
```bash
./deploy.sh status
curl http://localhost:8080/health
```

### Method 2: Manual Docker

1. **Build image**:
```bash
docker build -f docker/Dockerfile -t spin-torque-gym:latest .
```

2. **Run container**:
```bash
docker run -d \
  --name spin-torque-gym-prod \
  -p 8080:8080 \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  spin-torque-gym:latest serve
```

### Method 3: Kubernetes (Advanced)

For Kubernetes deployment, create the following manifests:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spin-torque-gym
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spin-torque-gym
  template:
    metadata:
      labels:
        app: spin-torque-gym
    spec:
      containers:
      - name: spin-torque-gym
        image: spin-torque-gym:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
```

Deploy with:
```bash
kubectl apply -f k8s/
```

## 📊 Monitoring and Health Checks

### Built-in Health Checks
The application includes comprehensive health monitoring:

- **HTTP Health Endpoint**: `GET /health`
- **Docker Health Check**: Integrated container health monitoring
- **Application Metrics**: Performance and usage statistics

### Monitoring Endpoints
- `/health` - Basic health status
- `/info` - System information
- `/demo` - Live environment demonstration

### Health Check Example
```bash
# Check application health
curl -s http://localhost:8080/health | jq

# Expected response:
{
  "status": "healthy",
  "timestamp": 1692123456.789
}
```

### Log Monitoring
```bash
# View application logs
./deploy.sh logs --service spin-torque-gym --follow

# View specific service logs
docker-compose -f production-docker-compose.yml logs -f spin-torque-gym
```

## 🔄 Scaling and Performance

### Horizontal Scaling
Scale the application to handle increased load:

```bash
# Scale to 3 replicas
./deploy.sh scale --service spin-torque-gym --replicas 3

# Monitor resource usage
docker stats --no-stream
```

### Performance Tuning
1. **Adjust worker processes**:
```bash
export MAX_WORKERS=8  # Increase for CPU-intensive workloads
```

2. **Optimize memory usage**:
```bash
export CACHE_SIZE=2000  # Increase cache for better performance
```

3. **Configure resource limits**:
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Increase CPU limit
      memory: 8G       # Increase memory limit
```

### Performance Benchmarks
Run performance tests:
```bash
./deploy.sh test --environment benchmark
```

Expected performance metrics:
- **Reset time**: < 0.001s
- **Step time**: < 5.0s (depending on solver complexity)
- **Memory usage**: < 4GB per instance
- **CPU usage**: Variable (20-80% during computation)

## 🔐 Security Considerations

### Container Security
- Uses non-root user (`spinuser`)
- Minimal base image (python:3.12-slim)
- No unnecessary packages in production image
- Security scanning integrated in CI/CD

### Network Security
- Only necessary ports exposed
- Optional TLS/SSL configuration
- CORS headers configured

### Data Security
- No sensitive data in images
- Volume mounts for persistent data
- Configurable security headers

### Security Best Practices
```bash
# Update base images regularly
docker pull python:3.12-slim

# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/tmp/scan \
  aquasec/trivy image spin-torque-gym:latest

# Use secrets management
docker secret create db_password password.txt
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Container fails to start
```bash
# Check logs
docker logs spin-torque-gym-prod

# Common causes:
# - Insufficient memory
# - Port conflicts
# - Missing environment variables
```

#### 2. Health check failures
```bash
# Test health endpoint
curl -v http://localhost:8080/health

# Check container health
docker inspect spin-torque-gym-prod | jq '.[0].State.Health'
```

#### 3. Performance issues
```bash
# Monitor resource usage
docker stats --no-stream spin-torque-gym-prod

# Check solver performance
curl http://localhost:8080/demo
```

#### 4. Memory leaks
```bash
# Monitor memory usage over time
watch -n 5 'docker stats --no-stream spin-torque-gym-prod | grep Memory'

# Restart if memory usage exceeds limits
docker restart spin-torque-gym-prod
```

### Debug Mode
Run in debug mode for troubleshooting:
```bash
docker run -it --rm \
  -e ENVIRONMENT=development \
  -e LOG_LEVEL=DEBUG \
  spin-torque-gym:latest bash
```

### Log Analysis
```bash
# Search for errors in logs
docker logs spin-torque-gym-prod 2>&1 | grep -i error

# Monitor real-time logs
docker logs -f spin-torque-gym-prod
```

## 📚 Maintenance

### Regular Updates
```bash
# Update and redeploy
git pull origin main
./deploy.sh deploy --version latest

# Update base images
docker pull python:3.12-slim
./deploy.sh build
```

### Backup and Restore
```bash
# Create backup
./deploy.sh backup

# Restore from backup
./deploy.sh restore
```

### Cleanup
```bash
# Clean up unused resources
./deploy.sh cleanup

# Force cleanup (remove everything)
./deploy.sh cleanup --force
```

## 🔗 Additional Resources

- **API Documentation**: `/docs` endpoint (if enabled)
- **Example Notebooks**: Available in `/examples` directory
- **Performance Benchmarks**: Run with `./deploy.sh benchmark`
- **Source Code**: [GitHub Repository](https://github.com/your-org/spin-torque-rl-gym)

## 📞 Support

For deployment issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Consult the project documentation
4. Create an issue in the project repository

---

**✅ Production Deployment Checklist**
- [ ] Prerequisites installed
- [ ] Environment configured  
- [ ] Build successful
- [ ] Tests passing
- [ ] Health checks working
- [ ] Monitoring configured
- [ ] Security reviewed
- [ ] Backup strategy in place
- [ ] Performance validated
- [ ] Documentation updated