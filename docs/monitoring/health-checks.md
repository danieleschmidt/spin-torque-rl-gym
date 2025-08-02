# Health Checks and Service Monitoring

Comprehensive health check implementations for all system components.

## Application Health Checks

### Basic Health Endpoint

```python
from flask import Flask, jsonify
from datetime import datetime
import psutil
import psycopg2
import redis
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/health')
def health_check():
    """
    Comprehensive health check endpoint.
    Returns 200 if all systems are healthy, 503 if any issues detected.
    """
    checks = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': get_app_version(),
        'checks': {}
    }
    
    overall_status = True
    
    # Database health
    db_status, db_details = check_database_health()
    checks['checks']['database'] = db_details
    overall_status = overall_status and db_status
    
    # Cache health
    cache_status, cache_details = check_cache_health()
    checks['checks']['cache'] = cache_details
    overall_status = overall_status and cache_status
    
    # System resources
    system_status, system_details = check_system_resources()
    checks['checks']['system'] = system_details
    overall_status = overall_status and system_status
    
    # GPU availability (if configured)
    if is_gpu_enabled():
        gpu_status, gpu_details = check_gpu_health()
        checks['checks']['gpu'] = gpu_details
        overall_status = overall_status and gpu_status
    
    # External dependencies
    deps_status, deps_details = check_external_dependencies()
    checks['checks']['external_dependencies'] = deps_details
    overall_status = overall_status and deps_status
    
    # Update overall status
    checks['status'] = 'healthy' if overall_status else 'unhealthy'
    
    return jsonify(checks), 200 if overall_status else 503

def check_database_health():
    """Check PostgreSQL database connectivity and performance."""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'postgres'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'spin_torque_db'),
            user=os.getenv('POSTGRES_USER', 'spin_torque_user'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        
        cursor = conn.cursor()
        
        # Basic connectivity test
        start_time = time.time()
        cursor.execute('SELECT 1')
        response_time = time.time() - start_time
        
        # Check active connections
        cursor.execute("""
            SELECT COUNT(*) FROM pg_stat_activity 
            WHERE state = 'active' AND datname = %s
        """, (os.getenv('POSTGRES_DB', 'spin_torque_db'),))
        active_connections = cursor.fetchone()[0]
        
        # Check database size
        cursor.execute("""
            SELECT pg_size_pretty(pg_database_size(%s))
        """, (os.getenv('POSTGRES_DB', 'spin_torque_db'),))
        db_size = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        # Determine health status
        is_healthy = (
            response_time < 1.0 and  # Response time under 1 second
            active_connections < 50   # Less than 50 active connections
        )
        
        return is_healthy, {
            'status': 'ok' if is_healthy else 'degraded',
            'response_time_ms': round(response_time * 1000, 2),
            'active_connections': active_connections,
            'database_size': db_size,
            'last_check': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False, {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.utcnow().isoformat()
        }

def check_cache_health():
    """Check Redis cache connectivity and performance."""
    try:
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=os.getenv('REDIS_PORT', 6379),
            password=os.getenv('REDIS_PASSWORD'),
            socket_timeout=2
        )
        
        # Basic connectivity and latency test
        start_time = time.time()
        r.ping()
        response_time = time.time() - start_time
        
        # Get memory usage
        info = r.info('memory')
        used_memory = info['used_memory']
        max_memory = info.get('maxmemory', 0)
        
        # Calculate memory usage percentage
        if max_memory > 0:
            memory_usage_percent = (used_memory / max_memory) * 100
        else:
            memory_usage_percent = 0
        
        # Get keyspace info
        keyspace_info = r.info('keyspace')
        
        # Determine health status
        is_healthy = (
            response_time < 0.1 and  # Response time under 100ms
            memory_usage_percent < 90  # Memory usage under 90%
        )
        
        return is_healthy, {
            'status': 'ok' if is_healthy else 'degraded',
            'response_time_ms': round(response_time * 1000, 2),
            'used_memory_mb': round(used_memory / 1024 / 1024, 2),
            'memory_usage_percent': round(memory_usage_percent, 2),
            'connected_clients': info.get('connected_clients', 0),
            'keyspace_info': keyspace_info,
            'last_check': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}")
        return False, {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.utcnow().isoformat()
        }

def check_system_resources():
    """Check system resource usage."""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Load average (Unix only)
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            load_avg = None
        
        # Determine health status
        is_healthy = (
            cpu_percent < 80 and
            memory_percent < 80 and
            disk_percent < 90
        )
        
        details = {
            'status': 'ok' if is_healthy else 'degraded',
            'cpu_percent': round(cpu_percent, 2),
            'memory_percent': round(memory_percent, 2),
            'memory_available_gb': round(memory.available / 1024**3, 2),
            'disk_percent': round(disk_percent, 2),
            'disk_free_gb': round(disk.free / 1024**3, 2),
            'last_check': datetime.utcnow().isoformat()
        }
        
        if load_avg:
            details['load_average'] = [round(x, 2) for x in load_avg]
        
        return is_healthy, details
        
    except Exception as e:
        logger.error(f"System resource check failed: {str(e)}")
        return False, {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.utcnow().isoformat()
        }

def check_gpu_health():
    """Check GPU availability and usage."""
    try:
        import GPUtil
        
        gpus = GPUtil.getGPUs()
        if not gpus:
            return False, {
                'status': 'unavailable',
                'message': 'No GPUs detected',
                'last_check': datetime.utcnow().isoformat()
            }
        
        gpu_info = []
        overall_healthy = True
        
        for gpu in gpus:
            gpu_healthy = (
                gpu.memoryUtil < 0.9 and  # Memory usage under 90%
                gpu.temperature < 85       # Temperature under 85°C
            )
            overall_healthy = overall_healthy and gpu_healthy
            
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load_percent': round(gpu.load * 100, 2),
                'memory_percent': round(gpu.memoryUtil * 100, 2),
                'memory_used_mb': round(gpu.memoryUsed, 2),
                'memory_total_mb': round(gpu.memoryTotal, 2),
                'temperature_c': gpu.temperature
            })
        
        return overall_healthy, {
            'status': 'ok' if overall_healthy else 'degraded',
            'gpu_count': len(gpus),
            'gpus': gpu_info,
            'last_check': datetime.utcnow().isoformat()
        }
        
    except ImportError:
        return True, {
            'status': 'not_configured',
            'message': 'GPU monitoring not available',
            'last_check': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"GPU health check failed: {str(e)}")
        return False, {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.utcnow().isoformat()
        }

def check_external_dependencies():
    """Check external service dependencies."""
    try:
        dependencies = []
        overall_healthy = True
        
        # Check MLflow (if configured)
        mlflow_url = os.getenv('MLFLOW_TRACKING_URI')
        if mlflow_url:
            mlflow_status = check_http_endpoint(f"{mlflow_url}/health")
            dependencies.append({
                'name': 'MLflow',
                'url': mlflow_url,
                'status': mlflow_status['status'],
                'response_time_ms': mlflow_status.get('response_time_ms', 0)
            })
            overall_healthy = overall_healthy and mlflow_status['healthy']
        
        # Add other external dependencies as needed
        
        return overall_healthy, {
            'status': 'ok' if overall_healthy else 'degraded',
            'dependencies': dependencies,
            'last_check': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"External dependencies check failed: {str(e)}")
        return False, {
            'status': 'error',
            'error': str(e),
            'last_check': datetime.utcnow().isoformat()
        }

def check_http_endpoint(url, timeout=5):
    """Check HTTP endpoint availability."""
    try:
        import requests
        
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        response_time = time.time() - start_time
        
        return {
            'healthy': response.status_code == 200,
            'status': 'ok' if response.status_code == 200 else 'error',
            'status_code': response.status_code,
            'response_time_ms': round(response_time * 1000, 2)
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'status': 'error',
            'error': str(e),
            'response_time_ms': timeout * 1000
        }

@app.route('/ready')
def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    Returns 200 when application is ready to serve traffic.
    """
    try:
        # Check critical dependencies only
        db_status, _ = check_database_health()
        cache_status, _ = check_cache_health()
        
        if db_status and cache_status:
            return jsonify({
                'status': 'ready',
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'not_ready',
                'timestamp': datetime.utcnow().isoformat()
            }), 503
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503

@app.route('/live')
def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if application process is alive.
    """
    return jsonify({
        'status': 'alive',
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': time.time() - app.start_time
    }), 200

# Helper functions
def get_app_version():
    """Get application version from package or environment."""
    try:
        import spin_torque_gym
        return getattr(spin_torque_gym, '__version__', 'unknown')
    except:
        return os.getenv('APP_VERSION', 'unknown')

def is_gpu_enabled():
    """Check if GPU support is enabled."""
    return os.getenv('ENABLE_JAX', 'false').lower() == 'true'

# Initialize app start time
app.start_time = time.time()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Deep Health Checks

### Training System Health

```python
@app.route('/health/training')
def training_system_health():
    """Detailed health check for training system components."""
    checks = {}
    overall_healthy = True
    
    # Physics simulation health
    physics_status = check_physics_simulation()
    checks['physics_simulation'] = physics_status
    overall_healthy = overall_healthy and physics_status['healthy']
    
    # Model inference health
    model_status = check_model_inference()
    checks['model_inference'] = model_status
    overall_healthy = overall_healthy and model_status['healthy']
    
    # Environment health
    env_status = check_environment_health()
    checks['environment'] = env_status
    overall_healthy = overall_healthy and env_status['healthy']
    
    return jsonify({
        'status': 'healthy' if overall_healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }), 200 if overall_healthy else 503

def check_physics_simulation():
    """Test physics simulation performance."""
    try:
        import numpy as np
        from spin_torque_gym.physics import simple_llgs_step
        
        # Test simulation step
        start_time = time.time()
        m = np.array([0.0, 0.0, 1.0])
        h_eff = np.array([0.1, 0.0, 0.0])
        
        # Simulate 100 steps
        for _ in range(100):
            m = simple_llgs_step(m, h_eff, dt=1e-12)
        
        simulation_time = time.time() - start_time
        steps_per_second = 100 / simulation_time
        
        is_healthy = steps_per_second > 1000  # At least 1000 steps/sec
        
        return {
            'healthy': is_healthy,
            'status': 'ok' if is_healthy else 'slow',
            'steps_per_second': round(steps_per_second, 2),
            'simulation_time_ms': round(simulation_time * 1000, 2)
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'status': 'error',
            'error': str(e)
        }

def check_model_inference():
    """Test model inference performance."""
    try:
        # Test model loading and inference
        start_time = time.time()
        
        # Load model (mock implementation)
        model = load_latest_model()
        
        # Test inference
        dummy_obs = create_dummy_observation()
        action = model.predict(dummy_obs)
        
        inference_time = time.time() - start_time
        
        is_healthy = (
            model is not None and
            action is not None and
            inference_time < 0.1  # Under 100ms
        )
        
        return {
            'healthy': is_healthy,
            'status': 'ok' if is_healthy else 'slow',
            'inference_time_ms': round(inference_time * 1000, 2),
            'model_loaded': model is not None
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'status': 'error',
            'error': str(e)
        }

def check_environment_health():
    """Test environment creation and stepping."""
    try:
        import gymnasium as gym
        
        start_time = time.time()
        
        # Create environment
        env = gym.make('SpinTorque-v0')
        
        # Test reset and step
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        env.close()
        
        env_time = time.time() - start_time
        
        is_healthy = (
            obs is not None and
            reward is not None and
            env_time < 1.0  # Under 1 second
        )
        
        return {
            'healthy': is_healthy,
            'status': 'ok' if is_healthy else 'slow',
            'environment_time_ms': round(env_time * 1000, 2),
            'observation_shape': obs.shape if hasattr(obs, 'shape') else None
        }
        
    except Exception as e:
        return {
            'healthy': False,
            'status': 'error',
            'error': str(e)
        }
```

## Metrics Endpoint

```python
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge, Info

# Define metrics
health_check_counter = Counter('health_checks_total', 'Total health checks', ['endpoint', 'status'])
health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration', ['endpoint'])
system_info = Info('system_info', 'System information')

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    # Update system info
    system_info.info({
        'version': get_app_version(),
        'python_version': platform.python_version(),
        'platform': platform.system(),
        'hostname': socket.gethostname()
    })
    
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

def track_health_check(endpoint_name):
    """Decorator to track health check metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with health_check_duration.labels(endpoint=endpoint_name).time():
                try:
                    result = func(*args, **kwargs)
                    status = 'success' if result[1] == 200 else 'failure'
                    health_check_counter.labels(endpoint=endpoint_name, status=status).inc()
                    return result
                except Exception as e:
                    health_check_counter.labels(endpoint=endpoint_name, status='error').inc()
                    raise
        return wrapper
    return decorator

# Apply metrics tracking to health check endpoints
health_check = track_health_check('health')(health_check)
readiness_check = track_health_check('ready')(readiness_check)
liveness_check = track_health_check('live')(liveness_check)
```

## Container Health Checks

### Docker Health Check

```dockerfile
# In Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health').raise_for_status()" || exit 1
```

### Docker Compose Health Check

```yaml
# In docker-compose.yml
services:
  app:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  postgres:
    image: postgres:15-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
```

## Kubernetes Health Checks

### Deployment with Probes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spin-torque-rl-gym
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: spin-torque-rl-gym:latest
        ports:
        - containerPort: 8080
        
        # Liveness probe - restart container if failing
        livenessProbe:
          httpGet:
            path: /live
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
        
        # Readiness probe - remove from service if failing
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
          successThreshold: 1
        
        # Startup probe - give more time for initial startup
        startupProbe:
          httpGet:
            path: /live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
          successThreshold: 1
        
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
```

## Custom Health Check Scripts

### System Health Monitor

```bash
#!/bin/bash
# system-health-monitor.sh

LOG_FILE="/var/log/health-monitor.log"
ALERT_WEBHOOK="http://alertmanager:9093/api/v1/alerts"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_disk_space() {
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 90 ]; then
        log "CRITICAL: Disk usage is ${DISK_USAGE}%"
        send_alert "disk_space" "critical" "Disk usage is ${DISK_USAGE}%"
        return 1
    elif [ "$DISK_USAGE" -gt 80 ]; then
        log "WARNING: Disk usage is ${DISK_USAGE}%"
        send_alert "disk_space" "warning" "Disk usage is ${DISK_USAGE}%"
        return 1
    fi
    return 0
}

check_memory_usage() {
    MEMORY_USAGE=$(free | grep '^Mem:' | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$MEMORY_USAGE" -gt 90 ]; then
        log "CRITICAL: Memory usage is ${MEMORY_USAGE}%"
        send_alert "memory_usage" "critical" "Memory usage is ${MEMORY_USAGE}%"
        return 1
    elif [ "$MEMORY_USAGE" -gt 80 ]; then
        log "WARNING: Memory usage is ${MEMORY_USAGE}%"
        send_alert "memory_usage" "warning" "Memory usage is ${MEMORY_USAGE}%"
        return 1
    fi
    return 0
}

check_application_health() {
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
    if [ "$HTTP_STATUS" != "200" ]; then
        log "CRITICAL: Application health check failed (HTTP $HTTP_STATUS)"
        send_alert "application_health" "critical" "Health check returned HTTP $HTTP_STATUS"
        return 1
    fi
    return 0
}

send_alert() {
    local alertname=$1
    local severity=$2
    local description=$3
    
    curl -X POST "$ALERT_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "[{
            \"labels\": {
                \"alertname\": \"$alertname\",
                \"severity\": \"$severity\",
                \"instance\": \"$(hostname)\"
            },
            \"annotations\": {
                \"summary\": \"Health check alert\",
                \"description\": \"$description\"
            }
        }]" \
        --max-time 10 || log "Failed to send alert"
}

main() {
    log "Starting health check"
    
    OVERALL_STATUS=0
    
    check_disk_space || OVERALL_STATUS=1
    check_memory_usage || OVERALL_STATUS=1
    check_application_health || OVERALL_STATUS=1
    
    if [ $OVERALL_STATUS -eq 0 ]; then
        log "All health checks passed"
    else
        log "Some health checks failed"
    fi
    
    exit $OVERALL_STATUS
}

main "$@"
```

### Database Health Monitor

```python
#!/usr/bin/env python3
"""Database health monitoring script."""

import os
import sys
import time
import psycopg2
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('db_health_monitor')

def check_database_connections():
    """Check number of active database connections."""
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', 5432),
        database=os.getenv('POSTGRES_DB', 'spin_torque_db'),
        user=os.getenv('POSTGRES_USER', 'spin_torque_user'),
        password=os.getenv('POSTGRES_PASSWORD')
    )
    
    cursor = conn.cursor()
    
    # Check active connections
    cursor.execute("""
        SELECT COUNT(*) as active_connections,
               MAX(EXTRACT(EPOCH FROM (now() - query_start))) as longest_query_seconds
        FROM pg_stat_activity 
        WHERE state = 'active' AND datname = %s
    """, (os.getenv('POSTGRES_DB', 'spin_torque_db'),))
    
    result = cursor.fetchone()
    active_connections = result[0]
    longest_query = result[1] or 0
    
    # Check for long-running queries
    cursor.execute("""
        SELECT pid, query, EXTRACT(EPOCH FROM (now() - query_start)) as duration
        FROM pg_stat_activity 
        WHERE state = 'active' 
        AND EXTRACT(EPOCH FROM (now() - query_start)) > 300
        AND datname = %s
    """, (os.getenv('POSTGRES_DB', 'spin_torque_db'),))
    
    long_queries = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    # Log results
    logger.info(f"Active connections: {active_connections}")
    logger.info(f"Longest query duration: {longest_query:.2f} seconds")
    
    if long_queries:
        logger.warning(f"Found {len(long_queries)} long-running queries")
        for pid, query, duration in long_queries:
            logger.warning(f"PID {pid}: {duration:.2f}s - {query[:100]}...")
    
    # Return health status
    return {
        'healthy': active_connections < 50 and longest_query < 600,
        'active_connections': active_connections,
        'longest_query_seconds': longest_query,
        'long_queries_count': len(long_queries)
    }

if __name__ == '__main__':
    try:
        health = check_database_connections()
        if health['healthy']:
            logger.info("Database health check passed")
            sys.exit(0)
        else:
            logger.error("Database health check failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Database health check error: {e}")
        sys.exit(1)
```

## Monitoring Integration

### Prometheus Targets

```yaml
# prometheus/targets.yml
- targets:
  - 'app:8080'
  labels:
    service: 'spin-torque-app'
    environment: 'production'
    health_endpoint: '/health'

- targets:
  - 'postgres:5432'
  labels:
    service: 'postgresql'
    environment: 'production'
    health_endpoint: '/health'  # Via postgres_exporter

- targets:
  - 'redis:6379'
  labels:
    service: 'redis'
    environment: 'production'
    health_endpoint: '/health'  # Via redis_exporter
```

### Grafana Health Dashboard

```json
{
  "dashboard": {
    "title": "Health Checks Dashboard",
    "panels": [
      {
        "title": "Service Health Status",
        "type": "stat",
        "targets": [{
          "expr": "up{job=\"spin-torque-app\"}",
          "legendFormat": "Application"
        }, {
          "expr": "up{job=\"postgresql\"}",
          "legendFormat": "Database"
        }, {
          "expr": "up{job=\"redis\"}",
          "legendFormat": "Cache"
        }],
        "fieldConfig": {
          "defaults": {
            "mappings": [{
              "options": {
                "0": {"text": "DOWN", "color": "red"},
                "1": {"text": "UP", "color": "green"}
              },
              "type": "value"
            }]
          }
        }
      },
      {
        "title": "Health Check Response Times",
        "type": "graph",
        "targets": [{
          "expr": "health_check_duration_seconds",
          "legendFormat": "{{endpoint}}"
        }]
      },
      {
        "title": "Health Check Success Rate",
        "type": "graph",
        "targets": [{
          "expr": "rate(health_checks_total{status=\"success\"}[5m]) / rate(health_checks_total[5m])",
          "legendFormat": "Success Rate"
        }]
      }
    ]
  }
}
```

This comprehensive health check system provides multiple layers of monitoring, from basic connectivity tests to deep application-specific health validation, ensuring robust observability of the entire system.