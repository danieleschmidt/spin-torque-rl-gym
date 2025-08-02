# Monitoring & Observability Guide

This document covers monitoring, observability, and alerting strategies for Spin-Torque RL-Gym.

## Overview

The monitoring stack provides comprehensive observability into:

- **Application Performance**: Response times, throughput, errors
- **System Resources**: CPU, memory, disk, network usage
- **Training Metrics**: Model performance, convergence, experiments
- **Business Metrics**: Success rates, energy efficiency, switching times
- **Infrastructure Health**: Database, caching, containers

## Monitoring Stack

### Core Components

1. **Prometheus**: Metrics collection and time-series database
2. **Grafana**: Visualization and dashboards
3. **MLflow**: Experiment tracking and model management
4. **Application Metrics**: Custom application metrics
5. **Health Checks**: Service health monitoring
6. **Log Aggregation**: Centralized logging with Fluentd

### Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Application │───▶│ Prometheus  │───▶│   Grafana   │
│   Metrics   │    │   Server    │    │ Dashboards  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   MLflow    │    │ PostgreSQL  │    │   Alerts    │
│  Tracking   │    │  Metrics    │    │   & Notifications
└─────────────┘    └─────────────┘    └─────────────┘
```

## Application Metrics

### Key Performance Indicators (KPIs)

#### Training Performance
- **Success Rate**: Percentage of successful switching attempts
- **Energy Efficiency**: Energy per successful switch (pJ)
- **Switching Time**: Time to complete magnetization switching (ns)
- **Convergence Rate**: Training episodes to convergence
- **Model Accuracy**: Validation accuracy on test cases

#### System Performance
- **Simulation Speed**: Physics simulation steps per second
- **Memory Usage**: RAM consumption during training
- **CPU Utilization**: Processor usage patterns
- **GPU Utilization**: GPU usage when available
- **Disk I/O**: Data read/write patterns

#### Infrastructure Metrics
- **Response Time**: API response latencies
- **Throughput**: Requests per second
- **Error Rate**: Percentage of failed requests
- **Database Performance**: Query execution times
- **Cache Hit Rate**: Redis cache effectiveness

### Custom Metrics Implementation

```python
# Example application metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Training metrics
training_episodes = Counter('training_episodes_total', 'Total training episodes')
success_rate = Gauge('success_rate', 'Current success rate')
energy_efficiency = Histogram('energy_per_switch_joules', 'Energy per switch in Joules')
switching_time = Histogram('switching_time_seconds', 'Switching time in seconds')

# System metrics
simulation_speed = Gauge('simulation_steps_per_second', 'Physics simulation speed')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage')

# Business metrics
experiment_count = Counter('experiments_total', 'Total experiments run')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
```

## Health Checks

### Application Health Endpoints

```python
# Health check implementation
@app.route('/health')
def health_check():
    """Basic health check endpoint."""
    try:
        # Check database connection
        db_status = check_database_connection()
        
        # Check Redis connection
        cache_status = check_cache_connection()
        
        # Check GPU availability (if enabled)
        gpu_status = check_gpu_availability()
        
        # Check disk space
        disk_status = check_disk_space()
        
        overall_status = all([db_status, cache_status, disk_status])
        
        return {
            'status': 'healthy' if overall_status else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {
                'database': 'ok' if db_status else 'failed',
                'cache': 'ok' if cache_status else 'failed',
                'gpu': 'ok' if gpu_status else 'unavailable',
                'disk': 'ok' if disk_status else 'low_space'
            }
        }, 200 if overall_status else 503
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 503

@app.route('/ready')
def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if application is ready to serve requests
    return {'status': 'ready'}, 200

@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return generate_latest()
```

### Kubernetes Health Checks

```yaml
# Kubernetes deployment with health checks
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spin-torque-rl-gym
spec:
  template:
    spec:
      containers:
      - name: app
        image: spin-torque-rl-gym:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
```

## Prometheus Configuration

### Prometheus Configuration File

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Application metrics
  - job_name: 'spin-torque-app'
    static_configs:
      - targets: ['app:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  # PostgreSQL metrics
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres:5432']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  # Node exporter (system metrics)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # cAdvisor (container metrics)
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Alert Rules

```yaml
# prometheus/alert_rules.yml
groups:
  - name: application_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected
          description: "Error rate is {{ $value }} errors per second"

      - alert: LowSuccessRate
        expr: success_rate < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Low success rate in training
          description: "Success rate is {{ $value }}, below 80% threshold"

      - alert: HighMemoryUsage
        expr: memory_usage_bytes / (1024^3) > 6
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High memory usage
          description: "Memory usage is {{ $value }}GB, above 6GB threshold"

      - alert: SlowSimulation
        expr: simulation_steps_per_second < 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Slow physics simulation
          description: "Simulation speed is {{ $value }} steps/sec, below 100 threshold"

  - name: infrastructure_alerts
    rules:
      - alert: DatabaseDown
        expr: up{job="postgresql"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: PostgreSQL database is down
          description: "PostgreSQL has been down for more than 1 minute"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: Redis cache is down
          description: "Redis has been down for more than 1 minute"

      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High disk usage
          description: "Disk usage is above 90%"
```

## Grafana Dashboards

### Main Application Dashboard

```json
{
  "dashboard": {
    "title": "Spin-Torque RL-Gym - Overview",
    "panels": [
      {
        "title": "Success Rate",
        "type": "stat",
        "targets": [{
          "expr": "success_rate",
          "legendFormat": "Success Rate"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        }
      },
      {
        "title": "Energy Efficiency",
        "type": "stat",
        "targets": [{
          "expr": "rate(energy_per_switch_joules_sum[5m]) / rate(energy_per_switch_joules_count[5m])",
          "legendFormat": "Avg Energy per Switch"
        }],
        "fieldConfig": {
          "defaults": {
            "unit": "joule",
            "decimals": 2
          }
        }
      },
      {
        "title": "Switching Time Distribution",
        "type": "histogram",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(switching_time_seconds_bucket[5m]))",
          "legendFormat": "95th percentile"
        }, {
          "expr": "histogram_quantile(0.50, rate(switching_time_seconds_bucket[5m]))",
          "legendFormat": "50th percentile"
        }]
      },
      {
        "title": "Training Episodes",
        "type": "graph",
        "targets": [{
          "expr": "rate(training_episodes_total[5m])",
          "legendFormat": "Episodes per second"
        }]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [{
          "expr": "memory_usage_bytes / (1024^3)",
          "legendFormat": "Memory (GB)"
        }, {
          "expr": "rate(cpu_usage_seconds_total[5m]) * 100",
          "legendFormat": "CPU (%)"
        }]
      }
    ]
  }
}
```

### Training Performance Dashboard

```json
{
  "dashboard": {
    "title": "Training Performance",
    "panels": [
      {
        "title": "Model Convergence",
        "type": "graph",
        "targets": [{
          "expr": "model_accuracy",
          "legendFormat": "Validation Accuracy"
        }]
      },
      {
        "title": "Loss Function",
        "type": "graph",
        "targets": [{
          "expr": "training_loss",
          "legendFormat": "Training Loss"
        }, {
          "expr": "validation_loss",
          "legendFormat": "Validation Loss"
        }]
      },
      {
        "title": "Learning Rate",
        "type": "graph",
        "targets": [{
          "expr": "learning_rate",
          "legendFormat": "Current Learning Rate"
        }]
      }
    ]
  }
}
```

## Log Management

### Log Aggregation with Fluentd

```ruby
# fluentd/fluent.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

# Application logs
<source>
  @type tail
  path /var/log/app/*.log
  pos_file /var/log/fluentd/app.log.pos
  tag app.*
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%NZ
</source>

# PostgreSQL logs
<source>
  @type tail
  path /var/log/postgresql/*.log
  pos_file /var/log/fluentd/postgresql.log.pos
  tag postgresql.*
  format /^(?<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{3} \w+) \[(?<pid>\d+)\] (?<level>\w+): (?<message>.*)/
</source>

# Nginx access logs
<source>
  @type tail
  path /var/log/nginx/access.log
  pos_file /var/log/fluentd/nginx.log.pos
  tag nginx.access
  format nginx
</source>

# Output to Elasticsearch (optional)
<match **>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name spin-torque-logs
  type_name _doc
  include_tag_key true
  tag_key @log_name
  flush_interval 10s
</match>
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'experiment_id'):
            log_entry['experiment_id'] = record.experiment_id
        if hasattr(record, 'duration'):
            log_entry['duration'] = record.duration
            
        return json.dumps(log_entry)

# Configure logger
logger = logging.getLogger('spin_torque_gym')
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Training started", extra={
    'experiment_id': 'exp_001',
    'device_type': 'stt_mram',
    'algorithm': 'PPO'
})
```

## MLflow Integration

### Experiment Tracking

```python
import mlflow
import mlflow.pytorch

# Start MLflow run
with mlflow.start_run(run_name="stt_mram_training"):
    # Log parameters
    mlflow.log_param("device_type", "stt_mram")
    mlflow.log_param("algorithm", "PPO")
    mlflow.log_param("learning_rate", 3e-4)
    mlflow.log_param("batch_size", 64)
    
    # Log metrics during training
    for episode in range(num_episodes):
        # Training step
        reward, success_rate, energy = train_step()
        
        # Log metrics
        mlflow.log_metric("reward", reward, step=episode)
        mlflow.log_metric("success_rate", success_rate, step=episode)
        mlflow.log_metric("energy_efficiency", energy, step=episode)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("training_plot.png")
```

### Model Registry

```python
# Register model
model_uri = f"runs:/{run.info.run_id}/model"
model_name = "stt-mram-controller"

# Register model version
result = mlflow.register_model(model_uri, model_name)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production"
)
```

## Alerting and Notifications

### Alertmanager Configuration

```yaml
# alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@spinrl.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'admin@spinrl.com'
        subject: 'Spin-Torque RL-Gym Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: 'Spin-Torque RL-Gym Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
```

### Custom Alert Handlers

```python
# Custom alert webhook handler
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/webhook/alerts', methods=['POST'])
def handle_alert():
    alert_data = request.json
    
    for alert in alert_data.get('alerts', []):
        if alert['status'] == 'firing':
            handle_firing_alert(alert)
        elif alert['status'] == 'resolved':
            handle_resolved_alert(alert)
    
    return {'status': 'received'}, 200

def handle_firing_alert(alert):
    """Handle firing alert."""
    alertname = alert['labels']['alertname']
    severity = alert['labels']['severity']
    
    if severity == 'critical':
        # Send to PagerDuty or similar
        notify_oncall_engineer(alert)
    
    # Log alert
    logger.error(f"Alert fired: {alertname}", extra={
        'alert': alert,
        'severity': severity
    })

def handle_resolved_alert(alert):
    """Handle resolved alert."""
    logger.info(f"Alert resolved: {alert['labels']['alertname']}")
```

## Performance Monitoring

### Application Performance Monitoring (APM)

```python
# OpenTelemetry integration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument Flask app
FlaskInstrumentor().instrument_app(app)

# Custom tracing
@tracer.start_as_current_span("train_model")
def train_model():
    with tracer.start_as_current_span("physics_simulation"):
        # Physics simulation code
        pass
    
    with tracer.start_as_current_span("reward_calculation"):
        # Reward calculation code
        pass
```

### Resource Monitoring

```bash
#!/bin/bash
# System resource monitoring script

# CPU usage
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')

# Memory usage
mem_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')

# Disk usage
disk_usage=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')

# GPU usage (if available)
if command -v nvidia-smi &> /dev/null; then
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
else
    gpu_usage="N/A"
fi

# Send metrics to Prometheus pushgateway
cat <<EOF | curl --data-binary @- http://pushgateway:9091/metrics/job/system_monitor
# TYPE cpu_usage gauge
cpu_usage $cpu_usage
# TYPE memory_usage gauge
memory_usage $mem_usage
# TYPE disk_usage gauge
disk_usage $disk_usage
# TYPE gpu_usage gauge
gpu_usage $gpu_usage
EOF
```

## Maintenance and Operations

### Backup and Recovery

```bash
#!/bin/bash
# Backup script for monitoring data

# Backup Prometheus data
docker run --rm -v prometheus-data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz /data

# Backup Grafana dashboards
docker exec grafana-container grafana-cli admin export-dashboard > grafana-dashboards-$(date +%Y%m%d).json

# Backup MLflow data
docker exec mlflow-container cp -r /mlflow/artifacts /backup/mlflow-artifacts-$(date +%Y%m%d)
```

### Monitoring the Monitoring

```yaml
# Monitor Prometheus itself
- alert: PrometheusDown
  expr: up{job="prometheus"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: Prometheus server is down

- alert: GrafanaDown
  expr: up{job="grafana"} == 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: Grafana server is down
```

## Best Practices

### Metrics Design
- Use consistent naming conventions
- Include relevant labels for dimensionality
- Avoid high-cardinality metrics
- Use appropriate metric types (counter, gauge, histogram, summary)

### Dashboard Design
- Focus on key business metrics
- Use appropriate visualizations
- Include context and annotations
- Design for different audiences (dev, ops, business)

### Alerting
- Alert on symptoms, not causes
- Avoid alert fatigue with proper thresholds
- Include actionable information in alerts
- Test alert channels regularly

### Performance
- Monitor query performance
- Use recording rules for expensive queries
- Implement proper retention policies
- Regular maintenance and cleanup