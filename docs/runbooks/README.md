# Operational Runbooks

This directory contains operational runbooks for common scenarios and troubleshooting procedures.

## Quick Reference

### Emergency Contacts
- **On-call Engineer**: [Contact Information]
- **Platform Team**: [Contact Information]  
- **Database Administrator**: [Contact Information]

### System Access
- **Production Environment**: [Access Instructions]
- **Monitoring Dashboards**: http://grafana.spinrl.com
- **Log Aggregation**: http://logs.spinrl.com
- **MLflow Tracking**: http://mlflow.spinrl.com

### Common Commands

```bash
# Check application status
docker-compose ps

# View application logs
docker-compose logs -f app

# Check system resources
docker stats

# Database connection test
docker-compose exec app python -c "import psycopg2; print('DB OK')"

# Redis connection test
docker-compose exec app python -c "import redis; print('Redis OK')"
```

## Incident Response

### Severity Levels

#### Critical (P0)
- **Response Time**: 15 minutes
- **Examples**: Complete service outage, data loss, security breach
- **Actions**: Page on-call, engage leadership, start incident bridge

#### High (P1)
- **Response Time**: 1 hour
- **Examples**: Degraded performance, partial outage, failed deployments
- **Actions**: Alert on-call, investigate, provide updates

#### Medium (P2)
- **Response Time**: 4 hours
- **Examples**: Minor feature issues, non-critical alerts
- **Actions**: Create ticket, investigate during business hours

#### Low (P3)
- **Response Time**: 24 hours
- **Examples**: Documentation issues, nice-to-have improvements
- **Actions**: Create ticket, prioritize in backlog

### Incident Response Process

1. **Detection**: Alert fired or issue reported
2. **Assessment**: Determine impact and severity
3. **Response**: Begin mitigation efforts
4. **Communication**: Update stakeholders
5. **Resolution**: Fix root cause
6. **Post-mortem**: Document lessons learned

## Common Scenarios

### Application Not Starting

#### Symptoms
- Container exits immediately
- Health checks failing
- Cannot connect to application

#### Investigation Steps

```bash
# Check container status
docker-compose ps

# Check container logs
docker-compose logs app

# Check resource usage
docker stats

# Verify environment variables
docker-compose exec app env | grep -E "(POSTGRES|REDIS|DEBUG)"

# Test dependencies
docker-compose exec app python -c "
import psycopg2
import redis
print('Dependencies OK')
"
```

#### Common Causes and Solutions

1. **Missing Environment Variables**
   ```bash
   # Check .env file exists and is properly formatted
   cat .env
   
   # Verify variables are loaded
   docker-compose config
   ```

2. **Database Connection Issues**
   ```bash
   # Check database is running
   docker-compose ps postgres
   
   # Test connection manually
   docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "SELECT 1;"
   
   # Check network connectivity
   docker-compose exec app ping postgres
   ```

3. **Permission Issues**
   ```bash
   # Check file permissions
   ls -la /app/
   
   # Fix ownership if needed
   docker-compose exec app chown -R appuser:appuser /app/
   ```

### High Memory Usage

#### Symptoms
- Memory alerts firing
- Application becoming slow
- Out of memory errors

#### Investigation Steps

```bash
# Check memory usage by container
docker stats --no-stream

# Check application memory usage
docker-compose exec app python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.2f} GB')
"

# Check for memory leaks
docker-compose exec app python -c "
import gc
print(f'Objects: {len(gc.get_objects())}')
gc.collect()
print(f'After GC: {len(gc.get_objects())}')
"
```

#### Solutions

1. **Increase Memory Limits**
   ```yaml
   # In docker-compose.yml
   services:
     app:
       deploy:
         resources:
           limits:
             memory: 8G
   ```

2. **Optimize Application**
   ```python
   # Enable garbage collection
   import gc
   gc.set_threshold(700, 10, 10)
   
   # Use memory profiling
   from memory_profiler import profile
   
   @profile
   def train_model():
       # Your training code
       pass
   ```

3. **Implement Memory Monitoring**
   ```python
   import psutil
   import logging
   
   def log_memory_usage():
       mem = psutil.virtual_memory()
       if mem.percent > 80:
           logging.warning(f"High memory usage: {mem.percent}%")
   ```

### Database Performance Issues

#### Symptoms
- Slow query performance
- Connection timeouts
- High database CPU usage

#### Investigation Steps

```bash
# Check database connections
docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "
SELECT COUNT(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';
"

# Check slow queries
docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Check database size
docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "
SELECT pg_size_pretty(pg_database_size('spin_torque_db'));
"

# Check table sizes
docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"
```

#### Solutions

1. **Optimize Queries**
   ```sql
   -- Add missing indexes
   CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
   CREATE INDEX idx_experiments_status ON experiments(status);
   
   -- Analyze query plans
   EXPLAIN ANALYZE SELECT * FROM metrics WHERE timestamp > NOW() - INTERVAL '1 hour';
   ```

2. **Connection Pooling**
   ```python
   from sqlalchemy import create_engine
   from sqlalchemy.pool import QueuePool
   
   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=20,
       max_overflow=30,
       pool_pre_ping=True
   )
   ```

3. **Database Maintenance**
   ```bash
   # Vacuum and analyze
   docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "VACUUM ANALYZE;"
   
   # Update statistics
   docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "ANALYZE;"
   ```

### Training Performance Degradation

#### Symptoms
- Slow convergence
- Low success rates
- High energy consumption

#### Investigation Steps

```bash
# Check GPU utilization
nvidia-smi

# Monitor training metrics
curl http://localhost:8080/metrics | grep -E "(success_rate|energy_efficiency|simulation_speed)"

# Check MLflow experiments
mlflow ui --backend-store-uri postgresql://...

# Analyze recent experiments
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
for exp in experiments[-5:]:
    runs = client.search_runs(exp.experiment_id)
    print(f'Experiment: {exp.name}, Runs: {len(runs)}')
"
```

#### Solutions

1. **Hyperparameter Tuning**
   ```python
   # Implement learning rate scheduling
   from torch.optim.lr_scheduler import StepLR
   
   scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)
   
   # Add early stopping
   class EarlyStopping:
       def __init__(self, patience=10, min_delta=0.001):
           self.patience = patience
           self.min_delta = min_delta
           self.counter = 0
           self.best_score = None
       
       def __call__(self, score):
           if self.best_score is None:
               self.best_score = score
           elif score < self.best_score + self.min_delta:
               self.counter += 1
               if self.counter >= self.patience:
                   return True
           else:
               self.best_score = score
               self.counter = 0
           return False
   ```

2. **Model Architecture Changes**
   ```python
   # Add regularization
   import torch.nn as nn
   
   class ImprovedPolicy(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(input_size, 256)
           self.dropout1 = nn.Dropout(0.2)
           self.fc2 = nn.Linear(256, 128)
           self.dropout2 = nn.Dropout(0.2)
           self.fc3 = nn.Linear(128, output_size)
           
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = self.dropout1(x)
           x = torch.relu(self.fc2(x))
           x = self.dropout2(x)
           return self.fc3(x)
   ```

3. **Data Pipeline Optimization**
   ```python
   # Implement data caching
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_device_config(device_type):
       # Expensive device configuration loading
       return config
   
   # Use multiprocessing for data generation
   from multiprocessing import Pool
   
   with Pool(processes=4) as pool:
       results = pool.map(generate_training_data, range(batch_size))
   ```

### Deployment Failures

#### Symptoms
- Containers fail to start after deployment
- Health checks failing
- Version rollback required

#### Investigation Steps

```bash
# Check deployment status
docker-compose ps

# Compare with previous version
docker images | grep spin-torque-rl-gym

# Check configuration changes
git diff HEAD~1 docker-compose.yml

# Verify image build
docker build --target production -t test-image .
docker run --rm test-image python -c "import spin_torque_gym; print('OK')"
```

#### Solutions

1. **Rollback Procedure**
   ```bash
   # Quick rollback
   docker-compose down
   docker tag spin-torque-rl-gym:previous spin-torque-rl-gym:latest
   docker-compose up -d
   
   # Verify rollback
   docker-compose ps
   curl http://localhost:8080/health
   ```

2. **Blue-Green Deployment**
   ```bash
   # Deploy to green environment
   docker-compose -f docker-compose.green.yml up -d
   
   # Test green environment
   curl http://green.spinrl.com/health
   
   # Switch traffic
   # Update load balancer configuration
   
   # Stop blue environment
   docker-compose -f docker-compose.blue.yml down
   ```

3. **Database Migration Issues**
   ```bash
   # Check migration status
   docker-compose exec app python -m spin_torque_gym.db.migration status
   
   # Rollback migrations if needed
   docker-compose exec app python -m spin_torque_gym.db.migration downgrade
   
   # Re-run migrations
   docker-compose exec app python -m spin_torque_gym.db.migration upgrade
   ```

## Monitoring and Alerting

### Key Metrics to Watch

1. **Application Health**
   - Response time < 500ms
   - Error rate < 1%
   - Success rate > 90%

2. **System Resources**
   - CPU usage < 80%
   - Memory usage < 80%
   - Disk usage < 90%

3. **Dependencies**
   - Database connections < 80% of limit
   - Redis memory usage < 90%
   - External API response times

### Alert Fatigue Prevention

1. **Proper Thresholds**
   ```yaml
   # Good: Context-aware thresholds
   - alert: HighErrorRate
     expr: rate(errors_total[5m]) / rate(requests_total[5m]) > 0.05
     for: 5m  # Avoid transient spikes
   
   # Bad: Static thresholds
   - alert: HighCPU
     expr: cpu_usage > 50  # Too sensitive
   ```

2. **Alert Grouping**
   ```yaml
   route:
     group_by: ['service', 'severity']
     group_wait: 30s
     group_interval: 5m
     repeat_interval: 4h
   ```

3. **Maintenance Windows**
   ```bash
   # Silence alerts during maintenance
   amtool silence add alertname=".*" --duration=2h --comment="Scheduled maintenance"
   ```

## Backup and Recovery

### Backup Procedures

```bash
#!/bin/bash
# Daily backup script

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/$BACKUP_DATE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
docker-compose exec postgres pg_dump -U spin_torque_user spin_torque_db | gzip > "$BACKUP_DIR/database.sql.gz"

# Backup volumes
docker run --rm -v app-data:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/app-data.tar.gz -C /data .
docker run --rm -v app-models:/data -v "$BACKUP_DIR":/backup alpine tar czf /backup/models.tar.gz -C /data .

# Backup configuration
cp -r docker/ "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/env.backup"

# Upload to S3 (optional)
aws s3 sync "$BACKUP_DIR" s3://spinrl-backups/$BACKUP_DATE/

# Clean old backups (keep 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} \;
```

### Recovery Procedures

```bash
#!/bin/bash
# Recovery script

BACKUP_DATE=$1
BACKUP_DIR="/backups/$BACKUP_DATE"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Available backups:"
    ls -1 /backups/
    exit 1
fi

# Stop services
docker-compose down

# Restore database
gunzip -c "$BACKUP_DIR/database.sql.gz" | docker-compose exec -T postgres psql -U spin_torque_user spin_torque_db

# Restore volumes
docker run --rm -v app-data:/data -v "$BACKUP_DIR":/backup alpine tar xzf /backup/app-data.tar.gz -C /data
docker run --rm -v app-models:/data -v "$BACKUP_DIR":/backup alpine tar xzf /backup/models.tar.gz -C /data

# Restore configuration
cp "$BACKUP_DIR/docker-compose.yml" .
cp "$BACKUP_DIR/env.backup" .env

# Start services
docker-compose up -d

# Verify recovery
sleep 30
curl http://localhost:8080/health
```

## Security Incident Response

### Suspected Security Breach

1. **Immediate Actions**
   ```bash
   # Isolate affected systems
   docker-compose down
   
   # Preserve evidence
   docker logs app > incident-logs-$(date +%Y%m%d-%H%M).txt
   
   # Block suspicious IPs (if using nginx)
   echo "deny 192.168.1.100;" >> nginx/conf.d/security.conf
   docker-compose restart nginx
   ```

2. **Investigation**
   ```bash
   # Check access logs
   grep -E "(404|401|403)" /var/log/nginx/access.log
   
   # Check failed login attempts
   docker-compose exec postgres psql -U spin_torque_user -d spin_torque_db -c "
   SELECT * FROM auth_logs WHERE status = 'failed' AND timestamp > NOW() - INTERVAL '24 hours';
   "
   
   # Check file integrity
   find /app -type f -newer /tmp/baseline -ls
   ```

3. **Containment and Recovery**
   ```bash
   # Update passwords
   docker-compose exec postgres psql -U postgres -c "ALTER USER spin_torque_user PASSWORD 'new_secure_password';"
   
   # Rotate API keys
   # Update .env with new credentials
   
   # Audit user accounts
   # Review and revoke suspicious sessions
   
   # Update security patches
   docker-compose pull
   docker-compose up -d --force-recreate
   ```

## Contact Information

### Escalation Matrix

| Severity | Primary Contact | Secondary Contact | Escalation |
|----------|----------------|-------------------|------------|
| P0       | On-call Engineer | Platform Lead | VP Engineering |
| P1       | On-call Engineer | Team Lead | Platform Lead |
| P2       | Team Member | Team Lead | - |
| P3       | Team Member | - | - |

### External Contacts

- **Cloud Provider Support**: [Contact Information]
- **Database Support**: [Contact Information]
- **Security Team**: [Contact Information]
- **Legal/Compliance**: [Contact Information]

## Post-Incident Procedures

### Post-Mortem Template

1. **Incident Summary**
   - What happened?
   - When did it occur?
   - How long did it last?
   - What was the impact?

2. **Timeline**
   - Detection time
   - Response time
   - Resolution time
   - Key actions taken

3. **Root Cause Analysis**
   - Primary cause
   - Contributing factors
   - Why wasn't it caught earlier?

4. **Action Items**
   - Immediate fixes
   - Long-term improvements
   - Process changes
   - Monitoring enhancements

5. **Lessons Learned**
   - What went well?
   - What could be improved?
   - Knowledge gaps identified

### Continuous Improvement

1. **Regular Reviews**
   - Monthly incident review meetings
   - Quarterly runbook updates
   - Annual disaster recovery tests

2. **Training**
   - New team member onboarding
   - Regular incident response drills
   - Knowledge sharing sessions

3. **Documentation**
   - Keep runbooks updated
   - Document new procedures
   - Share lessons learned