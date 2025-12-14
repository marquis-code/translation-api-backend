# Production Deployment Guide

This guide covers deploying the Medical Transcription System to production environments.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Security Considerations](#security-considerations)
- [Monitoring & Logging](#monitoring--logging)
- [Backup & Recovery](#backup--recovery)

## Architecture Overview

### Production Architecture

```
                    ┌──────────────┐
                    │   CDN/WAF    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Load Balancer│
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐       ┌────▼────┐
   │Frontend │        │Frontend │       │Frontend │
   │ Server  │        │ Server  │       │ Server  │
   └────┬────┘        └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼───────┐
                    │ API Gateway  │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐       ┌────▼────┐
   │ Backend │        │ Backend │       │ Backend │
   │  API    │        │  API    │       │  API    │
   └────┬────┘        └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐       ┌────▼────────┐
   │Database │        │  Cache  │       │ File Storage│
   │(Primary)│        │ (Redis) │       │    (S3)     │
   └────┬────┘        └─────────┘       └─────────────┘
        │
   ┌────▼────┐
   │Database │
   │(Replica)│
   └─────────┘
```

## Prerequisites

### Production Requirements

- **Database**: PostgreSQL 13+ or MongoDB 4.4+
- **Cache**: Redis 6+
- **Container Orchestration**: Kubernetes 1.20+ or Docker Swarm
- **SSL Certificates**: Valid SSL/TLS certificates
- **Domain**: Registered domain name
- **Monitoring**: Prometheus + Grafana or DataDog
- **Logging**: ELK Stack or CloudWatch

### Third-Party Services

- **AssemblyAI**: Production API key with appropriate limits
- **Email Service**: SendGrid/AWS SES for notifications
- **Cloud Storage**: AWS S3/Azure Blob for audio files
- **CDN**: CloudFlare/AWS CloudFront

## Deployment Options

### Option 1: Docker Compose (Small Scale)

**Best for**: Single server, development staging, small clinics (1-10 doctors)

**Pros**: Simple setup, easy management, low cost
**Cons**: Limited scalability, single point of failure

### Option 2: Kubernetes (Medium-Large Scale)

**Best for**: Multiple servers, 10+ doctors, high availability needs

**Pros**: Auto-scaling, high availability, easy updates
**Cons**: Complex setup, higher cost, requires DevOps expertise

### Option 3: Serverless (Variable Load)

**Best for**: Unpredictable traffic, cost optimization

**Pros**: Pay per use, infinite scaling, no server management
**Cons**: Cold start latency, vendor lock-in, complex debugging

## Docker Deployment

### Step 1: Create Dockerfiles

**Backend Dockerfile:**

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend Dockerfile:**

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY . .

# Build production bundle
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Step 2: Docker Compose Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: medical_transcription
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:6-alpine
    restart: unless-stopped
    
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/medical_transcription
      - REDIS_URL=redis://redis:6379/0
      - ASSEMBLYAI_API_KEY=${ASSEMBLYAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
```

### Step 3: Deploy

```bash
# Set environment variables
cp .env.example .env
# Edit .env with production values

# Build and start services
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale backend
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

## Cloud Deployment

### AWS Deployment

#### Using Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 medical-transcription

# Create environment
eb create production-env

# Deploy
eb deploy

# Open application
eb open
```

#### Using ECS (Recommended)

1. **Create ECR Repositories**

```bash
aws ecr create-repository --repository-name medical-transcription-backend
aws ecr create-repository --repository-name medical-transcription-frontend
```

2. **Push Docker Images**

```bash
# Backend
docker build -t medical-transcription-backend ./backend
docker tag medical-transcription-backend:latest ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/medical-transcription-backend:latest
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/medical-transcription-backend:latest

# Frontend
docker build -t medical-transcription-frontend ./frontend
docker tag medical-transcription-frontend:latest ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/medical-transcription-frontend:latest
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/medical-transcription-frontend:latest
```

3. **Create ECS Task Definition**

4. **Create ECS Service with Load Balancer**

5. **Configure Auto Scaling**

### Google Cloud Platform

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/${PROJECT_ID}/backend
gcloud run deploy backend --image gcr.io/${PROJECT_ID}/backend --platform managed

gcloud builds submit --tag gcr.io/${PROJECT_ID}/frontend
gcloud run deploy frontend --image gcr.io/${PROJECT_ID}/frontend --platform managed
```

### Azure

```bash
# Create resources
az group create --name medical-transcription-rg --location eastus
az acr create --resource-group medical-transcription-rg --name medicaltranscriptionacr --sku Basic

# Deploy to App Service
az webapp create --resource-group medical-transcription-rg --plan myAppServicePlan --name medical-transcription-backend --deployment-container-image-name medicaltranscriptionacr.azurecr.io/backend:latest
```

## Security Considerations

### 1. Environment Variables

**Never commit secrets to Git!**

Use secret management:
- AWS Secrets Manager
- HashiCorp Vault
- Kubernetes Secrets
- Azure Key Vault

```python
# Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
```

### 2. HTTPS/TLS

**Required for HIPAA compliance**

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Database Encryption

```sql
-- Enable encryption at rest
ALTER DATABASE medical_transcription SET encrypt = ON;

-- Enable SSL connections
ALTER SYSTEM SET ssl = on;
```

### 4. API Rate Limiting

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.get("/api/consultations", dependencies=[Depends(RateLimiter(times=60, seconds=60))])
async def get_consultations():
    pass
```

### 5. Input Validation

```python
from pydantic import BaseModel, validator

class UserLogin(BaseModel):
    username: str
    password: str
    
    @validator('username')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
```

## Monitoring & Logging

### Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Application Metrics

```python
from prometheus_client import Counter, Histogram

transcription_counter = Counter('transcriptions_total', 'Total transcriptions')
transcription_duration = Histogram('transcription_duration_seconds', 'Transcription duration')

@app.post("/api/consultations/start")
async def start_consultation():
    transcription_counter.inc()
    with transcription_duration.time():
        # Process transcription
        pass
```

### Centralized Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler]
)
```

## Backup & Recovery

### Database Backups

```bash
# Automated daily backups
0 2 * * * pg_dump -U postgres medical_transcription | gzip > /backups/db_$(date +\%Y\%m\%d).sql.gz

# Keep last 30 days
find /backups -name "db_*.sql.gz" -mtime +30 -delete
```

### Restore Procedure

```bash
# Restore from backup
gunzip -c /backups/db_20240101.sql.gz | psql -U postgres medical_transcription
```

### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 1 hour
2. **RPO (Recovery Point Objective)**: 24 hours
3. **Backup Strategy**:
   - Daily full backups
   - Hourly incremental backups
   - Multi-region replication
   - Point-in-time recovery enabled

## Performance Optimization

### 1. Caching Strategy

```python
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache(expire=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### 2. Database Indexing

```sql
-- Add indexes for common queries
CREATE INDEX idx_consultations_doctor ON consultations(doctor_id);
CREATE INDEX idx_consultations_status ON consultations(status);
CREATE INDEX idx_consultations_created ON consultations(created_at);
```

### 3. CDN for Static Assets

Use CloudFlare, AWS CloudFront, or Azure CDN for:
- Frontend static files (JS, CSS, images)
- Recorded audio files
- Generated reports

## Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "dependencies": {
            "database": await check_database(),
            "redis": await check_redis(),
            "assemblyai": await check_assemblyai()
        }
    }
```

## Continuous Deployment

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and Push Docker Image
        run: |
          docker build -t backend ./backend
          docker push ${REGISTRY}/backend:latest
          
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f k8s/
          kubectl rollout status deployment/backend
```

## Cost Optimization

### Estimated Monthly Costs (AWS)

**Small Clinic (5 doctors, 50 consultations/day)**
- EC2 instances (t3.medium x2): $60
- RDS PostgreSQL (db.t3.small): $30
- ElastiCache Redis: $15
- S3 storage (100GB): $2
- AssemblyAI API: $150
- **Total: ~$257/month**

**Medium Clinic (20 doctors, 200 consultations/day)**
- EC2 instances (t3.large x4): $240
- RDS PostgreSQL (db.t3.medium): $60
- ElastiCache Redis: $30
- S3 storage (500GB): $12
- AssemblyAI API: $600
- Load Balancer: $20
- **Total: ~$962/month**

## Compliance Checklist

- [ ] HIPAA compliance audit completed
- [ ] Data encryption at rest enabled
- [ ] Data encryption in transit (TLS 1.2+)
- [ ] Access logs enabled and retained
- [ ] Backup and disaster recovery tested
- [ ] Security vulnerability scanning
- [ ] Penetration testing completed
- [ ] Privacy policy and terms of service
- [ ] User consent mechanisms
- [ ] Data retention policies implemented
- [ ] Audit trail for all data access

## Support & Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor system health
- Review error logs
- Check backup completion

**Weekly:**
- Review security logs
- Update dependencies
- Performance optimization

**Monthly:**
- Security patches
- Database optimization
- Cost review and optimization

**Quarterly:**
- Disaster recovery drill
- Security audit
- Feature prioritization

---

**Questions?** Contact DevOps team at devops@yourdomain.com