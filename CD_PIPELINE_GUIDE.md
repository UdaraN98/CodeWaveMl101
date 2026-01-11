# CD Pipeline Options for Customer Churn Prediction API

## 🚀 Deployment Options Available

### Option 1: Containerized Deployment (Current Setup)
**Files Created:**
- `Dockerfile` - Container definition for the API
- `.github/workflows/cd.yml` - GitHub Actions CD pipeline
- `docker-compose.yml` - Local development and deployment
- `Scripts/deploy.sh` - Manual deployment script

**Features:**
- ✅ Docker containerization
- ✅ Multi-environment support (local, staging, production)
- ✅ Automated testing in pipeline
- ✅ Health checks and verification
- ✅ Self-hosted runner compatible

### Option 2: Cloud Platform Deployment

#### AWS Deployment Options:
1. **AWS ECS Fargate** - Serverless containers
2. **AWS Lambda + API Gateway** - Serverless functions
3. **AWS EC2** - Traditional VMs
4. **AWS SageMaker Endpoints** - ML-specific hosting

#### Google Cloud Platform:
1. **Cloud Run** - Serverless containers
2. **Google Kubernetes Engine (GKE)** - Container orchestration
3. **Compute Engine** - VMs

#### Microsoft Azure:
1. **Azure Container Instances** - Simple containers
2. **Azure Kubernetes Service** - Container orchestration
3. **Azure Functions** - Serverless

## 🎯 Current CD Pipeline Features

### 1. **Multi-Environment Support**
```bash
# Deploy to local
./Scripts/deploy.sh

# Deploy to staging  
./Scripts/deploy.sh -e staging -p 8001

# Deploy to production
./Scripts/deploy.sh -e production -p 80
```

### 2. **GitHub Actions Workflow**
- **Manual Trigger**: Use workflow_dispatch with environment selection
- **Automatic**: Triggers on pushes to main/week-5-ci-cd-final
- **Dependency Check**: Ensures CI has run and model artifacts exist

### 3. **Testing & Verification**
- Container health checks
- API endpoint testing
- Model loading verification
- Extended post-deployment verification

### 4. **Docker Compose for Development**
```bash
# Start API only
docker-compose up customer-churn-api

# Start with MLflow server
docker-compose --profile mlflow up
```

## 🚦 How to Use

### Quick Start (Local Deployment)
1. Ensure your CI pipeline has run and model artifacts exist
2. Run: `./Scripts/deploy.sh`
3. API will be available at `http://localhost:8000`

### GitHub Actions Deployment
1. Go to Actions tab in GitHub
2. Select "Deploy Model API" workflow
3. Click "Run workflow"
4. Choose your deployment target (local/staging/production)

### Environment URLs
- **Local**: `http://localhost:8000`
- **Staging**: `http://localhost:8001`  
- **Production**: Configurable

## 🔧 Configuration

### Environment Variables
- `BEST_MODEL_METRIC`: Metric to use for best model selection (default: f1)
- `MLFLOW_EXPERIMENT_NAME`: MLflow experiment name
- `BEST_MODEL_DIR`: Override for model directory path
- `LOCAL_DATA_SOURCE`: Path to training data

### Ports
- Local: 8000
- Staging: 8001
- Production: 80 (configurable)

## 🔍 Monitoring & Health Checks

### API Endpoints
- `/health` - Service health status
- `/schema` - Model input schema
- `/predict` - Make predictions
- `/docs` - Interactive API documentation

### Docker Health Checks
- Built-in container health monitoring
- Automatic restart on failure
- Comprehensive logging

## 🚀 Next Steps for Production

### Security Considerations
1. **Authentication**: Add API key or OAuth
2. **HTTPS**: SSL/TLS termination
3. **Rate Limiting**: Prevent API abuse
4. **Input Validation**: Enhanced request validation

### Scalability
1. **Load Balancer**: Multiple container instances
2. **Auto-scaling**: Based on CPU/memory usage
3. **Database**: Persistent logging and metrics
4. **Caching**: Redis for frequent predictions

### Monitoring
1. **Application Metrics**: Response time, error rates
2. **Model Metrics**: Prediction accuracy, drift detection
3. **Infrastructure**: CPU, memory, disk usage
4. **Alerting**: Email/Slack notifications

### Cloud Migration
If you want to deploy to cloud platforms, I can help you create:
- AWS ECS/Fargate deployment
- Google Cloud Run configuration
- Kubernetes manifests
- Terraform infrastructure as code

## 📋 Commands Reference

### Manual Deployment
```bash
# Basic local deployment
./Scripts/deploy.sh

# Staging deployment
./Scripts/deploy.sh -e staging -p 8001

# Production deployment with custom port
./Scripts/deploy.sh -e production -p 80

# Quick deployment (skip build and tests)
./Scripts/deploy.sh --no-build --skip-tests
```

### Docker Commands
```bash
# Build image manually
docker build -t customer-churn-api .

# Run container manually
docker run -d --name api -p 8000:8000 customer-churn-api

# View logs
docker logs customer-churn-api-local

# Stop and remove
docker stop customer-churn-api-local
docker rm customer-churn-api-local
```

### Testing Commands
```bash
# Health check
curl http://localhost:8000/health

# Get model schema
curl http://localhost:8000/schema

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "tenure": 12, ...}}'
```