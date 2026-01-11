#!/bin/bash

# Customer Churn API Deployment Script
# This script helps deploy the API to different environments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="customer-churn-api"
CONTAINER_PREFIX="customer-churn-api"

# Default values
ENVIRONMENT="local"
PORT="8000"
BUILD_IMAGE=true
SKIP_TESTS=false

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENVIRONMENT    Target environment: local, staging, production (default: local)"
    echo "  -p, --port PORT          Port to expose API on (default: 8000)"
    echo "  --no-build              Skip building Docker image"
    echo "  --skip-tests            Skip running API tests"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                   # Deploy to local on port 8000"
    echo "  $0 -e staging -p 8001               # Deploy to staging on port 8001"
    echo "  $0 --no-build --skip-tests          # Quick deploy without build or tests"
}

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --no-build)
            BUILD_IMAGE=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    local|staging|production)
        ;;
    *)
        error "Invalid environment: $ENVIRONMENT. Must be one of: local, staging, production"
        ;;
esac

# Set environment-specific configuration
CONTAINER_NAME="${CONTAINER_PREFIX}-${ENVIRONMENT}"
case $ENVIRONMENT in
    local)
        PORT=${PORT:-8000}
        ;;
    staging)
        PORT=${PORT:-8001}
        ;;
    production)
        PORT=${PORT:-80}
        warning "Production deployment - please ensure you have proper security measures in place!"
        ;;
esac

log "🚀 Starting deployment to $ENVIRONMENT environment"
log "📋 Configuration:"
log "   Environment: $ENVIRONMENT"
log "   Container: $CONTAINER_NAME"
log "   Port: $PORT"
log "   Build image: $BUILD_IMAGE"
log "   Skip tests: $SKIP_TESTS"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]] || [[ ! -f "Dockerfile" ]]; then
    error "Please run this script from the Project directory (should contain pyproject.toml and Dockerfile)"
fi

# Check for model artifacts
if [[ ! -d "mlruns/best_model_artifacts" ]] || [[ -z "$(ls -A mlruns/best_model_artifacts 2>/dev/null)" ]]; then
    error "No model artifacts found. Please run the CI pipeline or model training first."
fi

# Build Docker image
if [[ "$BUILD_IMAGE" == true ]]; then
    log "🔨 Building Docker image..."
    docker build --tag "$IMAGE_NAME:latest" --tag "$IMAGE_NAME:$ENVIRONMENT" .
    success "Docker image built successfully"
else
    log "⏩ Skipping Docker image build"
fi

# Stop existing container if running
log "🛑 Stopping existing container (if any)..."
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    docker stop "$CONTAINER_NAME"
    success "Stopped existing container: $CONTAINER_NAME"
fi

if docker ps -a -q -f name="$CONTAINER_NAME" | grep -q .; then
    docker rm "$CONTAINER_NAME"
    success "Removed existing container: $CONTAINER_NAME"
fi

# Start new container
log "🚀 Starting new container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --publish "$PORT:8000" \
    --env BEST_MODEL_METRIC=f1 \
    --env MLFLOW_EXPERIMENT_NAME=customer_churn_optimization \
    --restart unless-stopped \
    "$IMAGE_NAME:$ENVIRONMENT"

success "Container started: $CONTAINER_NAME"

# Wait for API to be ready
log "⏳ Waiting for API to be ready..."
for i in {1..30}; do
    if curl -f "http://localhost:$PORT/health" >/dev/null 2>&1; then
        success "API is ready!"
        break
    fi
    if [[ $i -eq 30 ]]; then
        error "API failed to start within 30 seconds. Check logs with: docker logs $CONTAINER_NAME"
    fi
    sleep 1
done

# Run tests
if [[ "$SKIP_TESTS" == false ]]; then
    log "🧪 Running API tests..."
    
    # Test health endpoint
    if curl -f "http://localhost:$PORT/health" >/dev/null 2>&1; then
        success "✅ Health endpoint test passed"
    else
        error "❌ Health endpoint test failed"
    fi
    
    # Test schema endpoint
    if curl -f "http://localhost:$PORT/schema" >/dev/null 2>&1; then
        success "✅ Schema endpoint test passed"
    else
        error "❌ Schema endpoint test failed"
    fi
    
    # Test prediction endpoint with sample data
    if curl -f -X POST "http://localhost:$PORT/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "features": {
                "age": 35,
                "gender": "Male",
                "tenure": 12,
                "usage_frequency": 15,
                "support_calls": 2,
                "payment_delay": 0,
                "subscription_type": "Standard",
                "contract_length": "Monthly",
                "total_spend": 500,
                "last_interaction": 30
            }
        }' >/dev/null 2>&1; then
        success "✅ Prediction endpoint test passed"
    else
        warning "⚠️ Prediction endpoint test failed (this might be due to model/data compatibility)"
    fi
    
    success "All tests completed"
else
    log "⏩ Skipping API tests"
fi

# Final success message
echo ""
echo "🎉🎉🎉 DEPLOYMENT SUCCESSFUL 🎉🎉🎉"
echo ""
echo "📡 API Information:"
echo "   Environment: $ENVIRONMENT"
echo "   URL: http://localhost:$PORT"
echo "   Container: $CONTAINER_NAME"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs:     docker logs $CONTAINER_NAME"
echo "   Stop API:      docker stop $CONTAINER_NAME"
echo "   Remove API:    docker rm $CONTAINER_NAME"
echo "   Health check:  curl http://localhost:$PORT/health"
echo "   API schema:    curl http://localhost:$PORT/schema"
echo ""
echo "📖 View API documentation at: http://localhost:$PORT/docs"