#!/bin/bash
set -e

# Variables
REGISTRY="${GLOBAL_REGISTRY}.example.com"
IMAGE_NAME="bert-moe"
IMAGE_TAG="latest"
NAMESPACE="model-serving"

# Build Docker image
echo "Building Docker image..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .

# Push to registry
echo "Pushing to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}

# Deploy to K3s
echo "Deploying to K3s..."
kubectl create namespace ${NAMESPACE} || true

# Apply Kubernetes manifests
kubectl -n ${NAMESPACE} apply -f k3s-deployment.yaml
kubectl -n ${NAMESPACE} apply -f k3s-hpa.yaml
kubectl -n ${NAMESPACE} apply -f k3s-ingress.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl -n ${NAMESPACE} rollout status deployment/bert-moe-deployment --timeout=300s

# Get service URL
SERVICE_IP=$(kubectl -n ${NAMESPACE} get svc bert-moe-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Service is available at: http://${SERVICE_IP}"

