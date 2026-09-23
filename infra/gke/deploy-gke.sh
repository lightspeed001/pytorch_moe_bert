#!/bin/bash
set -e

# Variables
GCP_PROJECT="your-gcp-project-id"
CLUSTER_NAME="bert-moe-cluster"
REGION="us-central1"
IMAGE_NAME="gcr.io/${GCP_PROJECT}/bert-moe"
IMAGE_TAG="latest"
NAMESPACE="model-serving"

# Authenticate with GCP
echo "Authenticating with GCP..."
gcloud auth login
gcloud config set project ${GCP_PROJECT}
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION}

# Create namespace
echo "Creating namespace..."
kubectl create namespace ${NAMESPACE} || true

# Apply IAM configuration
echo "Applying IAM configuration..."
kubectl apply -f gke-iam.yaml

# Apply storage configuration
echo "Applying storage configuration..."
kubectl apply -f gke-storage.yaml

# Build and push Docker image
echo "Building and pushing Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker push ${IMAGE_NAME}:${IMAGE_TAG}

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -n ${NAMESPACE} -f gke-deployment.yaml
kubectl apply -n ${NAMESPACE} -f gke-hpa.yaml
kubectl apply -n ${NAMESPACE} -f gke-service.yaml
kubectl apply -n ${NAMESPACE} -f gke-ingress.yaml
kubectl apply -n ${NAMESPACE} -f gke-monitoring.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl -n ${NAMESPACE} rollout status deployment/bert-moe-deployment --timeout=600s

# Get service URL
INGRESS_IP=$(kubectl -n ${NAMESPACE} get ingress bert-moe-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Service is available at: http://${INGRESS_IP}"
echo "Waiting for SSL certificate to be provisioned..."
sleep 300  # Wait for SSL certificate to be provisioned

# Verify HTTPS endpoint
echo "Verifying HTTPS endpoint..."
curl -k https://model.yourdomain.com/health

echo "Deployment complete!"

