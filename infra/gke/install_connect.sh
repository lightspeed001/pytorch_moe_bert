gcloud container clusters update ${CLUSTER_NAME} \
  --region ${REGION} \
  --workload-pool=${GCP_PROJECT}.svc.id.goog

