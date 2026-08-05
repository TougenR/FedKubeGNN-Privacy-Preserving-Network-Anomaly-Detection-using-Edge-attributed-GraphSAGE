#!/usr/bin/env bash
set -euo pipefail

project_id="${GCP_PROJECT_ID:-fedlearning-20260729-hn}"
zone="${GCP_ZONE:-asia-southeast1-b}"
repo_root="$(git rev-parse --show-toplevel)"

gcloud container clusters get-credentials fedkube-central \
  --project "${project_id}" --zone "${zone}"
central_context="$(kubectl config current-context)"
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update argo
helm upgrade --install argocd argo/argo-cd \
  --namespace argocd --create-namespace --version 9.1.3 \
  --set configs.params.server\.insecure=true \
  --wait

gcloud container clusters get-credentials fedkube-edge-01 \
  --project "${project_id}" --zone "${zone}"
edge_context="$(kubectl config current-context)"

kubectl config use-context "${central_context}"
argocd cluster add "${edge_context}" --name fedkube-edge-01 --yes
kubectl apply -f "${repo_root}/argocd/project.yaml"
kubectl apply -f "${repo_root}/argocd/rbac-cm.yaml"
kubectl apply -f "${repo_root}/argocd/operators.yaml"
kubectl apply -f "${repo_root}/argocd/applicationset.yaml"

echo "Argo CD bootstrap submitted; use argocd app list to observe reconciliation."
