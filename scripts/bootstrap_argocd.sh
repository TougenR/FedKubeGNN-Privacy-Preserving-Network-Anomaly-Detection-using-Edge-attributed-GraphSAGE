#!/usr/bin/env bash
set -euo pipefail

project_id="${GCP_PROJECT_ID:-fedlearning-20260729-hn}"
zone="${GCP_ZONE:-asia-southeast1-b}"
repo_root="$(git rev-parse --show-toplevel)"
helm_command="${HELM_COMMAND:-helm}"

command -v jq >/dev/null
if [[ "$("${helm_command}" version --short)" != v3.* ]]; then
  echo "Helm 3 is required; set HELM_COMMAND to a Helm 3 binary." >&2
  exit 2
fi

gcloud container clusters get-credentials fedkube-central \
  --project "${project_id}" --zone "${zone}"
central_context="$(kubectl config current-context)"
"${helm_command}" repo add argo https://argoproj.github.io/argo-helm
"${helm_command}" repo update argo
"${helm_command}" upgrade --install argocd argo/argo-cd \
  --namespace argocd --create-namespace --version 9.1.3 \
  --values "${repo_root}/deploy/federated/argocd/values.yaml" \
  --wait

gcloud container clusters get-credentials fedkube-edge-01 \
  --project "${project_id}" --zone "${zone}"
edge_context="$(kubectl config current-context)"

kubectl --context "${edge_context}" apply \
  -f "${repo_root}/deploy/federated/argocd/edge-manager.yaml"

umask 077
token_file="$(mktemp)"
config_file="$(mktemp)"
cleanup() { rm -f "${token_file}" "${config_file}"; }
trap cleanup EXIT

for _ in $(seq 1 30); do
  token_data="$(kubectl --context "${edge_context}" -n kube-system get secret \
    argocd-manager-long-lived-token -o jsonpath='{.data.token}')"
  if [[ -n "${token_data}" ]]; then
    printf '%s' "${token_data}" | base64 -d >"${token_file}"
    break
  fi
  sleep 1
done
test -s "${token_file}"

ca_data="$(kubectl config view --raw -o json | jq -r \
  --arg name "${edge_context}" \
  '.clusters[] | select(.name == $name) | .cluster["certificate-authority-data"]')"
edge_endpoint="$(gcloud container clusters describe fedkube-edge-01 \
  --project "${project_id}" --zone "${zone}" --format='value(endpoint)')"
test -n "${ca_data}"
test -n "${edge_endpoint}"
jq -n --rawfile bearerToken "${token_file}" --arg caData "${ca_data}" \
  '{bearerToken:$bearerToken,tlsClientConfig:{insecure:false,caData:$caData}}' \
  >"${config_file}"

kubectl --context "${central_context}" -n argocd create secret generic \
  cluster-fedkube-edge-01 \
  --from-literal=name=fedkube-edge-01 \
  --from-literal="server=https://${edge_endpoint}" \
  --from-file="config=${config_file}" --dry-run=client -o yaml \
  | kubectl --context "${central_context}" apply -f -
kubectl --context "${central_context}" -n argocd label secret \
  cluster-fedkube-edge-01 argocd.argoproj.io/secret-type=cluster --overwrite

kubectl --context "${central_context}" apply -f "${repo_root}/deploy/federated/argocd/project.yaml"
kubectl --context "${central_context}" apply -f "${repo_root}/deploy/federated/argocd/rbac-cm.yaml"
kubectl --context "${central_context}" apply -f "${repo_root}/deploy/federated/argocd/operators.yaml"
kubectl --context "${central_context}" apply -f "${repo_root}/deploy/federated/argocd/applicationset.yaml"

echo "Argo CD bootstrap submitted; use argocd app list to observe reconciliation."
