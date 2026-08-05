#!/usr/bin/env bash
set -euo pipefail

project_id="${GCP_PROJECT_ID:-fedlearning-20260729-hn}"
secret_dir="$(mktemp -d)"
cleanup() { rm -rf -- "${secret_dir}"; }
trap cleanup EXIT

umask 077
openssl genrsa -out "${secret_dir}/ca.key" 4096 >/dev/null 2>&1
openssl req -x509 -new -nodes -key "${secret_dir}/ca.key" -sha256 -days 365 \
  -subj "/CN=FedKube Phase 3 CA" -out "${secret_dir}/ca.crt"
openssl genrsa -out "${secret_dir}/tls.key" 4096 >/dev/null 2>&1
openssl req -new -key "${secret_dir}/tls.key" -subj "/CN=10.10.0.10" \
  -out "${secret_dir}/server.csr"
cat >"${secret_dir}/san.ext" <<'EOF'
subjectAltName=IP:10.10.0.10,IP:10.10.0.11,DNS:fedkube-central-superlink,DNS:fedkube-logs-es-http
extendedKeyUsage=serverAuth
EOF
openssl x509 -req -in "${secret_dir}/server.csr" \
  -CA "${secret_dir}/ca.crt" -CAkey "${secret_dir}/ca.key" -CAcreateserial \
  -out "${secret_dir}/tls.crt" -days 365 -sha256 -extfile "${secret_dir}/san.ext" \
  >/dev/null 2>&1
openssl rand -base64 36 >"${secret_dir}/elastic-password"

gcloud secrets versions add fedkube-flower-ca --project "${project_id}" \
  --data-file="${secret_dir}/ca.crt" >/dev/null
gcloud secrets versions add fedkube-flower-cert --project "${project_id}" \
  --data-file="${secret_dir}/tls.crt" >/dev/null
gcloud secrets versions add fedkube-flower-key --project "${project_id}" \
  --data-file="${secret_dir}/tls.key" >/dev/null
gcloud secrets versions add fedkube-elastic-password --project "${project_id}" \
  --data-file="${secret_dir}/elastic-password" >/dev/null

echo "Phase 3 secret versions were added to Secret Manager; values were not printed."
