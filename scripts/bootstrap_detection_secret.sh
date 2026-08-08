#!/usr/bin/env bash
set -euo pipefail

project_id="${1:-fedlearning-20260729-hn}"
secret_id="fedkube-detection-entity-hash"
accessor="external-secrets-central@${project_id}.iam.gserviceaccount.com"

if ! gcloud secrets describe "${secret_id}" --project "${project_id}" >/dev/null 2>&1; then
  gcloud secrets create "${secret_id}" \
    --project "${project_id}" \
    --replication-policy automatic \
    --labels owner=phase4-detection,purpose=entity-hash >/dev/null
fi

if [[ -z "$(gcloud secrets versions list "${secret_id}" \
  --project "${project_id}" \
  --filter 'state=ENABLED' \
  --format 'value(name)' \
  --limit 1)" ]]; then
  openssl rand -hex 32 | gcloud secrets versions add "${secret_id}" \
    --project "${project_id}" \
    --data-file=- >/dev/null
fi

gcloud secrets add-iam-policy-binding "${secret_id}" \
  --project "${project_id}" \
  --member "serviceAccount:${accessor}" \
  --role roles/secretmanager.secretAccessor \
  --quiet >/dev/null

echo "Detection entity-hash secret is ready in project ${project_id}."
