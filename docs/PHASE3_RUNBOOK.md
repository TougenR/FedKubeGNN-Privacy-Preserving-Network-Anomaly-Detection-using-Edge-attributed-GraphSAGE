# Phase 3 Runbook

Nothing in this runbook authorizes `terraform apply`. Stop after the main plan
until its resource count, machine sizes, addresses, IAM, and estimated cost are
explicitly approved.

## 1. Local validation and review plan

```bash
cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars
# Replace admin_source_ranges with the operator's current public-IP /32.

terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap plan -out=bootstrap.tfplan

scripts/terraform_review_plan.sh <operator-public-ip>/32
```

The review script omits only `backend.tf` in a temporary directory; it never
creates an applicable binary plan. The bootstrap plan creates only the
protected state bucket. After approval,
apply bootstrap first, reinitialize the main module with `terraform init
-reconfigure`, reproduce the main plan against remote state, and request a final
approval if it differs.

## 2. Populate secrets after infrastructure approval

Terraform creates empty Secret Manager containers. Run
`scripts/seed_phase3_secrets.sh` only after apply. It generates a private CA,
the IP-SAN server certificate, and an Elastic writer password in a temporary
directory; it sends values directly to Secret Manager and prints none of them.

Jenkins Credentials must contain:

- `dockerhub-credentials`: Docker Hub username and access token.
- `github-push-key`: deploy key or machine-user SSH key allowed to push `main`.

The GitHub repository needs a webhook to
`http://<jenkins-public-ip>/github-webhook/` for push events. The GCP firewall
loads GitHub's current hook CIDRs dynamically; NGINX exposes only that path.
Reach the Jenkins UI through an SSH/IAP tunnel, never the public webhook port.

## 3. Seed data and bootstrap GitOps

Upload one validated Phase 2 prepared dataset so that
`gs://fedlearning-20260729-hn-fedkube-training/prepared/current/manifest.json`
exists. Do not upload raw credentials or unchecked pickle/PyG objects.

After the clusters and secret versions exist:

```bash
scripts/bootstrap_argocd.sh
```

The script installs Argo CD on Central once, registers Edge-01, and applies the
Argo project, operators, RBAC, and ApplicationSet. ECK and External Secrets are
then reconciled by Argo CD.

## 4. Acceptance evidence

Capture evidence under ignored `artifacts/phase3/evidence/<timestamp>/`:

1. GitHub webhook delivery and Jenkins build URL.
2. Docker Hub tag and immutable digest.
3. Jenkins digest-only commit.
4. `argocd app get fedkube-central` and `fedkube-edge-01` showing
   `Synced/Healthy`.
5. Dataset Sync Job completion and PVC manifest checksum.
6. Six connected SuperNodes and at least one completed FL round.
7. A FedPer summary for all six scenarios, plus proof that Flower payloads and
   the Central checkpoint contain only `layers.*`.
8. Generated learning curves, final metric comparison, per-class F1, and
   confusion matrices in the GCS model-artifact bucket.
9. GCS shared-encoder object metadata and each Edge client head-PVC metadata
   (`ready=true`, completed round, model digest) without copying head tensors.
10. A Kibana Discover view of Central and Edge structured logs.

Destroy or scale down immediately after capturing the evidence. Protected GCS
buckets are not force-destroyed; intentionally empty or retain them before the
rest of `terraform destroy`.
