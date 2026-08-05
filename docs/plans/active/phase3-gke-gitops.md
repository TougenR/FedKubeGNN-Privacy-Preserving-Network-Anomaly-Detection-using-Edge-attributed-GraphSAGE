# Execution Plan: Phase 3 GKE GitOps Platform

Date: 2026-07-29

## Status

The explicitly approved platform resume plan has been applied and the Central
GKE, Edge GKE, and Jenkins VM are running. Terraform reports no drift. Jenkins
is configured by Ansible with Docker Hub/GitHub credentials, and the writable
deploy key plus push webhook are active. The next gate is the first `main` push,
Jenkins digest commit, and Argo CD bootstrap before the acceptance demo.

## Outcome

Provision a reproducible two-cluster GKE Standard platform in project
`fedlearning-20260729-hn` where GitHub and Jenkins produce an immutable Flower
application image, Argo CD alone reconciles Central and Edge workloads, six
IoT-23 scenario clients complete the Phase 2 training protocol, model artifacts
land in GCS, and Kubernetes/application logs are searchable in internal Kibana.

## Context

- `docs/decisions/0001-phase2-modularity-observability.md`: six-client benchmark
  and safe structured-log contract.
- `docs/PHASE2_ARCHITECTURE.md`: authoritative Phase 2 data and training protocol.
- `configs/phase2/iot23-federated.yaml`: six scenarios, 30 rounds, five local
  epochs, FedAvg then FedProx.
- `src/federated/flower/`: Flower 1.32 Message API ServerApp and ClientApp.
- GitHub repository: `TougenR/FedKubeGNN-Privacy-Preserving-Network-Anomaly-Detection-using-Edge-attributed-GraphSAGE`.
- Docker Hub repository: `hieunguyen595/fedkube-gnn`.

## Scope

In scope:

- One custom VPC, one Central GKE Standard cluster, and one Edge GKE Standard
  cluster with non-overlapping Pod, Service, and control-plane CIDRs.
- Three private GCS buckets for training data, model artifacts, and Terraform
  remote state; GKE Workload Identity for data and artifact access.
- A Jenkins Compute Engine VM configured by Ansible and triggered from GitHub.
- One application Helm chart, environment value files, Argo CD ApplicationSet,
  and bootstrap definitions for Central and Edge.
- Six simultaneous SuperNode/ClientApp pairs at Edge-01: `34-1`, `1-1`, `3-1`,
  `9-1`, `36-1`, and `39-1`.
- TLS from Edge SuperNodes through an internal NGINX load balancer to Central
  SuperLink.
- Single-node Elasticsearch, internal Kibana, and Filebeat collection with a
  seven-day retention policy.
- GitHub Actions validation and Jenkins image build/push/digest update without
  a deployment command.

Out of scope:

- Kafka, public application load balancers, production HA Elasticsearch,
  multi-region disaster recovery, GPU nodes, and six separate Edge clusters.
- Applying the platform Terraform plan, uploading private data, adding
  credentials, or running the billable end-to-end demo before explicit
  approval.

## Approach

1. Containerize the Flower app once and reuse it for ServerApp, ClientApps, and
   the run launcher while pinning Flower infrastructure images separately.
2. Build one Helm chart with `central` and `edge` modes. Central owns SuperLink,
   run launcher, artifact uploader, internal NGINX, Elasticsearch, Kibana, and
   central Filebeat. Edge owns dataset sync, PVC, six SuperNodes, and Filebeat.
3. Provision networking, GKE, storage, IAM, Secret Manager placeholders, and the
   Jenkins VM with Terraform; bootstrap remote state separately.
4. Configure Jenkins and Docker with Ansible. Keep all secret material in
   Jenkins Credentials, Secret Manager, gcloud ADC, or Workload Identity.
5. Validate pull requests with GitHub Actions. Jenkins builds `main`, pushes a
   content-addressed image, and commits only updated environment digests while
   ignoring environment-only webhook changes.
6. Let Argo CD reconcile all ongoing Kubernetes state. Bootstrap Argo CD once
   after infrastructure creation, then use ApplicationSet for Central/Edge.
7. Present `terraform plan` for approval. Only after approval, apply, seed data,
   start both benchmark strategies, and capture acceptance evidence.

## Risks And Recovery

- Two GKE control planes, `e2-standard-4` Central, `e2-custom-6-24576` Edge, and
  `e2-standard-2` Jenkins can consume the trial quickly. Budget alert is VND
  7,800,000; resources must be destroyed or scaled down immediately after
  evidence.
- Single-node Elasticsearch is an MVP observability store and has no HA. Its
  30 GiB persistent disk can be recreated; GCS remains authoritative for model
  and training artifacts.
- Six full-participation CPU clients may run slowly. Resource requests and
  affinity keep them on the one approved Edge node; the benchmark contract is
  not weakened to partial participation.
- TLS private keys and external tokens must never enter Git. Terraform creates
  Secret Manager containers only; secret versions are populated out of band.
- Recovery is `terraform destroy` against the reviewed state plus explicit
  confirmation that all three protected buckets are empty or intentionally
  retained. Argo changes can be reverted by Git commit.

## Progress

- [x] Confirm GCP account, project, billing, region, sizing, log retention, six
  clients, GitHub repository, and Docker Hub repository.
- [x] Add Phase 3 architecture and operator documentation.
- [x] Containerize Flower workloads and add container smoke proof.
- [x] Build Central/Edge Helm chart and environment values.
- [x] Build Terraform networking, GKE, GCS, IAM, Jenkins VM, and remote state.
- [x] Build Ansible Jenkins configuration.
- [x] Build GitHub Actions and Jenkins pipeline.
- [x] Build Argo CD ApplicationSet/bootstrap, RBAC, Workload Identity, internal
  NGINX, ECK, Kibana, and Filebeat resources.
- [x] Run local test, lint, template, Terraform, and Ansible validation.
- [x] Generate the bootstrap and platform review plans; present them for
  explicit approval before any apply.
- [x] Apply the explicitly approved bootstrap plan and initialize the main
  module against the protected GCS remote state.
- [x] Finish the approved platform apply after resolving the 12-vCPU quota;
  verify both GKE clusters and all three compute nodes are healthy and
  Terraform reports no drift.
- [x] Configure Jenkins with Ansible, verify the managed pipeline job and
  services, seed runtime TLS/Elastic secret versions, and prove Ansible
  idempotence.
- [x] Add Docker Hub and GitHub push credentials to Jenkins and create the
  GitHub webhook.
- [ ] Push the Phase 3 commit and bootstrap Argo CD.
- [ ] Run the acceptance demo.
- [x] On user approval, destroy the main platform to stop ongoing compute and
  network charges while retaining the project and Terraform state bucket.
- [ ] Store end-to-end evidence and move this plan to completed.

## Decisions

- 2026-07-29: Use GKE Standard in `asia-southeast1`, initially one Central and
  one Edge cluster on a shared custom VPC.
- 2026-07-29: Initially selected `e2-standard-4` Central, `e2-standard-8` Edge,
  and `e2-standard-2` Jenkins for the time-bounded demo; the billing account's
  VND-denominated alert is VND 7,800,000.
- 2026-07-29: After GCP enforced a global 12-vCPU Free Trial quota, the user
  approved resizing Edge to `e2-custom-6-24576` and lowering ClientApp CPU
  requests to `800m`. Central remains 4 vCPU, all six clients remain enabled,
  and the three machines can run concurrently at exactly 12 vCPU.
- 2026-07-29: Run all six scenario clients concurrently on Edge-01 and preserve
  the Phase 2 full-participation protocol. Future Edge clusters may distribute
  these client IDs without changing the chart contract.
- 2026-07-29: Use ECK Basic with one Elasticsearch node, 30 GiB storage,
  internal-only Kibana, Filebeat on both clusters, and seven-day retention.
- 2026-07-29: Use `hieunguyen595/fedkube-gnn` as the application image and pin it
  in environment files by immutable digest.
- 2026-07-29: Argo CD is the sole ongoing GKE deployer; Jenkins never invokes
  Helm or kubectl.
- 2026-07-29: Use distinct Central/Edge Kubernetes service-account names. GKE
  Workload Identity principals do not include cluster identity, so reusing one
  namespace/name pair across clusters would permit unintended GSA impersonation.
- 2026-07-29: Use zonal Standard clusters in `asia-southeast1-b`, ensuring
  `node_count=1` means one billable node and making one cluster management fee
  eligible for the GKE monthly free-tier credit. The MVP accepts the zonal
  control plane's lower SLA.
- 2026-07-29: Build the application on Python 3.11 instead of the current
  `flwr/superexec:1.32.0` base because that image uses Python 3.13 and cannot
  consume the Phase 2 `numpy<2` wheel without a source build.

## Validation

- Focused proof: Python unit tests plus container import/CLI smoke.
- Configuration proof: `helm lint`, `helm template` for Central and Edge,
  `terraform fmt -check`, `terraform validate`, `ansible-playbook
  --syntax-check`, workflow/YAML checks, and `git diff --check`.
- Integration proof after approval: GitHub webhook to Jenkins, Docker Hub digest
  commit, Argo Synced/Healthy, GCS-to-PVC sync, six connected SuperNodes, at
  least one completed federated round, model in GCS, and logs visible in Kibana.
- Repository-required checks: relevant test suites and clean generated output.

Observed 2026-07-29:

- Docker image `fedkube-gnn:phase3-local` built successfully as non-root user
  `app` with Python 3.11, Flower 1.32.1, PyTorch 2.5.1 CPU, and PyG 2.8.0.
- All 46 Phase 2 tests passed inside that image, including the five optional
  Flower protocol tests; ServerApp and ClientApp imports passed.
- Both Central and Edge passed `helm lint` and `helm template`; parsed render
  proof found exactly the six required client IDs plus Elasticsearch and Kibana.
- Both Terraform modules passed format and validate with pinned providers.
- Ansible syntax, shell syntax, YAML parsing, digest updater test, secret scan,
  and `git diff --check` passed.
- Referenced operator charts resolve: ECK 3.2.0, External Secrets 0.20.4, and
  Argo CD chart 9.1.3.
- Bootstrap review plan: `2 add, 0 change, 0 destroy`.
- The approved bootstrap binary plan applied successfully: Storage API enabled
  and `fedlearning-20260729-hn-fedkube-tfstate` created with versioning,
  uniform bucket-level access, public-access prevention, and seven-day soft
  delete. Apply result: `2 added, 0 changed, 0 destroyed`.
- Platform review plan: `61 add, 0 change, 0 destroy`. It included a static
  Cloud NAT address and initially selected one `e2-standard-4` Central node,
  one `e2-standard-8` Edge node, and one `e2-standard-2` Jenkins VM.
- The applicable plan regenerated against the GCS backend also reports
  `61 add, 0 change, 0 destroy`; Jenkins SSH is limited to the operator IP
  `42.119.86.122/32` and Google IAP.
- The approved platform apply exposed and fixed three ordering/network issues:
  Compute addresses now depend on API enablement, the IPv4 Jenkins endpoint
  filters GitHub's IPv4 hook ranges, and Workload Identity IAM bindings wait for
  the clusters to create the workload pool. Google providers now use the
  resource project for user-project quota and billing requests.
- The approved Edge replacement completed with `e2-custom-6-24576` and
  zero-surge node-pool upgrades. Central `e2-standard-4`, Edge 6-vCPU, and
  Jenkins `e2-standard-2` are all running within the 12-vCPU quota. Central
  uses `pd-standard`; chart PVCs use the `standard` StorageClass to remain
  within the Free Trial balanced-disk quota.
- The billing account uses VND. Budget `FedKube Phase 3 Free Trial` exists at
  VND 7,800,000 with 50%, 90%, and 100% thresholds. The final Terraform plan
  reports `No changes`.
- Jenkins 2.568.1 runs on Java 21 with Docker and webhook-only NGINX. JCasC
  created `fedkube-main`; all services are active, and the final Ansible run
  reported `changed=0`, `failed=0`.
- Secret Manager contains enabled version 1 for the Flower CA, server
  certificate, server key, and Elasticsearch password. Secret values were not
  printed or committed.
- Rebuilt image `fedkube-gnn:phase3-validated` passed all 47 tests inside the
  dependency-complete container, including all Flower tests; imports passed.
- GitHub CLI is authenticated as `TougenR` with repository `WRITE` permission,
  but repository-hook and deploy-key endpoints require administrator access.
  Docker Hub repository `hieunguyen595/fedkube-gnn` exists, but the local
  Docker client is not authenticated.
- The approved destroy binary plan contained `0 add, 0 change, 61 destroy` and
  explicitly excluded the project and bootstrap-managed Terraform state
  bucket. Apply completed with all 61 resources destroyed. Post-destroy checks
  found no Compute Engine instances/disks/addresses/routers or GKE clusters;
  the project remains active and billing-linked, and only
  `fedlearning-20260729-hn-fedkube-tfstate` remains.

Observed 2026-08-05:

- Docker Hub authentication succeeds as `hieunguyen595`; GitHub reports
  `TougenR` as repository administrator. The canonical repository moved to
  `TougenR/FedKubeGNN-Privacy-Preserving-Network-Anomaly-Detection-using-Edge-attributed-GraphSAGE`,
  and Git remote, Ansible, and Argo CD references now use that owner.
- Resume validation passed: 47 host tests with five optional Flower skips,
  focused Ruff checks, Central/Edge Helm lint and render, Terraform format and
  validate, Ansible syntax, shell syntax, and `git diff --check`.
- Current operator IP remains `42.119.86.122/32`; the global CPU quota is 12
  and the applicable resume plan was `61 add, 0 change, 0 destroy`,
  with Central `e2-standard-4`, Edge `e2-custom-6-24576`, Jenkins
  `e2-standard-2`, and budget VND 7,800,000. Binary plan SHA-256:
  `d5f276f49fe97f01e555736eee14f02846642a14d1044f22993bdf4d66b3a9e2`.
- The approved resume apply completed exactly `61 added, 0 changed, 0
  destroyed`. Both GKE clusters and node pools report `RUNNING`, the Jenkins VM
  reports `RUNNING`, and the immediate Terraform refresh reports `No changes`.
- Ansible configured Jenkins 2.568.1, Docker, Java 21, JCasC, and webhook-only
  NGINX; the service health checks pass. Jenkins Credentials contains only the
  expected Docker Hub and GitHub deploy-key IDs, and staged secret files are
  absent after provisioning.
- The writable repository deploy key is active, Jenkins can read `main`, and
  the GitHub push webhook ping returned HTTP 200.
- The rebased Python 3.11 integration image passed all 81 tests. Ruff, Helm
  lint/render, Terraform format/validate, Ansible syntax, shell syntax, and
  `git diff --check` also pass.

## Result

The approved Phase 3 platform is running without Terraform drift. Jenkins and
the GitHub trigger path are ready; the remaining work is to push the validated
Phase 3 commit, allow Jenkins to publish and commit the immutable image digest,
bootstrap Argo CD, then run and capture the acceptance demo.
