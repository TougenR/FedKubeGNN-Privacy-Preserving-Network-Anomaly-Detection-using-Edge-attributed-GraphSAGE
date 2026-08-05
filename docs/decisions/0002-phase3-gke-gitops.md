# 0002 Phase 3 GKE GitOps Platform

Date: 2026-07-29

## Status

Accepted

## Context

Phase 3 must deploy the six-client IoT-23 federation on GCP within the
approximately USD 300 Free Trial credit, retain a clear separation between
build and deployment authority, and expose enough operational logs for a user
to diagnose a full training run. The linked billing account reports VND, so
the enforceable budget resource uses VND 7,800,000.
The platform must start with one Central and one Edge location but allow more
Edge clusters without redesigning the application chart.

## Decision

- Use zonal GKE Standard with one Central and one Edge cluster in
  `asia-southeast1-b`, a shared custom VPC, and non-overlapping alias IP ranges.
  Zonal control planes make one cluster eligible for the monthly GKE free-tier
  credit; the MVP accepts the lower 99.5% control-plane SLA.
- Run Argo CD in Central and make it the only controller that continuously
  deploys application state to either GKE cluster.
- Run Jenkins on a Compute Engine VM configured by Ansible. GitHub validates
  changes; Jenkins builds and publishes an immutable Docker Hub image and
  updates Git environment digests only.
- Run all six Phase 2 IoT-23 scenarios as six Flower nodes on Edge-01. Preserve
  full participation, 30 rounds, five local epochs, FedAvg followed by FedProx.
- Keep training data, model artifacts, and Terraform state in three separate
  private GCS buckets and authorize pods through Workload Identity.
- Protect SuperNode-to-SuperLink traffic with TLS and publish SuperLink only
  through an internal NGINX load balancer.
- Use ECK Basic for a single-node Elasticsearch and internal Kibana deployment.
  Filebeat collects application and Kubernetes container logs from both
  clusters; indices expire after seven days.
- Use `e2-standard-4` for Central, `e2-custom-6-24576` for Edge, and
  `e2-standard-2` for Jenkins only for the time-bounded demo. Alert at VND
  7,800,000 and destroy or scale down after evidence is captured.

## Alternatives Considered

1. One GKE cluster per scenario, rejected because control-plane and node cost
   would exhaust the trial without improving the MVP protocol.
2. Jenkins-driven `helm upgrade`, rejected because it creates two competing
   deployment authorities and weakens GitOps auditability.
3. Cloud Logging only, rejected because the accepted demo explicitly requires
   Kibana and seven-day searchable application/Kubernetes logs.
4. A production three-node Elasticsearch cluster, deferred because the demo has
   a fixed trial budget and GCS, not Elasticsearch, owns durable artifacts.

## Consequences

Positive:

- Image production, desired state, and cluster reconciliation have distinct,
  auditable owners.
- The six-scenario benchmark remains compatible with Phase 2 while Edge 1..N
  can be introduced through values and ApplicationSet entries.
- Logs are queryable without exposing Kibana or Elasticsearch publicly.

Tradeoffs:

- The MVP has single-node failure domains and must not be described as HA.
- Six CPU clients and Elastic on two clusters require time-bounded operation to
  remain within the trial.
- Initial Argo CD/operator bootstrapping remains an explicit one-time platform
  operation before GitOps can own ongoing reconciliation.

## Follow-Up

- After the demo, evaluate Spot training nodes, multi-zone Central capacity,
  OpenTelemetry export, Elasticsearch snapshots, and distributing client IDs
  across additional Edge clusters.
