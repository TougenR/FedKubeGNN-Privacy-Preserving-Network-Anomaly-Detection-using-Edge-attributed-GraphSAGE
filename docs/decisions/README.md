# Decisions

Decision records preserve lasting product, architecture, data ownership,
security, compatibility, and validation choices that future work must inherit.

Use `docs/templates/decision.md`. Task-local implementation choices remain in
the active execution plan and do not require a separate decision.

An installed consumer begins with no fabricated decisions. Add local decision
documents here as real choices are accepted, then index them in this file.

## Accepted Decisions

- [`0001-phase2-modularity-observability.md`](0001-phase2-modularity-observability.md):
  modular extension points, train-only benchmark preparation, and the Phase 2
  observability contract.
- [`0002-phase3-gke-gitops.md`](0002-phase3-gke-gitops.md): GKE topology,
  six-client placement, GitOps authority, GCS ownership, and Kibana logging.
- [`0003-fedper-edge-personalization.md`](0003-fedper-edge-personalization.md):
  shared GraphSAGE encoder, Edge-owned classifier heads, cold-start readiness,
  and durable private-state ownership.
- [`0004-repository-system-boundaries.md`](0004-repository-system-boundaries.md):
  durable ownership for shared core, federated training, detection application,
  and their independent deployment systems.
- [`0005-centralized-fedper-research-serving.md`](0005-centralized-fedper-research-serving.md):
  immutable multi-head research bundles, trusted sensor routing, and the
  serving exception to Edge-local Phase 3 head ownership.
