# 0004 Repository System Boundaries

Date: 2026-08-06

## Status

Accepted

## Context

The repository grew through numbered phases. Shared E-GraphSAGE behavior,
federated training, a Minikube inference proof of concept, and two deployment
systems now depend on root directories whose names describe chronology rather
than ownership. Phase 4 needs an independently deployable detection application
without duplicating model behavior or importing Flower runtime concerns.

## Decision

- `src/core/` owns shared E-GraphSAGE, preprocessing, graph construction, and
  model/schema contracts.
- `src/federated/` owns training, Flower integration, aggregation, experiment
  orchestration, and personalized state production.
- `src/application/` owns collection, graph windows, inference serving, trusted
  routing, detection events, alerting, and the demo target.
- `deploy/federated/` owns Argo CD, Helm, environment values, Terraform,
  Ansible, and Docker definitions for training.
- `deploy/application/` owns Argo CD, Helm, environment values, and Docker
  definitions for the detection application.
- `configs/federated/` and `configs/application/` own system-specific settings.
- `artifacts/federated/` owns training runs/checkpoints/exports;
  `artifacts/application/` owns imported model bundles, replay evidence, and
  monitoring evidence. A component must not write into another component's
  artifact area.
- Dependency direction is `federated -> core` and `application -> core`.
  Production application code must not import Flower, a training run store, or
  client-training runtime code.
- `.github/workflows/` remains at repository root because GitHub requires that
  location. `Jenkinsfile` remains a root entrypoint and may dispatch logic owned
  elsewhere.
- During migration, old Python module paths may be thin compatibility imports.
  Remove them only after imports, tests, Docker builds, and documentation use the
  authoritative paths.

## Consequences

Positive:

- Shared behavior has one authority while training and serving remain
  independently testable and deployable.
- Deployment changes can target training or detection without ambiguous root
  ownership.
- Phase names stop becoming permanent production package names.

Tradeoffs:

- Path migration touches CI, Jenkins, Argo CD, Helm, Terraform, Ansible, Docker,
  scripts, tests, and documentation and therefore must be one coherent change.
- Compatibility modules temporarily increase surface area.
- Moving Terraform source requires an explicit no-replacement plan before any
  later deployment.

## Migration Safety

- Do not push an intermediate state where an Argo CD source path has disappeared.
- Do not change Terraform resource/module addresses as part of relocation.
- Do not apply Terraform or reconcile GKE while validating the refactor.
- Do not remove `phase3_monitoring/` until production imports and executable
  configuration have migrated and focused compatibility evidence passes.

The removal gate was satisfied on 2026-08-06: the legacy model contract moved
to `src/core/legacy_bundle.py`; production API, replay/evaluation, Docker, Helm,
and tests moved to the application boundary before the old PoC was removed.
