# Execution Plan: Centralized FedPer Detection Application

Date: 2026-08-06

## Status

Active

## Outcome

Deliver an independently deployable detection application that loads one
validation-selected shared E-GraphSAGE encoder and six exact best-round FedPer
heads from an immutable inference bundle, routes trusted sensors to their
assigned heads, evaluates the scientific multi-head behavior, and demonstrates
lab traffic detection through Elasticsearch/Kibana without depending on Flower
or the Phase 3 training runtime.

The repository must finish with durable ownership boundaries for shared model
behavior, federated training, detection application code, and the deployment of
each system. The user approved proceeding directly on the existing GCP/GKE
environment on 2026-08-06, superseding the earlier Minikube-first/no-cloud gate.

## Context

- `docs/WORKFLOW.md`: durable-plan and validation requirements.
- `docs/decisions/0002-phase3-gke-gitops.md`: Phase 3 deployment authority.
- `docs/decisions/0003-fedper-edge-personalization.md`: encoder/head ownership
  during training and cold-start behavior.
- `docs/decisions/0004-repository-system-boundaries.md`: target repository
  ownership and dependency direction.
- `docs/decisions/0005-centralized-fedper-research-serving.md`: authorized
  centralized copies, trusted routing, and inference-bundle contract.
- `docs/PHASE3_ARCHITECTURE.md`: current GitOps, logging, and privacy rules.
- The former `phase3_monitoring/` compatibility PoC was removed only after its
  contract, API/evaluation, Docker, tests, and Kubernetes responsibilities had
  replacements under the core/application boundaries.
- Exact GKE run `14339380272482304688`, FedPer run
  `fedper-20260805T162143974378Z-270b7ffe84`, dataset digest
  `c5ab9c02896c08c9f60e8efb9672a2090cbe595e4c344308f5e4dc2b0e51319a`,
  and model digest
  `42642e4cc839c09dfe8519511aa7cf7cdf5ca7350a8dd376e118ee31a6a74bbf`.
- `report/` is an untracked local archive owned by the user. It is read-only
  source evidence for this work and must not be edited, deleted, or committed.

## Scope

In scope:

- Repository boundary refactor with compatibility imports during migration.
- Coherent relocation of federated deployment code and all executable path
  consumers in CI, GitOps, Jenkins, Terraform/Ansible helpers, and docs.
- Immutable inference-bundle exporter with provenance and digest validation.
- Centralized multi-head loader, trusted sensor router, production API schema,
  scientific replay schema, and fail-closed readiness.
- Correctly routed, cross-head, local 6 x 7, and validation-selected oracle
  evaluation.
- Validation-selected rolling-window protocol and stability measurements.
- GKE lab ingress/collector, structured alert routing, Elasticsearch mapping,
  Kibana assets, and an independent Helm chart deployed through Argo CD.
- Local reports, confusion matrices, latency/drop-rate evidence, and privacy
  checks for Elasticsearch documents.

Out of scope:

- Federated training, Flower/SuperLink/ServerApp at application runtime.
- Modifying or recreating Phase 3 source checkpoints, heads, dataset, GCS
  objects, PVCs, or training infrastructure.
- Automatic traffic blocking, zero-day claims, production-readiness claims, or
  attacks outside a user-owned lab.
- Re-running or mutating Phase 3 training and its immutable source artifacts.

## Approach

1. Inventory all old path consumers, lock repository authority in ADRs, and
   capture the exact source-artifact identities.
2. Create `src/core`, `src/application`, `deploy/federated`,
   `deploy/application`, `configs/application`, `tests/application`, and the
   split artifact roots. Keep compatibility modules until all consumers pass.
3. Relocate deployment files in one coherent local change and update Jenkins,
   GitHub Actions, ApplicationSet, bootstrap/review scripts, Docker paths, and
   documentation together. Do not push or reconcile Argo CD during migration.
4. Export an immutable bundle without mutating source artifacts. Validate every
   digest, model/run/best-round relationship, schema, client mapping, and
   non-cold-start head before promotion into application artifacts.
5. Build application runtime and scientific evaluation without importing
   `src.federated.flower`, training run stores, or client training runtime.
6. Evaluate rolling-window candidates on validation, lock the selected graph
   protocol in the bundle/derived deployment configuration, then run test once.
7. Add the local detection stack, structured privacy-preserving events, Kibana
   assets, Helm packaging, and lab-only replay/live-ingress evidence.
8. Run focused, integration, repository, and deployment validation, then use the
   already-approved existing GCP project. Keep any temporary evaluation storage
   isolated from Phase 3 and record/remove it after evidence is collected.

## Risks And Recovery

- **Live GitOps path breakage:** Argo CD currently points to root `charts/` and
  `environments/`. Keep all migration changes local until the coherent diff is
  validated. If a push is later authorized, update ApplicationSet paths in the
  same commit; pause auto-prune before any staged migration.
- **Terraform address drift:** moving files must not change Terraform resource
  or module addresses. Run a review plan with refresh disabled; stop on any
  unexpected destroy/recreate. Recovery is reverting only path relocation and
  path consumers, never resetting user work or applying a plan.
- **Artifact provenance mismatch:** exporter is fail-closed and writes to a
  temporary directory before atomic promotion. Source artifacts remain
  read-only. Remove only an incomplete generated destination on failure.
- **FedPer privacy-boundary exception:** centralized head copies are authorized
  only for research/demo bundles. They must not replace the Phase 3 training
  ownership rule or be uploaded to the Phase 3 Central GCS artifact location.
- **Graph-protocol inflation:** select windows on validation only, freeze the
  choice, and evaluate test once. Retain batch-boundary and late-event evidence.
- **Local dependency gaps:** use the project Docker boundary when host Python
  lacks PyG. Missing optional tools are disclosed; they do not justify weakened
  validation.
- **Dirty worktree:** preserve unrelated changes and the untracked `report/`
  archive. Do not use destructive Git commands.

## Progress

- [x] Read workflow, accepted Phase 3 decisions, current CI/GitOps paths, and
  repository status.
- [x] Identify user-provided authority for repository boundaries, centralized
  research head copies, trusted routing, privacy, and cost controls.
- [x] Record lasting repository and serving decisions.
- [x] Complete dependency inventory and coherent deployment relocation.
- [x] Establish shared core and application package boundaries with compatibility
  imports.
- [x] Export and validate the exact GKE best-round multi-head research bundle.
- [x] Implement multi-head loader, trusted router, schemas, API, and readiness.
- [x] Implement correctly routed, cross-head, local matrix, and oracle evaluation.
- [x] Implement rolling-window candidates, select on validation, promote the
  immutable serving bundle, and evaluate the locked protocol exactly once on
  test. The selected protocol is
  `rolling-window-v1:duration=60s:max-flows=50:stride=1:lateness=1s`.
- [ ] Implement collector, alert router, Kibana assets, Docker, and Helm chart.
  Collector/window orchestration, ingress-adapter contract, structured router,
  strict Elasticsearch mapping, Docker boundary, and chart templates exist;
  a validated Kibana saved-object export and live deployment evidence remain.
- [x] Implement the internal demo console, exact six-scenario catalog, bounded
  fixed-target runner, privacy-reduced live monitor, NGINX internal gateway,
  and application-owned ECK resources in the independent application chart.
- [ ] Pass local scientific and live-ingress acceptance.
- [ ] Run repository-wide validation and document remaining limitations.
- [ ] Obtain separate approval before any GKE deployment or paid operation.
  Approval was granted on 2026-08-06; this item is retained only as history.

## Decisions

- 2026-08-06: Treat the user's Phase 4 specification in this thread as product
  authority for centralized research copies, trusted routing, experiment
  protocol, event privacy, and acceptance criteria.
- 2026-08-06: Keep `.github/workflows/` and the root `Jenkinsfile` as required
  entrypoints; move implementation/deployment ownership below `deploy/`.
- 2026-08-06: Do not edit or relocate `report/`; exporter reads exact archived
  artifacts and produces a new derived bundle under application ownership.
- 2026-08-06: Do not choose window or alert thresholds during implementation.
  Window selection uses validation evidence; alert thresholds require labeled
  validation and an observable false-alert tradeoff.
- 2026-08-06: Publish schema-v2 research bundles with `serving_ready=false` and
  no serving `graph_protocol`. Scientific evaluation can load them, but the live
  API uses strict loading and stays unready until validation selects a rolling
  protocol and a new immutable serving bundle is promoted.
- 2026-08-06: Reuse the already authorized Docker Hub repository
  `hieunguyen595/fedkube-gnn`; Jenkins separates training and application images
  with `fed-<git-sha>` and `app-<git-sha>` tags and commits their digests to the
  respective deployment boundary.
- 2026-08-06: Use `ingress-adapter-v1` only as an explicitly approximate PoC
  observation source. It must never be described as Zeek-equivalent input.
- 2026-08-06: The user approved replacing the Minikube-first path with direct
  execution on the existing GCP project and GKE Central cluster. This authorizes
  the Phase 4 GKE deployment and its bounded temporary evaluation storage, but
  not a Phase 3 retrain or mutation of source checkpoints/datasets.
- 2026-08-06: GCS retains only prepared graph arrays and therefore cannot support
  time-window validation because timestamps/raw flow fields were intentionally
  omitted. Reconstruct the exact deterministic sample from the six official
  IoT-23 sources in a GKE batch job, verify every source against the size and
  SHA-256 recorded in the immutable source manifest, process one source at a
  time, and discard raw downloads after sampling.
- 2026-08-06: Use the explicit Jenkins commit trailer `[application-only]` for
  this Phase 4 release. Jenkins still tests/builds/scans/pushes the application
  image and updates only `deploy/application/environments/`; the Phase 3 image,
  environment digests, pods, and training state remain unchanged. Without that
  trailer, the normal path-based federated/application release behavior remains
  unchanged.
- 2026-08-09: The user approved the GKE demo-console plan. The live alert
  confidence threshold is `0.85`, selected from validation with benign
  false-alert rate `0.000823` and malicious alert recall `0.776353`. The web
  console remains internal-only, exposes only fixed lab targets, and enforces
  bounded server-side scenario parameters. A displayed traffic pattern is not
  ground truth and must remain visually distinct from the model prediction.
- 2026-08-09: The approved live catalog is benign browsing, connection burst,
  bounded request flood, slow connections, fixed-target port probing, and
  periodic beacon-like traffic. The console documents all seven model output
  classes but does not claim that a synthetic pattern is equivalent to an
  IoT-23 malware label.

## Validation

- Focused proof: bundle digest/schema/provenance tests, routing tests, cold-start
  rejection, production-schema label rejection, graph-window invariants, alert
  privacy tests, and matrix/oracle-selection tests.
- Integration proof: exact GKE bundle load; correctly routed and cross-head
  validation/test evaluation; API-to-alert flow; Elasticsearch-compatible event
  serialization; Helm-rendered service wiring.
- End-to-end proof: GKE ingress traffic reaches the target, collector,
  window builder, FedPer inference, alert router, and Kibana; capture latency,
  drop rate, false-positive baseline, metrics, and screenshots.
- Deployment proof: both Helm charts lint/template; ApplicationSet references
  existing paths; Ansible syntax-check; Terraform fmt/validate and a plan with no
  unintended replacement; Jenkins/GitHub path behavior tests.
- Repository-required checks: full Python tests, compile, lint, `git diff
  --check`, no executable references to removed roots, no source-artifact
  mutation, and no cloud apply/deploy command.

Observed checkpoint evidence on 2026-08-06:

- Exact schema-v2 research bundle
  `fedper-gke-14339380272482304688-r0030-42642e4cc839-b02` is runtime-readable,
  read-only (`0555` directories / `0444` files), contains six heads, loads
  without Flower, and is rejected by strict live readiness as intended.
- Correctly routed validation/test fixed-7 macro-F1 remains
  `0.9941709665` / `0.9940726453`; accuracy gap is `0.000119` and macro-F1 gap
  is `0.000098`. Cross-head and validation-selected oracle evidence was not
  recomputed after test selection.
- Eleven application tests and five legacy core-contract tests pass inside the
  application image. Ruff, compile, digest-updater test, strict mapping JSON,
  `git diff --check`, and old-root/path checks pass.
- Application Helm lint/template passes, and enabling collector without a
  selected window fails rendering. Federated Helm, Terraform fmt/validate,
  Ansible syntax, and a remote-state-backed `terraform plan -refresh=false`
  previously passed with `No changes` after relocation.
- After the later GCP approval, a reviewed remote-state Terraform plan changed
  only the Jenkins SSH firewall and the two GKE master-authorized-network lists
  to the current operator IPv4 address: `0 added, 3 changed, 0 destroyed`.
  Applying that exact binary plan completed `0 added, 3 changed, 0 destroyed`;
  Central GKE access and all existing Argo CD applications then verified
  `Synced/Healthy`.
- The official `34-1` source was downloaded as a bounded preflight, matched its
  recorded SHA-256, and reconstructed the exact validation/test class support
  in the immutable derivation report. The GKE evaluation chart renders one
  digest-pinned batch Job, a temporary 1 GiB evidence PVC, and 14 GiB ephemeral
  raw workspace; server-side dry-run passed. Inference/demo/alert workloads stay
  disabled during validation selection.
- The dependency-complete container boundary passed all 120 repository tests.
  Sixteen application tests pass, including exact bundle/routing, label-free
  production schema, raw-source contract, window selection, event privacy, and
  overlap-safe alert emission. Ruff, compile, both Helm charts, Terraform
  format/validate, Ansible syntax, strict Elasticsearch mapping, server-side
  Kubernetes dry-run, and `git diff --check` pass.
- Commit `ede3b264c422b78b893657c40540316f4157ad50` passed all three GitHub
  Actions jobs and Jenkins build 30 produced application image digest
  `sha256:ccc98bfce2845a2bb649aca122c01dee3b8d6e390816dea0fb1e530ab2847ffd`.
  Jenkins builds 31 and 32 proved the environment-only loop guard. The Phase 3
  image digest and training configuration remained unchanged.
- The relocated live ApplicationSet briefly rendered a doubled values path.
  Commit `5247a14` corrected values files relative to the relocated federated
  chart, after which both `fedkube-central` and `fedkube-edge-01` returned to
  `Synced/Healthy` without a workload/image/training change.
- The manually synchronized `fedkube-detection` Argo CD Application is pinned
  to revision `c01eb447fb19913d073c7c35968d9b72bd35cd16` and deliberately has
  no automated sync. Its evaluation Job uses the immutable application image,
  an isolated 1 GiB evidence PVC, and node-local disposable raw workspace.
  Central/Edge remain `Synced/Healthy`; the detection Application is expected
  to remain `Progressing` until the batch Job finishes.

Later GKE rolling-window and alert-policy evidence on 2026-08-06:

- The validation selection report is stored locally at
  `artifacts/application/evaluation/window-gke-r0030/window-validation.json`
  with SHA-256
  `3e290b4ef65c75e511efc7976f152b94c9e48e2067163708f143974b97ffd644`.
  The selected validation metrics are accuracy `0.9043148880`, fixed-7
  macro-F1 `0.8950243416`, weighted-F1 `0.9023351280`, inference p50
  `1.3375 ms`, and p95 `2.0444 ms`.
- Serving bundle
  `fedper-gke-14339380272482304688-r0030-42642e4cc839-b02-serving-8d59392ff1`
  is immutable, serving-ready, and binds that validation report and protocol.
  Kubernetes Secret `fedper-model-bundle-serving-8d59392ff1` contains exactly
  the flattened serving artifacts; no credential is stored in the bundle.
- The locked test report is stored at
  `artifacts/application/evaluation/window-gke-r0030/window-test.json` with
  SHA-256
  `11e9bc1a94e0a6b01ff6446f39467e79c22373b9aeaa37304ebfa4de22f21601`.
  Test accuracy is `0.9288535923`, fixed-7 macro-F1 `0.9322982502`, and
  weighted-F1 `0.9284162681`; all seven per-class F1 values exceed their
  validation values. Test-minus-validation gaps are `+0.02454` accuracy,
  `+0.03727` macro-F1, and `+0.02608` weighted-F1, so this evidence shows no
  validation-to-test overfitting. Flow drop and batch-boundary prediction
  change rates are both zero. The lower rolling score relative to transductive
  graph evaluation is a serving-protocol gap, not evidence of overfitting.
- Alert-policy validation initially failed closed because the serving bundle ID
  was compared directly with the research bundle ID. Commit `14e59ff` changed
  the check to the explicit serving-to-source relationship, requires a
  serving-ready bundle, and added regression coverage. Jenkins build 41 then
  failed only because Trivy exhausted the VM disk. Removing unused Docker
  images recovered about 22 GiB; Jenkins was restarted to clear its temporary
  disk-offline state. Build 43 passed build and the critical vulnerability
  gate and published digest
  `sha256:63f4d46e3f4e03d6dd1e9f482b089793856d0f42cd2b4f43d0e94a10f80574e3`.
  Phase 3 paths/digests remained unchanged.
- The alert-policy report is stored at
  `artifacts/application/evaluation/window-gke-r0030/alert-policy-validation.json`
  with SHA-256
  `387af82833d0c5e2009fa0939b3c6c3397c518284e932d0a1c278f9b4dc9189e`.
  It is derived only from validation and intentionally has
  `selected_policy=null`. At confidence `0.80`, benign false-alert rate is
  `0.001372` and malicious alert recall is `0.821765`; at `0.85`, they are
  `0.000823` and `0.776353`. Selecting an operational threshold is blocked on
  the user-authorized maximum acceptable benign false-alert rate.
- Argo CD revision `e5f91fd8aaacc579a29f244cf3e9f047433917b6` is
  `Synced/Healthy` for detection, Central, and Edge. The alert-policy Job
  completed once in 2m33s. Its isolated 1 GiB evidence PVC remains bound until
  the threshold decision or an explicit evidence-storage cleanup.
- A Jenkins restart exposed that the legacy setup wizard still generated an
  unlock secret even though JCasC manages the admin account. Treating that
  bootstrap material as compromised, the live VM now runs with
  `-Djenkins.install.runSetupWizard=false` and the legacy unlock-secret file is
  absent. The durable Ansible systemd override enforces the same property and
  removes the legacy file; syntax-check passes. Jenkins is active and the
  `fedkube-main` pipeline job remains present.

Resume checkpoint after scientific window evaluation:

1. Obtain the maximum acceptable benign false-alert rate and select the
   confidence threshold from the validation trade-off without reading test for
   policy selection.
2. Record the policy in application configuration and serving provenance, then
   enable inference, demo target, collector, alert router, and the configured
   Elasticsearch/Kibana boundary through the manual Argo CD Application.
3. Run the lab-only live-ingress scenarios, verify event privacy and Kibana
   visibility, and capture latency, flow-drop, and benign false-positive
   evidence.
4. After evidence is copied locally, explicitly decide whether to prune the
   completed evaluation Job and its isolated 1 GiB PVC.

Approved demo-console continuation on 2026-08-09:

1. Build a responsive, same-origin FastAPI web console, bounded scenario
   runner, and privacy-reduced live monitor under `src/application/`.
2. Add application-owned Elasticsearch/Kibana resources and saved objects
   without using the Phase 3 logging workload as a runtime dependency.
3. Extend the application Helm chart with internal-only ingress, least-privilege
   network policy, Secret Manager-backed credentials, and explicit resources.
4. Validate locally, publish through Jenkins with `[application-only]`, then
   manually synchronize the Argo CD detection Application.
5. Execute all six approved lab patterns on Central GKE and retain UI,
   detection, latency, drop-rate, privacy, and Kibana evidence.

Implementation checkpoint on 2026-08-09:

- The approved `0.85` validation-selected policy is recorded in application
  configuration; automatic blocking remains false.
- The responsive console exposes the six approved patterns, documents all
  seven model classes, and keeps the traffic pattern separate from the model
  prediction. The runner rejects public/arbitrary targets, enforces parameter
  bounds, and permits one active run at a time.
- The GKE chart renders an internal NGINX LoadBalancer, console, observed demo
  target, collector, inference, alert router, application-owned
  Elasticsearch/Kibana, strict mapping bootstrap, data view, retained evidence
  PVC, and Secret Manager-backed entity hash key. Server-side dry-run passes.
- Central remains one `e2-standard-4` node. The demo profile adds 175m CPU
  requests with explicit limits; it neither resizes the node nor changes Phase
  3 workloads.
- The image builds, its UI/catalog smoke test passes, 26 application tests pass,
  and the dependency-complete repository boundary passes all 130 tests with 7
  environment-conditioned skips. Ruff, compile, JavaScript syntax, both Helm
  charts, Kubernetes dry-run, Terraform format/validate, Ansible syntax, JSON,
  shell syntax, and `git diff --check` pass.

## Result

Scientific evaluation, serving-bundle promotion, locked test execution, the
validation-selected alert policy, and internal demo-console implementation are
complete. The model shows no validation-to-test overfitting in the locked
rolling protocol. Live serving awaits the Jenkins immutable image release,
manual non-pruning Argo CD sync, six-scenario execution, and Kibana/UI evidence.
The plan remains active.
