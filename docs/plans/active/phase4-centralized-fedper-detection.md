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
- [x] Implement collector, alert router, Kibana assets, Docker, and Helm chart.
  The application-owned saved-object bundle contains a data view, five
  visualizations, a saved search, and a six-panel dashboard; Kibana 9.2.3
  imports all eight objects without warnings.
- [x] Implement the internal demo console, exact six-scenario catalog, bounded
  fixed-target runner, privacy-reduced live monitor, NGINX internal gateway,
  and application-owned ECK resources in the independent application chart.
- [x] Pass local scientific and live-ingress acceptance.
- [x] Run repository-wide validation and document remaining limitations.
- [x] Obtain separate approval before any GKE deployment or paid operation.
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
- 2026-08-09: The user approved a validation-calibrated multi-head serving
  extension after live diagnosis proved that GKE routes every demo flow from
  `sensor-34-1` only to head `34-1`. Production will encode each graph once,
  evaluate all six exact heads, retain the trusted-route result for
  explainability, and use only a validation-selected fusion policy for the
  primary detection decision. An `any-head` rule is explicitly forbidden:
  head `3-1` classified the known benign baseline as `Attack` above the current
  alert threshold. The policy must be selected on validation, locked before a
  single test evaluation, expose per-head disagreement without raw feature or
  probability data in Elasticsearch, and fail readiness when its provenance or
  head set does not match the serving bundle.

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

Live GKE console evidence on 2026-08-09:

- Argo CD revision `986eb92c63dfeb9c6c01eedfb0df666b54d2b8d7` reached
  `Synced/Healthy`. All six application services, the single-node
  Elasticsearch cluster, and Kibana are ready. The application-owned internal
  gateway is assigned `10.10.0.5`; no public endpoint was created.
- ECK required numeric application/curl UIDs, removal of its reserved security
  setting, and a GitOps drain/restore for the one-node Kibana Deployment.
  Kibana 9.2.3 is stable with a 512 MiB V8 heap and 1 GiB container limit. The
  bootstrap Job completed and imported the strict index template and data view.
- The six default scenarios completed with 311/311 successful requests:
  benign browsing 20, connection burst 60, bounded request flood 200, slow
  connections 10, fixed-target port probe 6, and periodic beacon 15. Collector
  evidence is 311 observations/windows, zero dropped flows, inference p50
  `65.012 ms`, and p95 `91.904 ms`.
- All 311 synthetic live flows were predicted `Benign` by trusted route
  `sensor-34-1 -> head 34-1`, with no policy-qualified alerts. This is retained
  as a valid negative result: the UI patterns are not IoT-23 ground truth and
  the approved threshold was not weakened to manufacture an alert.
- The follow-up sink contract records every privacy-reduced model decision in
  Elasticsearch with an explicit `is_alert` boolean; only non-Benign decisions
  above the validation-selected threshold count as alerts. This enables Kibana
  baseline/FPR visibility without changing model output or alert policy.
- Jenkins published the sink image at immutable digest
  `sha256:d53c2b4f8837e68600ae8408c1710c6976c181697eeb461e0c7d912fcd46b748`.
  Argo CD revision `0311cd16bd324e2b77187442eb1b3216c7cb75c6` rolled all six
  application services with bounded one-node `maxSurge=0` /
  `maxUnavailable=1` behavior and returned to `Synced/Healthy`.
- A five-request post-deployment benign baseline produced five monitor and
  Elasticsearch documents, all `Benign` with `is_alert=false`. The indexed
  documents contain only the 16 approved privacy-reduced fields; the forbidden
  raw IP, feature, probability, tensor, and ground-truth fields are absent.
- Commit `fe76ae77944c439f967317e3c6d2d9e6ebee2094` added the
  version-controlled Kibana saved-object bundle. GitHub Actions run
  `31280718082` passed all three jobs. The v4 bootstrap Job completed in seven
  seconds, and the live dashboard `fedper-detection-overview` reads back with
  six panels and six references. Argo CD is `Synced/Healthy` at that exact
  revision; Elasticsearch and Kibana are both green.
- The focused dependency-complete container run passes all 29 application
  tests, including saved-object reference/privacy tests. Ruff, compile, Helm
  lint/template, Kubernetes server-side dry-run, NDJSON parsing, and
  `git diff --check` also pass.

Live-prediction remediation approved on 2026-08-09:

1. Preserve missing numeric values through the production request as `null` so
   frozen preprocessing can recreate the train-time `*_missing` flags. An
   observed numeric zero remains a measured zero and must not be rewritten as
   missing.
2. Replace per-request fire-and-forget delivery from the demo target with one
   bounded worker queue, retry/backoff, and per-run delivery counters. The
   console must distinguish target attempts, delivered observations, collector
   accepts, predictions, late drops, and terminal delivery failures.
3. Add a validation-only Scientific Replay panel. Its fixtures use the fixed
   100th occurrence of each class from a predeclared owning client, retain at
   most the selected 60-second/50-flow context, pseudonymize IP identities while
   preserving graph equality, and never place expected labels in the production
   inference request. Replay output is educational validation evidence, not a
   new test metric or live-traffic claim.
4. Add a real Zeek JSON path rather than describing the ingress adapter as
   packet capture. Zeek is an opt-in target-pod sidecar pinned to LTS 8.0.9
   digest `sha256:c7dfad9ab8296b2994d113222e77a22ebc9c8963b2b1200b798484ac923bc94f`;
   its shipper is label-forbidding and uses the same bounded delivery contract.
   Enabling it disables the approximate ingress adapter to avoid duplicate
   observations. Required `NET_RAW`/`NET_ADMIN` capability and root capture are
   explicit in Helm values and must pass server-side admission before GKE sync.
5. Do not change the alert threshold, select a head by desired class, or claim
   the six synthetic HTTP patterns are IoT-23 attacks. Current evidence shows
   all six heads classify those completed HTTP/SF approximations as Benign;
   correct remediation is observation fidelity and a separate labeled replay.
6. Correlate Zeek records to a lab run by registering the active trusted
   `sensor_id` at the collector before the runner is released. The registration
   metadata is control-plane-only and is never added to the production model
   request. Because the Zeek path bypasses the target adapter, its adapter queue
   counters are explicitly shown as N/A while collector accepted/predicted and
   drop counters remain per-run.
7. Keep the current single Central node. Based on observed Phase 4 CPU usage,
   lower only Phase 4 scheduling requests (not limits) and preserve the model's
   memory headroom so the Zeek/shipper sidecars fit without resizing or creating
   billable infrastructure. Admission and live rollout must still prove the
   chosen requests are viable.

Observed diagnosis before remediation:

- The live buffer held 500/500 Benign predictions, all in confidence bucket
  `0.95-1`, while inference readiness and all request paths were healthy.
- The same live production endpoint predicted deterministic held-out replay
  samples for all seven classes when labels were retained only by the local
  evaluator. Locked rolling-window test accuracy/macro-F1 remain
  `0.9288535923` / `0.9322982502`.
- The approximate adapter emits `tcp/http`, `SF`, `ShADadFf`, and one packet in
  each direction for every scenario. By contrast, test DDoS is 99.1% `OTH`,
  PortScan is 99.5% `S0`, HeartBeat/Okiru are predominantly `S0`, and Attack is
  predominantly SSH. All six heads return Benign for each approximate profile,
  so routing or threshold changes would not repair the mismatch.
- Since the current target pod started, it logged 43 failed collector
  deliveries; the collector accepted 517 observations, produced 514 windows,
  and late-dropped three flows. The UI's target HTTP success count therefore
  overstates end-to-end prediction coverage and must be split into stage-level
  counters.

Live remediation evidence on 2026-08-09:

- The dependency-complete application image passes 36/36 application tests;
  nullable production values recreate train-time missing flags, bounded
  delivery retries are counted, registered Zeek runs remain outside the model
  payload, and the Zeek tailer reopens rotated/truncated `conn.log` files.
  Ruff, compile, JavaScript syntax, Helm lint/template, server-side GKE dry-run,
  and `git diff --check` pass. GitHub Actions run `31303189781` passed all three
  jobs; its Python job passed the full 140-test repository boundary.
- Jenkins builds 59 and 61 succeeded. The final application image is immutable
  digest `sha256:df0d4f0173ed4bdde6b489f8a22573302c2d746464705678fcaf1763b1458582`
  and release ID `96a84f4e9763e707d7a57de46e293ab0e417f0a1`.
- Argo CD alone synchronized the GKE stack at revision
  `9aaa5700c79958da39311bc73e654ca90cf985e4`; status is `Synced/Healthy` and
  every application Deployment is 1/1 ready. The target pod has target, pinned
  Zeek 8.0.9, and shipper containers at 3/3 ready with zero restarts.
- Real Zeek capture emits `http/SF/ShADadFf` records with measured duration,
  bytes, and packet counts. The capture filter excludes GKE health probes
  SNATed through `10.40.0.1`: a final five-request run produced exactly five
  Zeek records, five collector accepts, and five predictions with zero late
  drops, inference failures, alert-sink failures, or duplicates.
- All seven validation-only replay cases passed the production request
  contract with `request_contains_ground_truth=false`. Six fixed cases were
  classified correctly (Attack, C&C, C&C-HeartBeat, DDoS, Okiru, PortScan).
  The fixed Benign case was classified as PortScan at confidence `0.548787` and
  remains visible as a truthful error rather than being replaced post hoc.
- The final synthetic HTTP run still predicts Benign on trusted route
  `sensor-34-1 -> head 34-1`. Together with successful non-Benign replay, this
  confirms the inference/runtime path is functioning and that the live result
  is a domain/feature mismatch, not evidence that the endpoint always returns a
  hard-coded Benign class. Alert threshold `0.85` and trusted routing were not
  changed.

Vietnamese chart-monitor extension approved on 2026-08-09:

- Make every user-facing console string Vietnamese. Immutable model labels,
  protocol identifiers, sensor IDs, digests, and API field names remain exact
  technical contracts, but the UI presents a Vietnamese display name beside
  them when useful.
- Render the live monitor with repository-native SVG and no third-party chart
  dependency. Benign predictions have amplitude zero and therefore form a flat
  baseline. A non-Benign prediction rises according to the existing alert
  policy severity (`low=1`, `medium=2`, `high=3`) and is annotated with its
  Vietnamese class name.
- Preserve the validation-selected `0.85` alert threshold. A non-Benign result
  below that threshold is shown in amber as a model detection below the policy
  threshold; only `is_alert=true` uses the red policy-alert state. The selected
  traffic scenario never supplies the chart class or amplitude.
- Keep at most 80 newest predictions in the browser chart. Clearing the monitor
  is presentation-only and does not reset the collector cursor or server-side
  evidence.

Local implementation checkpoint on 2026-08-09:

- The console catalog, scenario descriptions, parameter units, scientific
  replay labels, model-class explanations, execution state, monitoring state,
  and user-visible API errors are presented in Vietnamese. Exact class labels
  remain visible only as model-contract identifiers.
- The wide responsive monitor now includes a native SVG history chart, a
  normal/detection/policy-alert legend, an accessible live-status banner, and
  explicit attack-class annotations. Its data path reads only collector model
  events and never reads the selected scenario when computing amplitude.
- JavaScript and Python syntax, Ruff, YAML/JSON parsing, `git diff --check`, and
  the 15 locally runnable application tests pass. The dependency-complete
  application suite was not rerun locally because this shell lacks
  `torch_geometric`; its four ML-dependent modules stop during collection.
  CI/Jenkins image validation and the GKE rollout are intentionally deferred
  to the next work session, leaving the current cloud revision unchanged.

Vietnamese chart-monitor GKE evidence on 2026-08-09:

- GitHub Actions run `31319109965` passed infrastructure, security, and Python;
  the dependency-complete Python job ran all 140 repository tests successfully
  and Ruff passed. Jenkins build 66 applied the application-only guard, built
  and import-smoked the application image, found zero CRITICAL vulnerabilities,
  and pushed immutable digest
  `sha256:3be1828c3d88c88fbdaac567b0a1419d39a046c6604a9506cc422f2c497a1104`.
- Jenkins committed only the application environment digest/release change at
  `30d097bded4b8d120213c0a9ca827516141b9b3c`; build 67 then proved the
  environment-only loop guard by skipping test, build, push, and GitOps-update
  stages. The federated image and environments were unchanged.
- Argo CD alone rolled out the application and reached `Succeeded`, `Synced`,
  and `Healthy` at revision `30d097bded4b8d120213c0a9ca827516141b9b3c`.
  All six application Deployments and Kibana are available with zero restarts.
- Gateway smoke tests read the Vietnamese title, live-monitor heading, scenario
  catalog, scientific-replay labels, and SVG chart from the deployed image. A
  five-request benign run completed 5/5 and the collector produced seven
  accepted/predicted rolling windows, all `Benign`, with zero drops, inference
  failures, sink failures, or policy alerts. The chart semantics therefore
  retain the flat normal baseline for this live evidence.
- A validation-only DDoS replay was independently predicted `DDoS` with
  confidence `0.9984752536` and `request_contains_ground_truth=false`. A Node
  semantic harness verified `Benign=0`, below-threshold non-Benign `=1`, and a
  high-severity alert `=3`, plus the Vietnamese PortScan display label. This
  proves the non-Benign chart branches without feeding validation labels or
  synthetic scenario names into the live monitor.
- Operator-only port forwards are active at `127.0.0.1:18080` for the console
  and `https://127.0.0.1:15601` for Kibana. Kibana reports `available`.

Multi-head fusion checkpoint on 2026-08-09:

- Live diagnosis confirmed the production endpoint encoded a graph and invoked
  only trusted head `34-1`; the existing all-head runtime path was restricted to
  evaluation. The same 50-flow connection-burst window produced `Attack` from
  heads `3-1` and `39-1`, while the other four heads returned `Benign`. Head
  `3-1` also returned `Attack` above threshold for a known benign baseline, so
  `any-head` alerting was rejected with direct evidence.
- The new API encodes once, invokes all six exact heads, emits a primary fused
  decision plus trusted-head and per-head diagnostics, and fails readiness when
  the policy does not match the bundle/model/dataset/protocol/head digests or
  validation provenance. The collector stores only trusted label,
  disagreement count, decision mode, and policy digest in Elasticsearch; head
  probability vectors remain excluded.
- A first class-F1 weighted candidate was rejected before test/deployment
  because validation fixed-7 macro-F1 was only `0.759394`, below the existing
  trusted rolling result `0.895024`. A temporally separated validation stacking
  selector trained on the first 70% of each client and selected on the final
  30%; `logistic-log-probability-balanced-c10` achieved validation macro-F1
  `0.943932`, accuracy `0.944033`, and zero benign alerts in 967 selection
  examples. Validation report SHA-256 is
  `21c551a1d99b183e6ad24765b8df7c3107524b3ae0400539514842f85e281b4b`;
  locked policy digest is
  `0beef419f9cb7da3239ec43d12cdce174ddf37e1597bbee4c22c7979d292b951`.
- The policy was frozen before one locked test execution. Test macro-F1 is
  `0.938705`, accuracy `0.941692`, and weighted-F1 `0.944668`, showing the
  classification fusion generalized. Alert calibration did not pass its gate:
  benign false-alert rate rose to `0.010007` (73/7,295), above the retained
  `0.001` limit, despite malicious alert recall `0.920680`. Therefore no GKE
  rollout or GitOps environment image update was performed from this checkpoint.
- The validation-only seven-case smoke shows five of six attack classes
  correctly under fusion; PortScan and the known erroneous Benign fixture both
  collapse to the same `C&C` result and confidence. This remains evidence that
  scenario/feature fidelity cannot be repaired by choosing a head or by an
  ensemble alone.
- Zeek-mode console status no longer polls the observed target, and registering
  a new lab run resets that sensor's graph buffer and flow correlation map.
  This removes management-traffic capture and cross-run window contamination
  before the next live calibration.
- Because the locked fusion threshold exceeded the false-alert gate, the GKE
  values select `trusted-shadow`: fusion remains the UI/diagnostic decision,
  while policy alerts continue to use trusted head `34-1`. Elasticsearch
  records both labels and the decision source without probabilities. Promotion
  to fusion alerts remains gated on deployment-domain benign calibration.
- Focused application validation passes 43 tests, Ruff, JavaScript syntax,
  Helm lint/template, JSON validation, `git diff --check`, real-bundle API
  readiness, and a built-image smoke. The dependency-complete repository image
  was not used for the broad test command; the application image's broad run
  retained only the known missing matplotlib/private Phase 1 fixture failures
  outside this change, plus 7 optional-Flower skips.

Multi-head fusion GKE rollout and live evidence on 2026-08-09:

- GitHub Actions run `31323047141` passed infrastructure, security, and the
  dependency-complete Python job. Jenkins build 69 passed import smoke and
  CRITICAL vulnerability scanning, pushed immutable application digest
  `sha256:4e0aa072f204dcac7c5176d9b51f93399cf84581ca393591d366d8d1814d23d8`,
  and committed only the GKE application environment update at
  `f31461dcb0d0c502b732078cad4dfbf81f12dba0`.
- Argo CD alone synchronized that revision and reports `Synced/Healthy`. All
  six application Deployments are ready, Elasticsearch is green, Kibana is
  available, and bootstrap job `eb-v5-fusion` completed. Inference readiness
  exposes all six client IDs, decision mode
  `validation-calibrated-multi-head-v1`, and policy digest
  `0beef419f9cb7da3239ec43d12cdce174ddf37e1597bbee4c22c7979d292b951`.
  Collector readiness confirms `ALERT_DECISION_SOURCE=trusted-shadow`.
- A 20-request benign live run produced fused `Attack` decisions while trusted
  head `34-1` remained `Benign`; heads `3-1` and `39-1` selected `Attack` and
  the other four selected `Benign`. No policy alert was emitted. This directly
  confirms both that every head is invoked and that promoting fusion alerts
  would currently create benign false positives.
- After an eight-second quiet-drain proved the Zeek cursor stable, a clean
  periodic-beacon run completed 15/15 requests and exactly 15 collector
  accepts/predictions with zero drops or downstream failures. It produced the
  same fused `Attack` / trusted `Benign` split and zero policy alerts. The six
  synthetic HTTP patterns therefore still do not provide model-discriminating
  IoT-23 flow features; multi-head routing cannot manufacture that fidelity.
- Running high-volume scenarios back-to-back exposed a separate evidence
  isolation limit: Zeek can emit terminal connection records after the runner
  has completed, and those records can be attributed to the next registered
  run even though its graph buffer is reset. For example, a six-connection port
  probe received 55 records while the preceding flood drained. Until the Zeek
  flow can be joined to a run token, controlled comparisons require a verified
  quiet cursor between runs and must not interpret rapid-run counters as
  scenario-specific evidence.
- The console distinguishes an amber fused detection from a red policy alert.
  In `trusted-shadow`, its explanation now states that the alert decision came
  from the trusted head rather than incorrectly describing every non-alerting
  fused detection as merely below a confidence threshold.

Compact class-first console approved on 2026-08-09:

- Replace the verbose multi-section page with a two-column desktop dashboard
  constrained to one viewport: exact model-class selection and validation
  details on the left; metrics, attack-signal chart, latest prediction, and six
  head diagnostics on the right. Small screens may scroll rather than making
  controls inaccessible.
- Model classes are immutable contract labels and must be shown verbatim:
  `Benign`, `Attack`, `C&C`, `C&C-HeartBeat`, `DDoS`, `Okiru`, and
  `PartOfAHorizontalPortScan`. Do not substitute Vietnamese class names.
- The class selector executes the existing fixed validation-only scientific
  replay. It must show sensor, trusted head, flow-window size, confidence, and
  top-three model outputs while retaining the rule that expected labels never
  enter the production inference request.
- Keep the attack-signal SVG and the live collector monitor. A replay result may
  add its model-predicted class to the browser chart only when visibly marked as
  validation replay; it is never counted as a live policy alert. Remove the
  verbose traffic-pattern controls, duplicate class glossary, long run counter
  block, and unbounded timeline from the primary screen without removing their
  backend APIs.

Compact console GKE evidence on 2026-08-09:

- Commit `d39f85009c903ed8f0fe0213ce3fb30d88f16008` passes the six focused
  console tests, JavaScript syntax, Ruff, DOM-reference audit, Helm
  lint/template, and `git diff --check`. GitHub Actions run `31324040264`
  passed Python, security, and infrastructure.
- Jenkins build 73 honored the application-only gate, passed image import and
  CRITICAL vulnerability checks, and pushed immutable digest
  `sha256:ea609a38979a9f7052f96d61ea3d636053ab90d44834036908d3dd20ee73c505`.
  Its GitOps environment commit is
  `b0f609afd690a58406bc2016a433782afcd9d680`.
- Argo CD alone synchronized that revision and reports `Synced/Healthy`. All
  application pods are ready with zero restarts. Deployed HTML contains the
  compact selector, SVG attack chart, and six-head panel; deployed JavaScript
  binds button names directly to `expected_class`; deployed CSS constrains the
  normal desktop layout to `100dvh` with page overflow disabled while retaining
  a small-screen scrolling fallback.
- The deployed catalog returns the seven exact labels in training order. All
  seven selector actions reached production inference with
  `request_contains_ground_truth=false`. Five attack fixtures were correct:
  `Attack` (`0.999671`), `C&C` (`0.937061`), `C&C-HeartBeat` (`0.988157`),
  `DDoS` (`0.999932`), and `Okiru` (`0.994773`). The fixed `Benign` and
  `PartOfAHorizontalPortScan` fixtures both predicted `C&C` at `0.538746` and
  remain visible as errors rather than being relabeled for the demo.

Continuous monitor and false-alarm remediation approved on 2026-08-10:

- Keep the raw model result visible. In particular, the fixed one-flow Benign
  validation fixture remains `C&C` at `0.538746`; the UI must not rewrite that
  scientific error to `Benign`.
- Separate raw classification from the alert decision using the immutable
  validation-selected class thresholds in
  `configs/application/multi-head-fusion-policy.json`. The replay endpoint must
  verify that the inference response and decision policy have the same digest.
  A non-Benign argmax below its class threshold is `below-threshold`, emits no
  alert, and contributes zero to the attack-rate graph.
- Replace event-arrival-only plotting with a fixed one-second browser sampling
  clock. Each sample records the number of policy-qualified detections received
  during that interval; no detections append a Benign zero, so the graph keeps
  moving and returns to baseline. Presentation bands are 1-2, 3-5, and 6+
  accepted detections/second; they do not alter model or alert policy.
- The class catalog owns a concise behavioral profile for all seven immutable
  labels. The API also derives exact protocol, service, connection-state,
  destination-port, duration, and packet statistics from each fixed validation
  replay window. The UI must distinguish general behavior from properties of
  the selected fixture, especially the one-flow Benign and port-scan examples.
- Validate the replay decision contract, label-free production request, profile
  completeness, continuous zero sampling, attack-frequency bands, JavaScript,
  Helm wiring, and the deployed GKE behavior. Deployment remains GitOps-only.

Continuous monitor GKE evidence on 2026-08-10:

- Application commit `1d9b39751f70b854bafc2e5270433c292f404180`
  passed 13 focused console/remediation tests, JavaScript syntax, Ruff, JSON,
  Helm lint/template, `git diff --check`, a local production-image build, and a
  production-image console readiness/catalog smoke. GitHub Actions run
  `31327614917` passed Python, security, and infrastructure.
- Jenkins build 76 honored the application-only gate, passed image import and
  CRITICAL vulnerability scanning, and pushed immutable application digest
  `sha256:6a385cbf1b42e15a08e601c768dea3fe5876fc93e4107138eefe57656e71d4a5`.
  Its environment-only commit is
  `84dc5e2e33161e866a57257dab1f041d172d1566`. The detection Application keeps
  its intentional reviewed/manual sync policy; an explicit Argo CD core sync
  performed the rollout. Argo CD reports `Synced/Healthy`, and all application
  pods are ready with zero restarts.
- Direct GKE replay proved the remediated boundary. The fixed Benign fixture
  still reports raw `C&C` at `0.538746`, below the immutable C&C threshold
  `0.761949`, so its decision is `below-threshold` and `is_alert=false`.
  `Attack`, `C&C`, `C&C-HeartBeat`, `DDoS`, and `Okiru` are correctly classified
  and policy-qualified. The one-flow `PartOfAHorizontalPortScan` fixture remains
  raw `C&C` at `0.538746` and below-threshold; this unresolved scientific model
  error is not relabeled for the demo. Every replay response confirms that the
  production request contained no ground truth.
- Firefox WebDriver BiDi exercised the deployed page rather than only reading
  static assets. It observed all seven immutable class labels, an 80-point
  one-second chart, and the selected fixture's protocol, service/state, port,
  packet, byte, duration, sensor/head, behavior, and limitation fields. Benign
  replay produced zero chart spikes and retained the Benign banner. Attack
  replay produced one labeled `Attack · 1/s` spike; the next quiet sample
  returned the current rate to `0 detection/s` and the banner to Benign while
  retaining the historical spike in the rolling 80-second view.

Session metric and prediction-panel correction approved on 2026-08-10:

- The compact dashboard metrics are session-scoped and must update for both
  collector-driven live inference and validation replay. Collector windows and
  policy alerts remain the live source of truth; successful replay calls add
  one replay inference and, only when their calibrated decision is `alert`, one
  replay qualified detection. The UI shows the live/replay split and must not
  describe a replay-qualified detection as a live policy alert.
- Replay round-trip latency is measured in the browser and shown as replay p95
  after at least one replay; otherwise the collector's live inference p95 is
  shown. Flow-drop rate remains collector-only because replay bypasses flow
  collection. Clear establishes a new live counter baseline and resets replay
  counters, events, and chart samples.
- Remove the FedPer-head diagnostics panel from the compact primary screen.
  Preserve multi-head inference and event payloads in the backend, but use the
  full bottom width for a longer latest-prediction history. This is a display
  change only and does not alter routing, fusion, or alert policy.

Session metric correction GKE evidence on 2026-08-10:

- Application commit `be448119c6e678f3f9deb4e4d3265865a2fe36d2` passed the
  focused console and live-remediation tests (13 total), JavaScript syntax,
  Ruff, Helm lint/template, and `git diff --check`. GitHub Actions run
  `31351132677` passed Python, security, and infrastructure.
- Jenkins produced environment-only commit
  `1f5e63aa2207e206c8da0b322bb2f2c57dcd899c`, pinning immutable application
  digest
  `sha256:52dc8a503f142b222fd6c390ad045779ee0cad7b3dab4ade25bda5305674e902`.
  An explicit Argo CD core sync rolled out that revision; the detection
  Application reports `Synced/Healthy`, and all long-running application pods
  are ready with zero restarts.
- Firefox WebDriver BiDi exercised the deployed GKE console through its local
  port-forward. Initial session counters were `0/0`. A Benign validation replay
  changed inference to `1` with source `live 0 · replay 1` while qualified
  detections remained `0`. Attack changed inference to `2` and qualified
  detections to `1`, with source `live 0 · replay 1`; replay round-trip p95 was
  populated. Clear reset both session counters and latency to their initial
  state.
- The deployed DOM contains neither the former head grid nor its heading. At
  the tested desktop viewport, Latest Prediction and its parent bottom monitor
  each measured 964 pixels wide, and the prediction history rendered in two
  columns. The backend's multi-head diagnostics remain unchanged.

Scientific traffic-profile and generator-VM extension approved on 2026-08-10:

- Before creating executable traffic profiles, derive a seven-class reference
  from the exact IoT-23 validation replay under
  `artifacts/application/replay/exact-gke-r0030`. Verify every validation file
  against its manifest digest. The source is the deterministic held-out sample
  reconstructed from six official IoT-23 `conn.log.labeled` files; the locked
  test split must not influence profile construction or tuning.
- The reference analysis covers protocol, service, connection state, history,
  destination port, missingness, duration, bytes, packets, IP bytes, missed
  bytes, inter-arrival time, 60-second flow density, and source/destination
  topology. It records support and owning clients so a high-frequency class is
  not mistaken for a universally reproducible behavior.
- Executable profiles are fixed-target, versioned, bounded, and fail closed.
  They never accept an arbitrary hostname, IP, port, URL, packet payload, or
  shell command. A selected profile is control-plane metadata and never enters
  the inference request or chooses a head, label, confidence, alert, or chart
  amplitude.
- A profile is only a candidate until Zeek-observed output is compared with the
  validation reference. Acceptance uses a deterministic within-class bootstrap
  envelope on predefined observable features plus nearest-reference checks;
  profiles outside that envelope remain visibly non-equivalent. The locked test
  split may be read exactly once only after the profile and comparison protocol
  are frozen.
- Separate model-equivalence from natural-timing equivalence. The stratified
  validation timeline is authoritative for the model's locked rolling-window
  view but not the original malware arrival process. Phase 4 may target the
  former and must label it accordingly; a natural-timing claim requires a new
  contiguous train-only reconstruction from the digest-pinned sources.
- Add one private Compute Engine traffic-generator VM in the existing Central
  subnet with no external address, OS Login/IAP administration, a least-
  privilege service account, and firewall access restricted to the internal
  detection gateway. Use the smallest viable E2 shape and a small standard
  persistent disk; do not resize GKE or create a public attack endpoint.
- Configure the VM through Ansible with a non-root systemd traffic agent. The
  web console may request only catalogued bounded profiles through an
  authenticated internal control path. Register and reset the collector run
  before release, wait for a quiet Zeek drain between runs, and expose
  attempted/captured/accepted/predicted/drop/failure counters separately.
- The current target-side Zeek capture sees the gateway-to-target connection,
  not the original generator identity. The first implementation may prove the
  complete model path through the existing internal gateway, but scientific
  topology evidence requires capture at the ingress boundary with observed
  source preservation. Validate the actual Zeek source/destination fields on
  GKE before claiming that boundary is complete.
- Terraform plan must show only the approved generator VM, service account/IAM,
  and narrowly scoped firewall/control resources, with no replacement of
  existing GKE, Jenkins, storage, network, or state resources. Keep the apply
  behind a separately reviewable plan checkpoint even though the user approved
  the extension, because exact cost and resource changes must remain visible.

Traffic-profile/VM extension progress:

- [x] Locate the exact digest-verified IoT-23 validation replay and preserve the
  locked test split.
- [x] Implement and validate the seven-class reference-profile analysis.
- [x] Publish the immutable local reference and tracked scientific boundary at
  `docs/PHASE4_TRAFFIC_PROFILE_ANALYSIS.md`. Reference digest is
  `1170c604041f572ef17c2cd12b16b5274a086a8d2a90a064c97e2315ff98b170`;
  12,144 validation rows were analyzed and locked test was not read.
- [x] Freeze the seven-class executable candidate catalog. Benign is explicitly
  a control, SSH/C&C remain blocked pending compatible targets, DDoS remains
  unsupported because the validation fingerprint depends on checksum/history
  evidence, and the three SYN-only candidates are bounded to fixed private IPs.
- [x] Freeze the deterministic bootstrap/nearest-reference comparator before
  interpreting any captured VM flow as class-equivalent. It reads only the
  digest-verified validation replay, uses 2,000 deterministic within-class
  resamples, checks categorical Jensen-Shannon distance, numeric/topology
  envelopes, and requires the selected class to be the nearest reference. A
  real validation Okiru sample passes the frozen implementation; locked test
  remains unread.
- [x] Implement the authenticated bounded traffic agent and local contract
  tests. Its request contract accepts only `profile_id`; target/port/event
  authority stays in the versioned server-side catalogs.
- [x] Extend Terraform, Ansible, Helm, CI, the collector authentication boundary,
  and compact-console backend. Keep the GKE environment flag disabled until the
  Secret Manager versions and VM services are healthy, so Argo CD does not
  consume an intermediate state.
- [x] Produce a refreshed remote-state Terraform plan at
  `artifacts/federated/terraform/phase4-traffic-generator-plan.txt`: exactly
  `18 add, 0 change, 0 destroy`. The additions are one private `e2-small` VM
  with 20 GiB standard disk, one service account, four internal addresses, six
  firewall rules, two Secret Manager containers, and four secret-level IAM
  bindings. Private egress is restricted to the six fixed lab endpoints.
- [x] Record the current price estimate. The 2026-08-09 Singapore Cloud Billing
  catalog rates used for the conservative calculation are USD 0.02690931 per
  E2 core-hour and USD 0.003786237 per GiB-hour. At the documented `e2-small`
  allocation (0.5 fractional vCPU, 2 GiB), compute is approximately USD 0.02103
  per hour / USD 15.35 per 730-hour month. A 20 GiB standard disk at USD
  0.000054795 per GiB-hour is approximately USD 0.80/month before any account
  free-tier credit. Budget roughly USD 16.15/month plus small Secret Manager,
  NAT/network, and tax/currency effects; stopping the VM removes compute usage
  but retains disk/storage charges.
- [x] Record final apply and live evidence. The approved binary plan was
  applied on 2026-08-10: 17 non-compute resources were created, but GCP rejected
  the VM because global CPU quota is already `12/12` (Central 4 + Edge 6 +
  Jenkins 2). A refreshed state-backed plan now shows exactly `1 add, 0 change,
  0 destroy` for the VM. The user selected rotation on 2026-08-10; no
  quota-increase request was submitted. Jenkins was stopped, the reviewed
  one-add plan created private `e2-small` instance
  `fedkube-traffic-generator` at `10.10.0.20`, and Ansible configured the
  bounded agent, VM-local Zeek, and authenticated shipper. After live evidence,
  `scripts/switch_demo_compute.sh ci` stopped the generator before restarting
  Jenkins, preserving the `12/12` CPU ceiling.
- [x] Apply only after the plan checkpoint, then capture live GKE/Zeek/model
  evidence and evaluate fixed profiles without relabeling model output.

Live private-generator evidence on 2026-08-10:

- Argo CD alone synchronized revisions `91734c9` and `2004f30`; the detection
  Application is `Synced/Healthy`. Both Secret Manager-backed ExternalSecrets
  are ready, the internal gateway retains `10.10.0.5`, port `8082` is dedicated
  to authenticated observations, and the former target-side Zeek sidecars were
  removed to prevent duplicate flow capture. A one-time Argo-scoped Deployment
  replace repaired legacy apply ownership without replacing the Service or
  changing its address.
- `command-control-heartbeat`, `okiru`, and `horizontal-port-scan` completed
  respectively `2/2`, `2/2`, and `3/3` attempted/successful sends. Collector
  counters exactly matched each run, every accepted flow produced a prediction,
  and all runs recorded zero late drops, duplicates, inference failures, and
  alert-sink failures. Multi-head fusion predicted the three intended classes;
  the trusted `34-1` shadow decision differed for heartbeat and Okiru.
- The frozen 2,000-sample validation comparator selected the intended class as
  nearest for all three candidate profiles but rejected every candidate from
  the reference envelope. Live Zeek encoded `service`, missing duration/byte
  fields, and in one case IP-byte totals differently from the IoT-23 reference.
  These runs remain reproducible detection demonstrations, not class-equivalent
  malware traffic. The locked test split was not read.
- The two-flow benign control completed with no pipeline errors but fusion
  classified both windows as `Attack` at confidence bucket `0.95-1`; trusted
  head `34-1` classified both as `Benign`. Keep `trusted-shadow`: promoting
  fusion alerts would create a 100% false-alert rate on this bounded control.
- Real replay exposed and locally corrected two fail-closed defects: the
  comparator now binds the catalog to `derived_dataset_digest` rather than the
  unrelated source digest, and collector-registration failure now cancels the
  gated agent run with `DELETE /v1/runs/current` instead of releasing traffic.
  Seventeen focused host tests pass. The host cannot collect PyG-dependent
  tests, while the dependency-complete application-image boundary passes all
  63 application tests plus eight subtests; CI remains the release proof.
- Jenkins build 88 honored the application-only gate, passed its image import
  and CRITICAL vulnerability scan, then published immutable digest
  `sha256:e5e0cd09f880bb02293260291a8cc078b160b52811e84e8feb00e1ebfa58379a`.
  Build 89 passed the environment-only loop guard. Argo CD alone deployed the
  resulting revision `74c43ba`; all six application Deployments are ready with
  zero restarts and the detection Application reports `Synced/Healthy`.

Traffic-controller UI extension approved on 2026-08-10:

- Preserve the one-viewport monitor and replace the single left control surface
  with two tabs: validation-only Model replay and live Traffic generator. The
  traffic tab exposes all seven fixed profiles, including the scientific status
  and fixed mechanism/target group/port/event interval/expected observables.
  Blocked profiles remain inspectable but cannot be started.
- UI start requests contain only `profile_id`. Repository catalogs remain the
  authority for target, port, event count, timing, and scientific boundary. The
  run panel polls attempted/successful sends and collector receive/accept/
  prediction/drop/failure counters. Stop calls the authenticated agent's fixed
  current-run cancellation endpoint; it cannot identify an arbitrary process
  or target.
- Correct the console observation label to `zeek` when capture is VM-local, not
  only when the legacy target-side Zeek container is enabled. Local validation
  passes 14 focused control tests, all 64 application tests plus eight subtests,
  JavaScript syntax, Ruff, Helm lint/template, and `git diff --check`.
- [x] Publish the immutable application image, deploy only through Argo CD, and
  exercise profile selection, start, live counters, stop, and the one-screen
  layout against the GKE traffic-generator VM.

Traffic-controller UI GKE evidence on 2026-08-10:

- Jenkins build 91 passed the application-only build/import/CRITICAL-scan/push
  path and published digest
  `sha256:b87611f54efac68bf675b75a911f8b302951e5a8ed7b43f097c76fa8beafe7e1`.
  Build 92 passed the environment-only loop guard. Argo CD alone deployed
  revision `4978b0c`; the detection Application is `Synced/Healthy`.
- The deployed traffic API returns all seven exact model class labels. Benign,
  C&C-HeartBeat, Okiru, and PartOfAHorizontalPortScan are executable; Attack,
  C&C, and DDoS remain inspectable but start-disabled with their scientific
  blocking reason. The deployed console reports observation mode `zeek`.
- Firefox exercised the deployed GKE page at 1440 x 900. It observed seven
  profile controls, an inspectable blocked profile with Start disabled, an
  executable candidate with Start enabled, and a ready traffic agent. Browser
  viewport and document heights both measured 814 px, proving the desktop page
  has no vertical scroll.
- Starting C&C-HeartBeat and pressing Stop during its 30-second interval yielded
  status `cancelled` after one successful send. Zeek/collector subsequently
  reported exactly one received, accepted, and predicted flow with zero drop or
  downstream failure; the fusion prediction was C&C-HeartBeat. The stop control
  therefore interrupts a live bounded run without bypassing the model path.

Dataset-shaped Attack/C&C/DDoS traffic extension approved on 2026-08-10:

- Reuse the existing internal gateway address and LoadBalancer. Add fixed
  protocol-emulator listeners behind TCP/22 and TCP/6667; do not add another
  load balancer, public endpoint, arbitrary target, Terraform resource, or
  external traffic destination.
- Attack uses a bounded SSH identification plus request/response exchange so
  VM-local Zeek can record a completed TCP/22 `ssh` connection. C&C uses a
  deterministic mixture of SYN-only traffic to the existing private blackhole
  and bounded IRC-like sessions through TCP/6667. Neither selected profile nor
  requested controls enter inference or select a model head/class.
- DDoS emits a bounded, checksum-invalid ACK-only TCP/80 burst to the fixed
  private blackhole so checksum-aware Zeek records OTH/history `C` flows and
  the rolling window reaches the dataset's 50-flow view. VM TX/GSO/TSO offload
  is disabled rather than weakening Zeek checksum validation. UI and reports
  must still describe it as a validation-shaped candidate until the frozen
  comparator accepts it, not as class-equivalent malware reproduction.
- Every profile exposes only `events` and `interval_ms` controls. Their
  per-profile min/max bounds live in the signed catalog, the server enforces a
  two-minute maximum scheduled duration, and requests still cannot supply a
  target, port, payload, command, sensor, head, or label. The UI also displays
  the derived event rate.
- Validation requires contract/unit tests, real Zeek `conn.log` evidence for
  Attack/C&C/DDoS, collector accepted/predicted counters, and observed model
  output. Deployment remains GitOps-only; Jenkins builds the immutable image,
  Ansible updates the private VM agent, and Argo CD alone changes GKE.

Dataset-shaped traffic extension progress:

- [x] Reconfirm the digest-pinned validation fingerprints without reading the
  locked test split.
- [x] Implement bounded profile controls and SSH/IRC generators.
- [x] Add the private gateway protocol emulator and validate Helm output. Use a
  two-revision rollout: publish the new image with the GKE emulator gate off,
  then enable the sidecar/listeners only after that digest is Healthy.
- [x] Run focused local validation: 21 affected tests, Ruff, JavaScript syntax,
  Helm lint, disabled/enabled Helm renders, Ansible syntax, and diff checks pass.
  Host-wide application collection remains unavailable because the host lacks
  `torch_geometric`; the dependency-complete Jenkins image remains the release
  boundary.
- [x] Publish through Jenkins/Argo CD, update the VM through Ansible, and record
  live end-to-end evidence for all three newly executable profiles.

First live extension audit on 2026-08-10:

- Jenkins build 93 succeeded and published application digest
  `sha256:40d6c917d2547e1a7acbb09376f337a3b801293f72da039e39472bef14b801d4`.
  The two-revision gate worked: Argo CD first reached Healthy with the emulator
  disabled, then revision `5a76010` added the sidecar and internal ports 22 and
  6667 on the existing `10.10.0.5` LoadBalancer. No new cloud resource or
  public listener was created.
- Ansible completed with `ok=16 changed=6 unreachable=0 failed=0`; the agent,
  VM-local Zeek, and shipper are active. The deployed API exposes all seven
  profiles plus bounded event/interval controls.
- Attack completed 4/4 sends and 4/4 collector/model predictions with no drop
  or downstream failure. Fusion predicted Attack at confidence bucket 0.95-1,
  but Zeek recorded `service=null`; the first emulator's arbitrary post-banner
  bytes caused the SSH analyzer not to retain the service. Do not accept that
  fingerprint.
- C&C completed 3/3 and 3/3 predictions with no failure. Zeek recorded one
  S0/unknown flow and two IRC flows; fusion predicted C&C on the 2- and 3-flow
  windows. The IRC flows closed as RSTO because the client left unread server
  responses; revise to graceful close before final acceptance.
- DDoS completed 50/50 and 50/50 predictions with no failure. Zeek produced
  OTH/history A, and all 50 windows were predicted HorizontalPortScan. This is
  direct evidence that the known checksum artifact materially affects the
  model; never overwrite the output label. The corrective candidate disables
  TX/GSO/TSO offload, enables Zeek checksum validation, and intentionally
  corrupts only catalogued ACK-only DDoS packets to target history C.
- The corrective SSH candidate uses valid, size-controlled KEXINIT framing:
  client application bytes target approximately the IoT-23 median 589 and
  server bytes approximately 1,801, fragmented into bounded packet counts.
  Twenty-one affected tests, Ruff, Ansible syntax, and diff checks pass. It
  still requires a second immutable release and live Zeek/model proof.

Final corrected live extension audit on 2026-08-10:

- Jenkins build 97 passed the image import and CRITICAL vulnerability scan and
  published immutable digest
  `sha256:83e36734931b83ac38b52ec74323fb5f8be68fdeafdc93a5896964fca16b9b32`.
  Argo CD alone deployed the application revision; the sidecar and existing
  internal LoadBalancer listeners remained ready without a new GCP resource.
- DDoS completed 50/50 sends and 50/50 predictions with zero late drop,
  duplicate, inference failure, or alert-sink failure. Zeek recorded exactly
  50 TCP/80 `OTH`/history `C` flows with no response bytes or packets. All 50
  rolling predictions were `DDoS`, reaching the `0.95-1` confidence bucket.
- C&C completed 3/3 sends and predictions with zero pipeline failure. Zeek
  recorded one S0/unknown flow and two graceful `irc`/`SF` flows. The first
  one-flow window predicted Okiru; the two- and three-flow windows predicted
  C&C at confidence bucket `0.95-1`. This preserves the observable cold-window
  behavior instead of rewriting it.
- Attack completed 4/4 sends and predictions in the corrected audit and a final
  one-flow verification also completed 1/1. Zeek recorded analyzer-confirmed
  `ssh`, `SF`, history `ShAdDaFf`, approximately 3.41 seconds, 585 origin bytes,
  1,803 response bytes, 20 origin packets, and 19 response packets. The model
  predicted `Attack` at confidence bucket `0.95-1`. A low-priority Zeek hook
  retains the `ssh` service only when the SSH analyzer actually populated the
  connection's SSH record; profile metadata never supplies that service.
- All seven profiles now expose catalog-bounded event count and inter-event
  interval controls. The deployed server enforces each profile's min/max and a
  two-minute maximum schedule; the UI displays derived events/second. Targets,
  ports, payloads, labels, sensors, model heads, and routes remain immutable.

Monitor chart refinement approved on 2026-08-10:

- Preserve the alert policy and add every non-Benign model decision to the
  one-second chart buffer. A shadow/below-policy detection is amber and a
  policy-qualified `is_alert=true` decision is red; Benign remains the cyan
  baseline. This is a presentation change only and must not promote a fusion
  decision into an alert.
- Replace rigid polyline transitions with bounded cubic Bézier segments. Keep
  the class label only at the end of the latest detection groups so burst runs
  remain readable, and use a smaller chart annotation font while increasing
  the primary status, metric, legend, and prediction text.

Monitor chart refinement evidence on 2026-08-10:

- Nineteen focused catalog/agent/API/console tests, JavaScript syntax, Ruff,
  and `git diff --check` pass. Jenkins build 99 passed the application image
  import and CRITICAL vulnerability scan and published immutable digest
  `sha256:f533ca7b841eb73fca57ac96601990662587ab1e5f27c32292055b06a5347610`.
  Argo CD alone deployed revision `68b417e`; the detection Application is
  `Synced/Healthy` and all seven Deployments are ready.
- A 1440 x 900 headless browser render of the deployed console remains within
  the single-screen layout. It contains the Benign/model-detection/policy-alert
  legend, larger primary copy, a smooth cyan baseline, and the deployed
  JavaScript contains the amber/red Bézier segment renderer.
- After the restarted Zeek/shipper reached a stable observation count, a clean
  one-event Attack run completed 1/1 send, receive, accept, and prediction with
  zero drop or downstream failure. Fusion returned `Attack` in confidence
  bucket `0.95-1`, trusted head returned `Benign`, and `is_alert=false`; this is
  the exact live event now admitted to the amber chart path without changing
  the policy decision.
- The first post-boot probe was excluded because the shipper drained old DDoS
  observations while the run was being registered. Demo operation must wait
  for a stable observation count after VM rotation before treating a new run as
  isolated evidence.

Attacker/defender separation approved on 2026-08-10:

- Run a dedicated Attacker Console on the traffic-generator VM, bound only to
  `127.0.0.1:8090` and reached through an SSH/IAP local tunnel. Its backend may
  read the existing VM token files and orchestrate the authenticated agent and
  collector run boundary; neither token is sent to the browser. The page shows
  generator identity, fixed target, selected profile, bounded controls, and
  send/delivery counters, but no model prediction or alert.
- Remove the live traffic-generator controls from the GKE SOC Console. The SOC
  surface remains the independent read-only live model monitor; validation
  replay can remain visibly separated as scientific evaluation rather than an
  attacker control.
- Move Zeek capture and its shipper from the generator VM to the GKE gateway
  Pod so the sensor is on the victim ingress boundary. Set the Internal
  LoadBalancer Service to `externalTrafficPolicy: Local`, capture before NGINX
  terminates/proxies the connection, and accept this topology only after live
  `conn.log` proves `id.orig_h=10.10.0.20` for all relevant ports.
- Keep the agent run gated until the collector registration succeeds. Expose
  only the required registration/metrics endpoints on the existing private
  observation listener, protected by the current observation token and source
  NetworkPolicy. Do not expose monitor/model output to the attacker console.
- Disable and stop VM-local Zeek/shipper only after gateway capture passes, so
  a rollback can re-enable the two existing units. Avoid simultaneous shippers
  to prevent duplicate flows.
- If gateway-side capture cannot preserve the generator source, stop before
  creating new cloud resources. Packet Mirroring requires a separately
  reviewed Terraform/cost plan and remains an unimplemented fallback.

Attacker/defender separation progress:

- [x] Audit the actual packet path before moving capture. Benign/Attack and the
  IRC half of C&C traverse `10.10.0.5`, but DDoS, C&C-HeartBeat, Okiru,
  HorizontalPortScan, and the blackhole half of C&C terminate at
  `10.20.0.20-22` and never enter the GKE gateway. A gateway-only Zeek cutover
  would therefore silently remove coverage for most attack profiles. Keep the
  VM-local sensor/shipper as the continuity boundary while the UI/control
  separation is delivered; do not run a second shipper and create duplicates.
- [x] Implement the loopback-only Attacker Console and focused contract tests.
- [x] Remove traffic control from the SOC Console and retain monitoring proof.
- [x] Add the token-protected private run-control proxy and Helm validation.
- [ ] Prepare, but do not apply, a separately reviewable Packet Mirroring sensor
  Terraform/cost plan. Gateway-side capture is not a valid full-profile
  replacement because the fixed blackhole destinations bypass it.
- [x] Update Ansible to deploy the Attacker Console and disable VM capture only
  after a future independent-sensor cutover checkpoint. This UI release must
  keep the existing VM capture active.
- [x] Publish through Jenkins/Argo CD, apply Ansible, prove the two browser
  surfaces are separated, run at least one clean end-to-end attack, and record
  rollback/evidence.

Local separation proof on 2026-08-10:

- Eleven focused SOC/Attacker Console tests pass on the host. Twenty-five
  affected tests and the complete 72-test application suite pass inside the
  dependency-complete application container.
- JavaScript syntax, Ruff, Python compile, application Helm lint/template,
  Ansible syntax, and `git diff --check` pass. The rendered GKE manifest has
  only the observation token, contains the token-protected private run-control
  proxy, and no longer contains `TRAFFIC_AGENT_URL`, the traffic-agent Secret,
  or the SOC-to-agent egress policy.
- The Attacker Console API returns generator/target identity and sanitized
  delivery counters only. Both credentials remain server-side; the agent run
  remains gated until collector registration succeeds and is cancelled if
  registration fails.

Live attacker/SOC separation evidence on 2026-08-10:

- GitHub successfully delivered commit `19c8e78` after Jenkins was brought up
  under the quota-safe compute rotation. Jenkins build 101 selected
  `application=true` and `federated=false`, passed the CRITICAL scan, and
  published digest
  `sha256:348b6b96ebbd13bc8a545f4ed5470ad3d84f263ed0daeb28e700a846d02a11d7`.
  Jenkins then wrote GitOps revision `14c8ec3`; Argo CD alone reconciled it to
  `Synced/Healthy`.
- The first manual Argo sync used client-side apply and retained the removed
  traffic-agent token volume, while pruning its ExternalSecret. The new console
  Pod correctly failed closed with `FailedMount`. Re-running the same Argo
  revision with server-side apply removed the stale field. The final SOC
  Deployment has no `TRAFFIC_AGENT_URL`, volume, mount, traffic-agent
  ExternalSecret, or agent-egress NetworkPolicy. This is the documented
  recovery path for a similar schema-removal migration.
- Ansible completed `ok=17 changed=5 unreachable=0 failed=0`. The loopback-only
  Attacker Console, traffic agent, VM Zeek, and shipper are all active. Both web
  credentials remain readable only by the VM service account/group boundary;
  `/api/config` returns only schema version 1 and generator/target identity.
- The Attacker Console is forwarded at `127.0.0.1:18091`; the independent GKE
  SOC Console is forwarded at `127.0.0.1:18080`. The former exposes seven fixed
  model-class profiles and sanitized send/delivery counters. The latter has no
  traffic controls or traffic API (`404`) and retains scientific replay plus
  live monitoring.
- A clean one-event Attack run `traffic-d6e63cbcc0c2` completed one send, one
  collector receive/accept, and zero drops, duplicates, or processing failures.
  VM Zeek observed source `10.10.0.20` to victim gateway `10.10.0.5:22`, service
  `ssh`, state `SF`, 585/1,803 origin/response bytes, and 20/19 packets. SOC
  sequence 280 independently reported fusion `Attack` at confidence bucket
  `0.95-1`; trusted head remained `Benign` and `is_alert=false`, preserving the
  configured shadow-policy semantics.
- Capture is intentionally still VM-local. A gateway-only sensor would lose
  the fixed `10.20.0.20-22` blackhole profiles; completing physical sensor
  separation therefore remains gated on a separately reviewed Packet Mirroring
  collector Terraform/cost plan. No new GCP resource was created in this
  release.

## Result

Scientific evaluation, serving-bundle promotion, locked test execution, the
validation-selected alert policy, internal demo console, GKE stack, and the
original six in-cluster live scenarios are complete. The model shows no
validation-to-test overfitting in the locked rolling protocol. The all-decision
privacy-reduced sink and six-panel Kibana dashboard are deployed and verified
on GKE. The private traffic-generator extension is deployed and has completed
the four controlled runs above. Its executable attack candidates remain
scientifically non-equivalent to the IoT-23 validation reference, and the
fusion policy remains shadow-only because the live benign control is a proven
false positive. At this UI-demo checkpoint Jenkins is stopped and the generator
VM is running under the approved quota-safe rotation policy.
