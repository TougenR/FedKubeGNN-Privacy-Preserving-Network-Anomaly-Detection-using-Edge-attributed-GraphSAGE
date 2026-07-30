# Phase 1 Clean Protocol

Date: 2026-07-30

## Purpose and entry point

`src.phase1_clean` is the leakage-remediated E-GraphSAGE experiment path.
`src.run_experiments` remains the historical reproduction path and must not be
used for clean claims. The clean runner does not change the model architecture,
the feature implementation, or the closed-set classification task.

The policy source is `config.yaml:phase1_clean`:

- output root: `artifacts/phase1_clean`;
- final imbalance mode: `class_weight`;
- seeds: `42`, `1337`, and `2026`;
- fixed external eight-class taxonomy.

## Pooled and per-scenario protocol

```mermaid
flowchart LR
  A["Raw rows + stable row IDs"] --> B["Deterministic train / validation / test membership"]
  B --> C["Fit preprocessor on train raw rows only"]
  C --> D["Transform all rows with frozen preprocessor"]
  D --> E["Build full graph and reattach membership masks"]
  E --> F["Class weights from train labels only"]
  F --> G["Gradient on train_mask"]
  G --> H["Early stopping on val_mask"]
  H --> I["One final evaluation on test_mask"]
```

The graph is deliberately transductive: validation/test edge features and
topology may participate in full-graph message passing. Their labels do not fit
the preprocessor, determine class weights, enter gradient loss, or select the
imbalance mode. This limitation must accompany any pooled result.

Stable row IDs are SHA-256 values over protocol version, scenario name, and raw
row position. The runner resolves masks from these IDs after transformation
rather than assuming that intermediate DataFrame order is unchanged.

`per_scenario` uses the same boundary independently for each scenario.

## LOSO protocol

```mermaid
flowchart LR
  A["Choose held-out scenario"] --> B["Split each non-held-out scenario into train / validation"]
  B --> C["Fit shared preprocessor on union of train rows"]
  C --> D["Transform/build non-held-out training-domain graphs"]
  D --> E["Gradient on train masks"]
  E --> F["Select best epoch on validation masks"]
  F --> G["Only now transform/build held-out graph"]
  G --> H["One final held-out evaluation"]
```

The held-out scenario is excluded from preprocessing fit, class-weight
calculation, undersampling, training, and model selection. Validation labels
are excluded from gradient loss. The non-held-out graphs remain transductive
within the training domain, so validation edge features and topology can be
visible during message passing; this is an accepted protocol limitation.

## Taxonomy and imbalance policy

The fixed mapping is declared before any split in `config.yaml:phase1_clean`
and enforced against the `FIXED_LABELS` constant in `src/phase1_clean.py`:

| Index | Label |
|---:|---|
| 0 | Attack |
| 1 | Benign |
| 2 | C&C |
| 3 | C&C-HeartBeat |
| 4 | DDoS |
| 5 | Okiru |
| 6 | Okiru-Attack |
| 7 | PartOfAHorizontalPortScan |

A class with zero training support retains its output index, receives class
weight `0`, and is recorded with support `0`. This preserves the deployment
contract; it does not claim that the model learned that class.

`class_weight` is selected in config before the run. The clean runner does not
inspect pooled test or LOSO held-out scores to select a mode. The optional
undersampling helper is retained only for regression coverage and may remove
training rows only.

## Preprocessing and final-test isolation

Scaler statistics, categorical vocabularies, rare-service grouping, feature
columns, and class weights are learned only from training membership.
Validation is used once per epoch for early stopping. The best validation
checkpoint is restored before exactly one final test/held-out evaluation.

Metrics from the historical `none`, `class_weight`, and `undersample` Phase A
runs remain historical comparisons. They are not selection evidence for the
clean rerun.

## Run-scoped artifact contract

Every new run directory contains:

```text
model.pt
preprocessor.pkl
schema.json
labels.json
metadata.json
metrics.json
split_manifest.json
predictions.csv
```

The checkpoint embeds model config, feature dimension, exact ordered feature
columns, fixed label mapping, seed, protocol, bundle version, and schema
digests. `schema.json` and `labels.json` carry their own canonical digests.
`split_manifest.json` records per-scenario split counts and stable-ID digests
plus disjointness and coverage checks. `metadata.json` records protocol,
support/counts, best epoch, validation/final metrics, git state, timestamp, and
known limitations.

`predictions.csv` contains one row per final test/held-out example: stable row
ID, scenario, split, protocol, seed, true/predicted label and index, confidence,
entropy, true-class training support, the zero-support flag, and logits plus
probabilities in fixed taxonomy order. These rows are captured during the
existing single final forward pass; prediction export does not add another
evaluation or alter training behavior.

`validate_clean_bundle()` checks these files and then calls the existing Phase
3 fail-closed loader. Feature-order, feature-count, or label-mapping drift is
rejected. Clean bundles must not be written under
`artifacts/phase1_results/`.

## Commands

Verify all six required raw scenarios and write
`dataset_manifest.json`/`dataset_manifest.csv`:

```bash
python scripts/verify_phase1_dataset.py \
  --config config.yaml \
  --out-dir artifacts/phase1_clean
```

The verifier accepts the same explicit scenario format as the clean runner:

```bash
python scripts/verify_phase1_dataset.py \
  --scenarios \
    1-1=/path/to/1-1/conn.log.labeled \
    3-1=/path/to/3-1/conn.log.labeled \
    9-1=/path/to/9-1/conn.log.labeled \
    34-1=/path/to/34-1/conn.log.labeled \
    36-1=/path/to/36-1/conn.log.labeled \
    39-1=/path/to/39-1/conn.log.labeled
```

Toy pooled + one-fold LOSO smoke, one epoch each, temporary bundles only:

```bash
python -m src.phase1_clean --config config.yaml --toy-smoke
```

One-seed local dry run with the two scenario files currently present:

```bash
python -m src.phase1_clean \
  --config config.yaml \
  --protocols pooled loso \
  --scenarios \
    34-1=data/CTU-IoT-Malware-Capture-34-1/conn.log.labeled \
    3-1=data/CTU-IoT-Malware-Capture-3-1/conn.log.labeled \
  --seed 42 \
  --epochs 1 \
  --cap-per-class 100 \
  --out-dir artifacts/phase1_clean/dry-run-seed-42
```

Three-seed clean rerun on Vast.ai, after all six configured raw files exist:

```bash
for seed in 42 1337 2026; do
  python -m src.phase1_clean \
    --config config.yaml \
    --protocols pooled loso \
    --seed "$seed" \
    --out-dir "artifacts/phase1_clean/seed-$seed"
done
```

Analyze all completed seed roots:

```bash
python scripts/analyze_phase1_clean.py \
  artifacts/phase1_clean/seed-* \
  --output-dir artifacts/phase1_clean/analysis
```

The analyzer writes `summary.csv`, pooled/LOSO summaries, class support,
binary metrics, `report.md`, confusion figures when predictions are present,
and `entropy_summary.csv` only when a complete fixed-order probability or
logit artifact exists. Missing artifacts are reported as `NOT_AVAILABLE`; no
metric is reconstructed by assumption.

Each output root must be new or contain no colliding run directories. Do not
point `--out-dir` at `artifacts/phase1_results`.
