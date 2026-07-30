# Execution Plan: Phase 1 Clean Canonical Data Cache

Date: 2026-07-30

## Status

Completed

## Outcome

Parse and deterministically clean each raw Phase 1 scenario at most once per
raw-data/parser/schema fingerprint, then reuse a canonical Parquet cache across
pooled, every LOSO fold, and later seed processes without moving any learned
preprocessing or split state into the cache.

## Context

- `src/phase1_clean.py`
- `src/multi_scenario.py`
- `src/data_io.py`
- `src/preprocess.py`
- `docs/PHASE1_CLEAN_PROTOCOL.md`

## Scope

In scope:

- Add fingerprinted scenario-level cleaned Parquet caches.
- Preserve existing pandas parsing and `clean_flows` semantics.
- Add cache CLI controls and progress/load reports.
- Prove cache-on/off equivalence at frame, split, preprocessor, and class-weight
  boundaries.
- Benchmark toy and one available real scenario without training a full model.

Out of scope:

- Model, taxonomy, metric, split, fit boundary, bundle contract, or historical
  runner changes.
- Polars parser rewrite.

## Approach

1. Factor canonical streaming parse/clean into a cache module.
2. Fingerprint raw stat, parser/clean version, and canonical schema contract.
3. Build Parquet atomically via temporary chunk files so raw input is parsed
   once and schema drift across chunks preserves pandas concatenation semantics.
4. Reapply the existing deterministic per-class cap while reading raw/cache
   row groups.
5. Opt only `src.phase1_clean` into cache mode; historical callers retain their
   existing path.
6. Add equivalence/invalidation tests and observed benchmarks.

## Risks And Recovery

- Parquet support requires `pyarrow`; fail with an actionable error when absent.
- Cache content is pre-split and contains no fitted state.
- Existing cache files are immutable by fingerprint; rebuild uses atomic
  replacement of only the exact target cache.
- Recovery is removal of the additive cache module/CLI wiring and cache files;
  raw data and experiment artifacts remain untouched.

## Progress

- [x] Trace current raw-file reads and dependency availability.
- [x] Implement canonical cache, inter-process lock, and progress reporting.
- [x] Wire clean CLI and verifier cache controls.
- [x] Add regression tests.
- [x] Complete repository validation and archive this plan.

## Decisions

- 2026-07-30: Use raw size + `mtime_ns` rather than SHA256 in the hot-path
  fingerprint. This avoids an additional 12.9 GB read while still satisfying
  the locked fingerprint policy.
- 2026-07-30: Retain pandas parsing/cleaning; use `pyarrow` only as the Parquet
  storage engine.
- 2026-07-30: Cache every cleaned row. Apply `cap_per_class` after canonical
  cleaning so pilot and full caps reuse the same data version.
- 2026-07-30: The verifier warms the same cache used by pilot/full runs. Its
  fingerprint field uses the canonical stat/code/schema digest instead of an
  extra full-file SHA pass.
- 2026-07-30: Use a per-fingerprint `fcntl` lock so concurrent seed processes
  cannot duplicate the raw parse.

## Validation

- Focused cache regression suite.
- Existing Phase 1 and Phase 3 suites.
- Toy cache MISS/HIT benchmark.
- Real `34-1` cache MISS/HIT benchmark if locally available.
- Compile and `git diff --check`.

## Result

Implementation and focused regression proof complete. Local observed benchmark:

- toy 720 rows: MISS 0.264 s / one raw open; HIT 0.021 s / zero raw opens;
- real IoT-23 34-1, 23,145 rows: MISS ~0.34 s / one raw open; HIT ~0.01 s /
  zero raw opens.

Validation completed:

- 23 Phase 1 unit tests passed;
- 8 Phase 3 contract/API tests passed;
- 7 focused cache/verifier tests passed, including exact compatibility with
  the existing capped raw parser;
- legacy `scripts/test_multi_scenario.py` passed;
- CLI help, Python compilation, and `git diff --check` passed.
