# Phase 1 Clean Report-Ready Analysis

This report aggregates immutable clean bundles only. It does not train a model, alter stored metrics, or infer unavailable samples.

## 1. Run inventory

- Inputs: `artifacts/phase1_clean/seed-42-full`, `artifacts/phase1_clean/seed-1337-full`, `artifacts/phase1_clean/seed-2026-full`
- Bundles: **21**
- Seeds: **3** ([42, 1337, 2026])
- `single_seed=false`
- `statistical_uncertainty=AVAILABLE`
- Existing input figures discovered: **7**
- Existing input analysis CSV/MD discovered: **7**

The prior seven figures, when present, are one count confusion matrix for pooled plus one for each of six LOSO bundles. The report-ready set below adds normalized matrices, aggregate comparisons, support, entropy, and learning curves.

## 2. Pooled result

| seed | accuracy | weighted_f1 | fixed_macro_f1 | validation_macro_f1 | best_epoch | train_time |
|---|---|---|---|---|---|---|
| 42 | 0.985679601662483 | 0.9856821354499398 | 0.9286915860439056 | 0.8666000068651785 | 148 | NOT_AVAILABLE |
| 1337 | 0.9850211925435167 | 0.9850624983067572 | 0.9075945179252547 | 0.8662504831578797 | 148 | NOT_AVAILABLE |
| 2026 | 0.9839924282951319 | 0.9842028316883745 | 0.8636567520368803 | 0.8955894397946669 | 138 | NOT_AVAILABLE |

## 3. LOSO result

| held_out | seed_count | fixed_macro_f1_mean | fixed_macro_f1_std | fixed_macro_f1_min | fixed_macro_f1_max | fixed_macro_f1_mean_std | seen_macro_f1_mean | seen_macro_f1_std | seen_macro_f1_min | seen_macro_f1_max | seen_macro_f1_mean_std | binary_f1_mean | binary_f1_std | binary_f1_min | binary_f1_max | binary_f1_mean_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 3 | 0.19134068946143876 | 0.029343802000976693 | 0.15254674501581383 | 0.22349934180761705 | 0.191341 ± 0.029344 | 0.5102418385638368 | 0.07825013866927119 | 0.4067913200421702 | 0.5959982448203122 | 0.510242 ± 0.078250 | 0.8748265315069874 | 0.09882022681455643 | 0.7350739319900182 | 0.9449558879938627 | 0.874827 ± 0.098820 |
| 3-1 | 3 | 0.4005180081447988 | 0.030723364693954713 | 0.3698641729731912 | 0.44251221841528565 | 0.400518 ± 0.030723 | 0.8010360162895976 | 0.061446729387909425 | 0.7397283459463824 | 0.8850244368305713 | 0.801036 ± 0.061447 | 0.9926847375550039 | 0.00028190762387429914 | 0.9922923918448533 | 0.9929422006653608 | 0.992685 ± 0.000282 |
| 34-1 | 3 | 0.07001454983716153 | 0.02301173814218216 | 0.0374723121201577 | 0.08653509155759082 | 0.070015 ± 0.023012 | 0.18670546623243076 | 0.0613646350458191 | 0.09992616565375385 | 0.23076024415357552 | 0.186705 ± 0.061365 | 0.5246957774203452 | 0.07631222680820844 | 0.4167874001028182 | 0.5801232342708639 | 0.524696 ± 0.076312 |
| 36-1 | 3 | 0.09775551288329538 | 0.03841340788275885 | 0.043430752169091265 | 0.12492963032463877 | 0.097756 ± 0.038413 | 0.7820441030663631 | 0.3073072630620708 | 0.3474460173527301 | 0.9994370425971102 | 0.782044 ± 0.307307 | 0.8888083407631928 | 0.15712502020765193 | 0.6666000066660001 | 0.9999250056245782 | 0.888808 ± 0.157125 |
| 39-1 | 3 | 0.3038577835876866 | 0.004792211377148792 | 0.2999199943377324 | 0.310603528293221 | 0.303858 ± 0.004792 | 0.6077155671753732 | 0.009584422754297583 | 0.5998399886754648 | 0.621207056586442 | 0.607716 ± 0.009584 | 0.9764906943338704 | 0.005449710210902609 | 0.9724696027001749 | 0.9841952834091849 | 0.976491 ± 0.005450 |
| 9-1 | 3 | 0.24036633403515972 | 0.012007080182357856 | 0.22338663503979958 | 0.24900561003176233 | 0.240366 ± 0.012007 | 0.9614653361406389 | 0.048028320729431424 | 0.8935465401591983 | 0.9960224401270493 | 0.961465 ± 0.048028 | 0.9444745985548387 | 0.06629746233566101 | 0.8507253158633599 | 0.9925043435095557 | 0.944475 ± 0.066297 |

## 4. Hardest scenarios

- Hardest fold by clean fixed macro-F1 mean: **34-1**.
- Hardness is reported from observed held-out performance; private classes are identified separately from class support.

## 5. Class absence explanation

A class with `train_support=0` remains in the fixed eight-class output space but was not learned in that fold. `private_to_held_out=true` means it appears in held-out test support while absent from train.

## 6. Fixed vs seen vs binary evaluation

- LOSO fixed macro-F1 mean over supplied runs: 0.21730881299159016
- LOSO seen-class macro-F1 mean: 0.6415347212447068
- LOSO binary F1 mean: 0.866996780022373

Fixed-class macro-F1 is the primary closed-set score. Seen-class and binary F1 explain failure modes but do not replace it.

## 7. Entropy analysis

| protocol | seed | held_out | comparison | auroc | auprc | direction | positive_count | negative_count | status | reason |
|---|---|---|---|---|---|---|---|---|---|---|
| loso | 1337 | 1-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 16800 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 1337 | 3-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 18687 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 1337 | 34-1 | absent_from_train vs known_correct | 0.5058532816537468 | 0.9071441727666822 | higher_entropy_indicates_absent_from_train | 10000 | 1935 | AVAILABLE |  |
| loso | 1337 | 36-1 | absent_from_train vs known_correct | 0.9996244273460371 | 0.9999499975034986 | higher_entropy_indicates_absent_from_train | 20003 | 2663 | AVAILABLE |  |
| loso | 1337 | 39-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 17559 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 1337 | 9-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 19849 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| pooled | 1337 | ALL | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 23937 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 2026 | 1-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 9519 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 2026 | 3-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 20263 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 2026 | 34-1 | absent_from_train vs known_correct | 0.2881044255319149 | 0.7917393497989655 | higher_entropy_indicates_absent_from_train | 10000 | 3525 | AVAILABLE |  |
| loso | 2026 | 36-1 | absent_from_train vs known_correct | 0.9994366785650571 | 0.9995220900811518 | higher_entropy_indicates_absent_from_train | 20003 | 2663 | AVAILABLE |  |
| loso | 2026 | 39-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 17489 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 2026 | 9-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 16491 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| pooled | 2026 | ALL | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 23912 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 42 | 1-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 13836 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 42 | 3-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 19492 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 42 | 34-1 | absent_from_train vs known_correct | 0.3082325587985265 | 0.7961193681583006 | higher_entropy_indicates_absent_from_train | 10000 | 3529 | AVAILABLE |  |
| loso | 42 | 36-1 | absent_from_train vs known_correct | 0.9648812445225629 | 0.9947061299125195 | higher_entropy_indicates_absent_from_train | 20003 | 2662 | AVAILABLE |  |
| loso | 42 | 39-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 17836 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| loso | 42 | 9-1 | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 19802 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |
| pooled | 42 | ALL | absent_from_train vs known_correct | NOT_AVAILABLE | NOT_AVAILABLE | higher_entropy_indicates_absent_from_train | 0 | 23953 | NOT_AVAILABLE | Requires at least one positive and one negative sample. |

Entropy is uncertainty analysis only. No result here is described as zero-day detection; AUROC/AUPRC and direction must support any narrower claim.

## 8. Historical vs clean comparison

| protocol | historical_metric | clean_seed42_metric | clean_mean_metric | note |
|---|---|---|---|---|
| pooled | 0.8773151305403201 | 0.9286915860439056 | 0.8999809520020134 | Historical result is leakage-prone/context-only and is not included in clean aggregation. |
| loso | 0.22781310584865685 | 0.2262343836539726 | 0.21730881299159013 | Historical result is leakage-prone/context-only and is not included in clean aggregation. |

Historical leakage-prone values are context only and are never mixed into clean means.

## 9. Statistical limitations

- Multi-seed mean/std/min/max are reported with population standard deviation (`ddof=0`). Three seeds quantify seed variation but do not establish external-dataset generalization.
- `train_time` is NOT_AVAILABLE unless explicitly stored in metadata; it is never reconstructed from timestamps.

## 10. Figure recommendation

See `FIGURE_SELECTION.md`. Main-report figures are capped at six; learning curves, entropy diagnostics, count matrices, and stability details belong in the appendix unless central to the argument.

## Availability

- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-1337-full/loso-held-out-1-1-seed-1337`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-1337-full/loso-held-out-3-1-seed-1337`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-1337-full/loso-held-out-36-1-seed-1337`: known_incorrect vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-1337-full/loso-held-out-39-1-seed-1337`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-1337-full/loso-held-out-9-1-seed-1337`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-1337-full/pooled-seed-1337`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-2026-full/loso-held-out-1-1-seed-2026`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-2026-full/loso-held-out-3-1-seed-2026`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-2026-full/loso-held-out-36-1-seed-2026`: known_incorrect vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-2026-full/loso-held-out-39-1-seed-2026`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-2026-full/loso-held-out-9-1-seed-2026`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-2026-full/pooled-seed-2026`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-42-full/loso-held-out-1-1-seed-42`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-42-full/loso-held-out-3-1-seed-42`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-42-full/loso-held-out-39-1-seed-42`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-42-full/loso-held-out-9-1-seed-42`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- `/Users/nguyen_bao/Projects/AIproject/FedKube-IDS/artifacts/phase1_clean/seed-42-full/pooled-seed-42`: absent_from_train vs known_correct: Requires at least one positive and one negative sample.
- All requested figures were created.
