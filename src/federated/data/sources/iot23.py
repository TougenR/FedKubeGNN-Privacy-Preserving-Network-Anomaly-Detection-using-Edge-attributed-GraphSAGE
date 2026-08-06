"""Deterministic bounded-memory IoT-23 source reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _field_names(path: Path) -> list[str]:
    canonical = {
        "det_label": "detailed-label",
        "detailed_label": "detailed-label",
        "label_val": "label",
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                # Preserve a merged final header exactly as Phase 1 does;
                # split_label_column handles the corresponding merged values.
                return [canonical.get(name, name) for name in fields]
    raise ValueError(f"No #fields header found in {path}.")


def read_clean_priority_sample(
    path: str | Path,
    *,
    cap_per_class: int | None,
    chunk_size: int,
    seed: int,
) -> Any:
    """Read all chunks while retaining a uniform deterministic sample per class.

    Every row receives one seeded random priority. Keeping the globally smallest
    priorities avoids the first-chunks bias of stopping once a class cap is full.
    Peak retained rows stay bounded by roughly ``classes * 2 * cap``.
    """
    import pandas as pd
    from src.data_io import split_label_column
    from src.core.preprocess import clean_flows

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"IoT-23 source not found: {source}")
    fields = _field_names(source)
    reader = pd.read_csv(
        source,
        sep="\t",
        comment="#",
        header=None,
        names=fields,
        na_values=[],
        keep_default_na=False,
        skip_blank_lines=True,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        chunksize=chunk_size,
    )
    rng = np.random.default_rng(seed)
    retained: dict[str, Any] = {}
    for raw_chunk in reader:
        clean = clean_flows(split_label_column(raw_chunk))
        clean["__sample_priority"] = rng.random(len(clean))
        for raw_label, group in clean.groupby("detailed-label", sort=False):
            label = str(raw_label)
            combined = (
                group
                if label not in retained
                else pd.concat([retained[label], group], ignore_index=True)
            )
            if cap_per_class is not None and len(combined) > cap_per_class:
                combined = combined.nsmallest(
                    cap_per_class, "__sample_priority", keep="first"
                )
            retained[label] = combined
    if not retained:
        raise ValueError(f"No usable IoT-23 rows found in {source}.")
    sampled = pd.concat([retained[key] for key in sorted(retained)], ignore_index=True)
    return sampled.drop(columns=["__sample_priority"])
