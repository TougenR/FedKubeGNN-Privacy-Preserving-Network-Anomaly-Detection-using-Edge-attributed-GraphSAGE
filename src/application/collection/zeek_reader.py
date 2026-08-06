"""Read Zeek JSON conn records without introducing evaluation labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TextIO


class ZeekRecordError(ValueError):
    pass


REQUIRED_FIELDS = (
    "ts",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "conn_state",
)


def parse_zeek_json(line: str) -> dict:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ZeekRecordError("Invalid Zeek JSON record.") from exc
    if not isinstance(value, dict):
        raise ZeekRecordError("Zeek record must be a JSON object.")
    forbidden = {"label", "detailed-label", "detailed_label"} & set(value)
    if forbidden:
        raise ZeekRecordError(
            f"Production Zeek record contains evaluation labels: {sorted(forbidden)}."
        )
    missing = [name for name in REQUIRED_FIELDS if name not in value]
    if missing:
        raise ZeekRecordError(f"Zeek conn record is missing fields: {missing}.")
    return value


def read_zeek_json(stream: TextIO) -> Iterator[dict]:
    for line_number, line in enumerate(stream, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield parse_zeek_json(stripped)
        except ZeekRecordError as exc:
            raise ZeekRecordError(f"Zeek line {line_number}: {exc}") from exc


def read_zeek_json_file(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        yield from read_zeek_json(handle)
