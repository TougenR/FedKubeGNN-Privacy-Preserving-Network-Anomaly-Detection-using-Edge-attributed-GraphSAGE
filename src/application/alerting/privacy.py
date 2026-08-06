"""Fail closed if an Elasticsearch document contains forbidden raw fields."""

from __future__ import annotations

from typing import Any, Mapping


FORBIDDEN_FIELD_NAMES = {
    "id.orig_h",
    "id.resp_h",
    "source_ip",
    "destination_ip",
    "raw_ip",
    "features",
    "edge_attr",
    "tensor",
    "ground_truth",
    "detailed-label",
    "detailed_label",
    "probabilities",
}


def validate_elasticsearch_document(document: Mapping[str, Any]) -> None:
    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                name = str(key)
                if name in FORBIDDEN_FIELD_NAMES:
                    raise ValueError(
                        f"Elasticsearch document contains forbidden field '{name}'."
                    )
                walk(child, f"{prefix}.{name}" if prefix else name)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child, prefix)

    walk(document)
