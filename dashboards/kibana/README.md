# Kibana assets

`elasticsearch-index-template.json` is the strict storage boundary for Phase 4
detection events. Install it before creating the data view
`fedper-detections*`. The strict mapping deliberately rejects any field outside
the approved privacy-reduced event contract.

The dashboard must be assembled only after local events exist, then exported
here as a Kibana saved-object NDJSON file. Required panels are detection count
by class, alert rate by client, confidence/entropy buckets, model/head version,
latency p50/p95, benign false-positive rate, and scenario timeline. Head
disagreement and ground truth belong to scientific evaluation indices, not the
production detection-event index.
