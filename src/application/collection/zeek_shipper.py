"""Tail a real Zeek JSON conn.log and deliver label-free flow observations."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from src.application.api.schema import ProductionFlow
from src.application.collection.delivery import ObservationDispatcher
from src.application.collection.zeek_reader import parse_zeek_json


logger = logging.getLogger(__name__)


def production_flow_from_zeek(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize Zeek JSON without inventing missing numeric measurements."""
    document = {
        "ts": record["ts"],
        "uid": record.get("uid", "-"),
        "id.orig_h": record["id.orig_h"],
        "id.orig_p": record["id.orig_p"],
        "id.resp_h": record["id.resp_h"],
        "id.resp_p": record["id.resp_p"],
        "proto": record["proto"],
        "service": record.get("service") or "-",
        "duration": record.get("duration"),
        "orig_bytes": record.get("orig_bytes"),
        "resp_bytes": record.get("resp_bytes"),
        "conn_state": record["conn_state"],
        "local_orig": "-",
        "local_resp": "-",
        "missed_bytes": record.get("missed_bytes"),
        "history": record.get("history") or "-",
        "orig_pkts": record.get("orig_pkts"),
        "orig_ip_bytes": record.get("orig_ip_bytes"),
        "resp_pkts": record.get("resp_pkts"),
        "resp_ip_bytes": record.get("resp_ip_bytes"),
        "tunnel_parents": "-",
    }
    return ProductionFlow.model_validate(document).model_dump(by_alias=True)


async def ship(
    *,
    path: Path,
    collector_url: str,
    sensor_id: str,
    poll_seconds: float,
    queue_size: int,
) -> None:
    dispatcher = ObservationDispatcher(
        endpoint=collector_url,
        queue_size=queue_size,
        workers=1,
        retry_attempts=5,
        retry_backoff_seconds=0.5,
    )
    await dispatcher.start()
    try:
        while not path.exists():
            await asyncio.sleep(poll_seconds)
        with path.open("r", encoding="utf-8") as handle:
            while True:
                line = handle.readline()
                if not line:
                    await asyncio.sleep(poll_seconds)
                    continue
                try:
                    record = parse_zeek_json(line)
                    flow = production_flow_from_zeek(record)
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Rejected Zeek conn record: %s", exc)
                    continue
                accepted = dispatcher.enqueue(
                    {
                        "sensor_id": sensor_id,
                        "source": "zeek-json-v1",
                        "run_id": None,
                        "scenario_id": None,
                        "flow": flow,
                    },
                    run_id=None,
                )
                if not accepted:
                    logger.error("Zeek delivery queue is full; conn record dropped.")
    finally:
        await dispatcher.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--collector-url", required=True)
    parser.add_argument("--sensor-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--queue-size", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        ship(
            path=args.path,
            collector_url=args.collector_url,
            sensor_id=args.sensor_id,
            poll_seconds=args.poll_seconds,
            queue_size=args.queue_size,
        )
    )


if __name__ == "__main__":
    main()
