from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

from src.application.traffic_agent.catalog import (
    TrafficTargetCatalog,
    load_profile_catalog,
)
from src.application.traffic_agent.executor import (
    TrafficExecutor,
    TrafficProfileDisabledError,
    TrafficRunConflictError,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = load_profile_catalog(
    ROOT / "configs" / "application" / "scientific-traffic-profiles.yaml"
)


def targets() -> TrafficTargetCatalog:
    return TrafficTargetCatalog.model_validate(
        {
            "schema_version": 1,
            "source_ipv4": "10.10.0.20",
            "groups": {
                "gateway-http": {"endpoints": ["http://10.10.0.5/target/"]},
                "ssh-emulator": {"endpoints": ["10.20.0.30"]},
                "irc-emulator": {"endpoints": ["10.20.0.31"]},
                "single-blackhole": {"endpoints": ["10.20.0.20"]},
                "multi-blackhole": {
                    "endpoints": ["10.20.0.20", "10.20.0.21", "10.20.0.22"]
                },
            },
        }
    )


class TrafficAgentExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_is_gated_then_uses_only_catalogued_targets(self) -> None:
        calls: list[dict] = []

        def packet_sender(**kwargs) -> bool:
            calls.append(kwargs)
            return True

        executor = TrafficExecutor(
            catalog=CATALOG,
            targets=targets(),
            packet_sender=packet_sender,
            release_timeout_seconds=1,
            interval_scale=0.0001,
        )
        record = await executor.start("horizontal-port-scan")
        self.assertEqual(record.status, "waiting-for-release")
        await asyncio.sleep(0.01)
        self.assertEqual(calls, [])
        executor.release(record.run_id)
        while executor.current().status in {"waiting-for-release", "running"}:
            await asyncio.sleep(0.01)
        self.assertEqual(executor.current().status, "completed")
        self.assertEqual(executor.current().succeeded, 3)
        self.assertEqual(
            [call["destination"] for call in calls],
            ["10.20.0.20", "10.20.0.21", "10.20.0.22"],
        )
        self.assertTrue(all(call["destination_port"] == 22 for call in calls))
        self.assertTrue(all(call["flags"] == 0x02 for call in calls))

    async def test_disabled_ddos_fails_before_sending(self) -> None:
        executor = TrafficExecutor(catalog=CATALOG, targets=targets())
        with self.assertRaises(TrafficProfileDisabledError):
            await executor.start("ddos")

    async def test_only_one_run_can_wait_or_execute(self) -> None:
        executor = TrafficExecutor(
            catalog=CATALOG,
            targets=targets(),
            release_timeout_seconds=1,
        )
        await executor.start("command-control-heartbeat")
        with self.assertRaises(TrafficRunConflictError):
            await executor.start("okiru")
        await executor.stop()

    async def test_unreleased_run_fails_closed(self) -> None:
        executor = TrafficExecutor(
            catalog=CATALOG,
            targets=targets(),
            release_timeout_seconds=0.01,
        )
        await executor.start("command-control-heartbeat")
        await asyncio.sleep(0.03)
        self.assertEqual(executor.current().status, "failed")
        self.assertEqual(executor.current().error, "release-timeout")


if __name__ == "__main__":
    unittest.main()
