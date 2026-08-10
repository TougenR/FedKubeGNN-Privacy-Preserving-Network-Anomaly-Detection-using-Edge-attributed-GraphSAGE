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
                "ssh-emulator": {"endpoints": ["10.10.0.5"]},
                "irc-emulator": {
                    "endpoints": ["10.20.0.20", "10.10.0.5"]
                },
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

    async def test_ddos_uses_adjusted_bounded_ack_only_schedule(self) -> None:
        calls: list[dict] = []

        def packet_sender(**kwargs) -> bool:
            calls.append(kwargs)
            return True

        executor = TrafficExecutor(
            catalog=CATALOG,
            targets=targets(),
            packet_sender=packet_sender,
            interval_scale=0.0001,
        )
        record = await executor.start("ddos", events=10, interval_ms=20)
        executor.release(record.run_id)
        while executor.current().status in {"waiting-for-release", "running"}:
            await asyncio.sleep(0.01)
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(call["flags"] == 0x10 for call in calls))
        self.assertTrue(all(call["corrupt_checksum"] for call in calls))
        self.assertEqual(executor.current().events, 10)
        self.assertEqual(executor.current().interval_ms, 20)

    async def test_ssh_and_irc_use_fixed_session_and_blackhole_mix(self) -> None:
        packets: list[dict] = []
        sessions: list[dict] = []

        executor = TrafficExecutor(
            catalog=CATALOG,
            targets=targets(),
            packet_sender=lambda **kwargs: packets.append(kwargs) is None,
            session_sender=lambda **kwargs: sessions.append(kwargs) is None,
            interval_scale=0.0001,
        )
        ssh = await executor.start("attack-ssh", events=1, interval_ms=1000)
        executor.release(ssh.run_id)
        while executor.current().status in {"waiting-for-release", "running"}:
            await asyncio.sleep(0.01)
        self.assertEqual(sessions[0]["protocol"], "ssh")
        self.assertEqual(sessions[0]["destination"], "10.10.0.5")

        irc = await executor.start("command-control", events=3, interval_ms=500)
        executor.release(irc.run_id)
        while executor.current().status in {"waiting-for-release", "running"}:
            await asyncio.sleep(0.01)
        self.assertEqual(packets[-1]["destination"], "10.20.0.20")
        self.assertEqual(packets[-1]["flags"], 0x02)
        self.assertEqual([item["complete"] for item in sessions[-2:]], [False, True])

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
