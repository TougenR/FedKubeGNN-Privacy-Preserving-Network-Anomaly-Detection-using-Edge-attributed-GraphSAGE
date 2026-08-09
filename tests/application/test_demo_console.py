from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from src.application.demo_console.app import current_run, state as console_state

from src.application.scenario_runner.catalog import EXPECTED_SCENARIOS, load_catalog
from src.application.scenario_runner.executor import (
    ScenarioConflictError,
    ScenarioExecutor,
)


ROOT = Path(__file__).resolve().parents[2]


class DemoConsoleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog(ROOT / "configs/application/demo-scenarios.yaml")

    def test_catalog_has_exact_approved_scenarios_and_seven_classes(self) -> None:
        self.assertEqual({item.id for item in self.catalog.scenarios}, EXPECTED_SCENARIOS)
        self.assertEqual(len(self.catalog.model_classes), 7)
        self.assertEqual(self.catalog.sensor_id, "sensor-34-1")
        replay = (ROOT / "configs/application/scientific-replay.json").read_text()
        self.assertIn('"selection_split": "validation"', replay)
        page = (ROOT / "src/application/demo_console/static/index.html").read_text()
        self.assertIn("Phát lại khoa học", page)
        self.assertIn('id="alert-banner"', page)
        self.assertIn('id="detection-chart"', page)
        self.assertIn('id="predicted"', page)
        self.assertIn('id="head-grid"', page)

        script = (ROOT / "src/application/demo_console/static/app.js").read_text()
        self.assertIn('event.predicted_class === "Benign"', script)
        self.assertIn("state.chartEvents.slice(-80)", script)
        self.assertIn("event.is_alert", script)
        self.assertIn('event.alert_decision_source === "trusted-shadow"', script)
        self.assertIn("Fusion 6 head phát hiện", script)
        self.assertIn("renderHeadDiagnostics", script)
        self.assertNotIn("scenario_id === \"DDoS\"", script)

    def test_parameters_are_server_side_bounded(self) -> None:
        flood = self.catalog.scenario("request-flood")
        with self.assertRaisesRegex(ValueError, "between 5 and 50"):
            flood.validate_parameters(
                {"requests_per_second": 5000, "duration_seconds": 10}
            )
        with self.assertRaisesRegex(ValueError, "parameters must be"):
            flood.validate_parameters(
                {
                    "requests_per_second": 10,
                    "duration_seconds": 10,
                    "target": "http://example.com",
                }
            )

    def test_executor_accepts_only_internal_http_targets_and_six_scan_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal HTTP URL"):
            ScenarioExecutor(
                catalog=self.catalog,
                target_url="https://example.com",
                scan_urls=["http://target:8080"] * 6,
            )
        with self.assertRaisesRegex(ValueError, "exactly six"):
            ScenarioExecutor(
                catalog=self.catalog,
                target_url="http://target",
                scan_urls=["http://target:8080"],
            )
        with self.assertRaisesRegex(ValueError, "internal HTTP URL"):
            ScenarioExecutor(
                catalog=self.catalog,
                target_url="http://example.com",
                scan_urls=["http://target:8080"] * 6,
            )

    def test_only_one_scenario_can_run_and_port_probe_is_allowlisted(self) -> None:
        async def exercise() -> None:
            executor = ScenarioExecutor(
                catalog=self.catalog,
                target_url="http://target",
                scan_urls=[f"http://target:{port}" for port in range(8080, 8086)],
            )
            gate = asyncio.Event()

            async def held_request(*_args, **_kwargs):
                await gate.wait()

            with patch.object(executor, "_request", side_effect=held_request):
                record = await executor.start("port-probe", {"port_count": 2})
                self.assertEqual(record.sensor_id, "sensor-34-1")
                with self.assertRaises(ScenarioConflictError):
                    await executor.start(
                        "benign-browsing",
                        {"request_count": 5, "interval_ms": 200},
                    )
                gate.set()
                await executor.stop()

        asyncio.run(exercise())

    def test_gated_run_does_not_emit_traffic_before_registration(self) -> None:
        async def exercise() -> None:
            executor = ScenarioExecutor(
                catalog=self.catalog,
                target_url="http://target",
                scan_urls=[f"http://target:{port}" for port in range(8080, 8086)],
            )
            called = asyncio.Event()

            async def tracked_request(*_args, **_kwargs):
                called.set()

            with patch.object(executor, "_request", side_effect=tracked_request):
                await executor.start(
                    "benign-browsing",
                    {"request_count": 5, "interval_ms": 200},
                    gated=True,
                )
                await asyncio.sleep(0)
                self.assertFalse(called.is_set())
                executor.release()
                await asyncio.wait_for(called.wait(), timeout=1)
                await executor.stop()

        asyncio.run(exercise())

    def test_zeek_run_status_never_polls_observed_target(self) -> None:
        async def exercise() -> None:
            executor = ScenarioExecutor(
                catalog=self.catalog,
                target_url="http://target",
                scan_urls=[f"http://target:{port}" for port in range(8080, 8086)],
            )
            await executor.start(
                "benign-browsing",
                {"request_count": 5, "interval_ms": 200},
                gated=True,
            )
            console_state.update(
                {
                    "catalog": self.catalog,
                    "executor": executor,
                    "observation_mode": "zeek",
                    "target_status_url": "http://observed-target",
                    "collector_url": "http://collector",
                }
            )

            def fake_get(url):
                self.assertNotIn("observed-target", url)
                return {"available": True, "accepted": 0}

            with patch(
                "src.application.demo_console.app.get_json", side_effect=fake_get
            ):
                response = await current_run()
            self.assertFalse(response["run"]["pipeline"]["delivery"]["available"])
            await executor.stop()
            console_state.clear()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
