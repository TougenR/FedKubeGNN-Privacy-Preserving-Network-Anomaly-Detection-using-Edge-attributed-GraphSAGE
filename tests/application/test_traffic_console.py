from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from src.application.collection.transport import ServiceRequestError
from src.application.traffic_console.app import (
    StartTrafficRequest,
    config,
    current_run,
    start_run,
    state,
    stop_run,
)


ROOT = Path(__file__).resolve().parents[2]


class TrafficConsoleTests(unittest.TestCase):
    def setUp(self) -> None:
        state.clear()
        state.update(
            agent_url="http://10.10.0.20:8091",
            agent_token="a" * 32,
            control_url="http://10.10.0.5:8082",
            observation_token="o" * 32,
            sensor_id="sensor-34-1",
            identity={
                "generator_name": "fedkube-traffic-generator",
                "generator_source_ipv4": "10.10.0.20",
                "generator_zone": "asia-southeast1-b",
                "target_name": "fedkube-detection-gateway",
                "target_ipv4": "10.10.0.5",
                "sensor_id": "sensor-34-1",
            },
        )

    def tearDown(self) -> None:
        state.clear()

    def test_attacker_surface_has_no_model_or_alert_contract(self) -> None:
        document = asyncio.run(config())
        self.assertEqual(document["console_schema_version"], 1)
        self.assertEqual(document["access_boundary"], "attacker-only")
        serialized = str(document).lower()
        self.assertNotIn("token", serialized)
        self.assertNotIn("prediction", serialized)
        self.assertNotIn("alert", serialized)
        page = (ROOT / "src/application/traffic_console/static/index.html").read_text()
        self.assertIn("Attacker Console", page)
        self.assertIn("MÁY PHÁT TRAFFIC", page)
        self.assertNotIn('id="predicted"', page)
        self.assertNotIn('id="alert', page)

    def test_start_registers_before_release_with_server_side_tokens(self) -> None:
        calls: list[tuple[str, dict[str, str] | None]] = []

        def fake_post(url, _body, *, headers=None):
            calls.append((url, headers))
            if url.endswith("/v1/runs"):
                return {"run_id": "traffic-a1", "status": "waiting-for-release"}
            if url.endswith("/runs/register"):
                return {"status": "registered"}
            return {"run_id": "traffic-a1", "status": "running"}

        with patch("src.application.traffic_console.app.post_json", side_effect=fake_post):
            result = asyncio.run(start_run("attack", StartTrafficRequest(events=2, interval_ms=100)))
        self.assertEqual(result["status"], "running")
        self.assertEqual([url for url, _ in calls], [
            "http://10.10.0.20:8091/v1/runs",
            "http://10.10.0.5:8082/runs/register",
            "http://10.10.0.20:8091/v1/runs/traffic-a1/release",
        ])
        self.assertEqual(calls[0][1], {"Authorization": f"Bearer {'a' * 32}"})
        self.assertEqual(calls[1][1], {"X-FedKube-Observation-Token": "o" * 32})

    def test_failed_registration_cancels_gated_agent_run(self) -> None:
        with (
            patch(
                "src.application.traffic_console.app.post_json",
                side_effect=[
                    {"run_id": "traffic-deadbeef"},
                    ServiceRequestError("registration failed"),
                ],
            ),
            patch(
                "src.application.traffic_console.app.delete_json",
                return_value={"run": {"status": "cancelled"}},
            ) as cancel,
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(start_run("okiru", StartTrafficRequest(events=2, interval_ms=100)))
        self.assertEqual(raised.exception.status_code, 502)
        cancel.assert_called_once_with(
            "http://10.10.0.20:8091/v1/runs/current",
            headers={"Authorization": f"Bearer {'a' * 32}"},
        )

    def test_current_adds_only_pipeline_counters(self) -> None:
        with patch(
            "src.application.traffic_console.app.get_json",
            side_effect=[
                {"run": {"run_id": "traffic-a1", "profile_id": "attack"}},
                {"run_id": "traffic-a1", "accepted": 1, "predicted": 1},
            ],
        ) as get:
            result = asyncio.run(current_run())
        self.assertEqual(result["run"]["pipeline"]["collector"]["accepted"], 1)
        self.assertNotIn("predicted", result["run"]["pipeline"]["collector"])
        self.assertEqual(
            get.call_args_list[1].kwargs["headers"],
            {"X-FedKube-Observation-Token": "o" * 32},
        )

    def test_stop_uses_only_authenticated_agent_current_run(self) -> None:
        with patch(
            "src.application.traffic_console.app.delete_json",
            return_value={"run": {"status": "cancelled"}},
        ) as delete:
            response = asyncio.run(stop_run())
        self.assertEqual(response["run"]["status"], "cancelled")
        delete.assert_called_once_with(
            "http://10.10.0.20:8091/v1/runs/current",
            headers={"Authorization": f"Bearer {'a' * 32}"},
        )


if __name__ == "__main__":
    unittest.main()
