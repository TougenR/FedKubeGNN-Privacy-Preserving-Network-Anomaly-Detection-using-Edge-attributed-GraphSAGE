from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from src.application.collection.transport import ServiceRequestError
from src.application.traffic_console.app import (
    _pipeline_snapshot,
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
        self.assertIn("ATTACKER VM -", page)
        self.assertIn("DEFENDER SYSTEM -", page)
        self.assertIn('class="pipeline-zone attacker-zone"', page)
        self.assertIn('class="pipeline-zone defender-zone"', page)
        self.assertIn('class="network-bridge"', page)
        self.assertIn("observation JSON", page)
        self.assertIn("private boundary", page)
        self.assertIn("kali㉿fedkube-attacker", page)
        self.assertIn('id="terminal-command"', page)
        self.assertIn('id="terminal-output"', page)
        self.assertIn("TRAFFIC AGENT", page)
        self.assertIn("ZEEK CONN.LOG", page)
        self.assertIn("10.20.0.20–22", page)
        self.assertNotIn("FIXED LAB CONFIGURATION", page)
        self.assertNotIn('id="predicted"', page)
        self.assertNotIn('id="alert', page)
        script = (ROOT / "src/application/traffic_console/static/app.js").read_text()
        self.assertIn("http://127.0.0.1:8090/api/runs/${state.profile.id}", script)
        self.assertIn("execution_evidence", script)
        self.assertIn("setTimeout(poll, active() ? 500 : 2000)", script)
        self.assertIn("const PLAYBACK_STAGE_MS = 300", script)
        self.assertIn("current: null", script)
        self.assertIn("pending: null", script)
        self.assertIn("playback.pending = {target}", script)
        self.assertIn("Math.min(playback.displayed[key]", script)
        self.assertIn('if (backendStatus === "error") visualStatus = "error"', script)
        self.assertIn("reducedMotion.matches", script)
        self.assertIn("+${delta} EVT", script)
        self.assertNotIn("hping3 --", script)
        self.assertNotIn("nmap ", script)
        self.assertNotIn("Authorization", script)
        self.assertNotIn("X-FedKube-Observation-Token", script)
        styles = (
            ROOT / "src/application/traffic_console/static/styles.css"
        ).read_text()
        self.assertIn("@media(prefers-reduced-motion:reduce)", styles)
        self.assertIn("animation:playback-border", styles)
        self.assertIn("article.playing:not(.error)", styles)
        self.assertNotIn("animation:packet", styles)

    def test_start_registers_before_release_with_server_side_tokens(self) -> None:
        calls: list[tuple[str, dict[str, str] | None]] = []

        def fake_post(url, _body, *, headers=None):
            calls.append((url, headers))
            if url.endswith("/v1/runs"):
                return {"run_id": "traffic-a1", "status": "waiting-for-release"}
            if url.endswith("/runs/register"):
                return {"status": "registered"}
            return {"run_id": "traffic-a1", "status": "running"}

        with patch(
            "src.application.traffic_console.app.post_json", side_effect=fake_post
        ):
            result = asyncio.run(
                start_run("attack", StartTrafficRequest(events=2, interval_ms=100))
            )
        self.assertEqual(result["status"], "running")
        self.assertEqual(
            [url for url, _ in calls],
            [
                "http://10.10.0.20:8091/v1/runs",
                "http://10.10.0.5:8082/runs/register",
                "http://10.10.0.20:8091/v1/runs/traffic-a1/release",
            ],
        )
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
                asyncio.run(
                    start_run("okiru", StartTrafficRequest(events=2, interval_ms=100))
                )
        self.assertEqual(raised.exception.status_code, 502)
        cancel.assert_called_once_with(
            "http://10.10.0.20:8091/v1/runs/current",
            headers={"Authorization": f"Bearer {'a' * 32}"},
        )

    def test_current_adds_sanitized_pipeline_evidence_only(self) -> None:
        with patch(
            "src.application.traffic_console.app.get_json",
            side_effect=[
                {"run": {"run_id": "traffic-a1", "profile_id": "attack"}},
                {
                    "run_id": "traffic-a1",
                    "gateway_received": 1,
                    "received": 1,
                    "accepted": 1,
                    "windowed": 1,
                    "predicted": 1,
                    "routed": 1,
                    "zeek_evidence": [
                        {
                            "source": "attacker-vm",
                            "target": "fixed-private-lab-target",
                            "port": 22,
                        }
                    ],
                },
            ],
        ) as get:
            result = asyncio.run(current_run())
        pipeline = result["run"]["pipeline"]
        self.assertEqual(pipeline["counters"]["inferred"], 1)
        self.assertEqual(pipeline["counters"]["stored"], 1)
        self.assertEqual(pipeline["stages"][-1]["status"], "acknowledged")
        serialized = str(result).lower()
        for forbidden in (
            "predicted_label",
            "confidence",
            "head_predictions",
            "is_alert",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            get.call_args_list[1].kwargs["headers"],
            {"X-FedKube-Observation-Token": "o" * 32},
        )

    def test_pipeline_waits_without_timeout_and_uses_confirmed_failures(self) -> None:
        waiting = _pipeline_snapshot(
            {"status": "completed", "succeeded": 3, "failed": 0},
            {"gateway_received": 0, "received": 0},
        )
        self.assertEqual(waiting["stages"][1]["status"], "waiting")
        self.assertEqual(waiting["stages"][-1]["status"], "idle")
        failed = _pipeline_snapshot(
            {"status": "completed", "succeeded": 3, "failed": 0},
            {
                "received": 3,
                "accepted": 2,
                "late_dropped": 1,
                "windowed": 2,
                "predicted": 1,
                "inference_failures": 1,
                "alert_sink_failures": 1,
            },
        )
        self.assertEqual(failed["stages"][4]["status"], "error")
        self.assertEqual(failed["stages"][6]["status"], "error")
        self.assertEqual(failed["stages"][7]["status"], "error")

    def test_gateway_overwrites_internal_hop_marker(self) -> None:
        gateway = (
            ROOT / "deploy/application/helm/detection-stack/templates/gateway.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'proxy_set_header X-FedKube-Gateway-Hop "internal-nginx-v1";',
            gateway,
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
