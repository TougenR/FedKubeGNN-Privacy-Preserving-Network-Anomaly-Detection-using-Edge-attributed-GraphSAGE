from __future__ import annotations

import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from src.application.traffic_agent.app import (
    StartTrafficRun,
    require_token,
    state,
)


class TrafficAgentApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        state.clear()

    def test_token_is_required_and_compared_exactly(self) -> None:
        state["token"] = "a" * 32
        require_token("Bearer " + "a" * 32)
        for authorization in (None, "", "Bearer " + "b" * 32, "Basic YTpi"):
            with self.subTest(authorization=authorization), self.assertRaises(
                HTTPException
            ) as raised:
                require_token(authorization)
            self.assertEqual(raised.exception.status_code, 401)

    def test_start_contract_accepts_only_profile_and_bounded_controls(self) -> None:
        self.assertEqual(
            StartTrafficRun.model_validate(
                {"profile_id": "okiru", "events": 10, "interval_ms": 500}
            ).profile_id,
            "okiru",
        )
        with self.assertRaises(ValidationError):
            StartTrafficRun.model_validate(
                {
                    "profile_id": "okiru",
                    "target": "8.8.8.8",
                    "events": 10,
                    "interval_ms": 500,
                }
            )


if __name__ == "__main__":
    unittest.main()
