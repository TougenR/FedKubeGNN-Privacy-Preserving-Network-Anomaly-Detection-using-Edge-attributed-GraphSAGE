"""Async execution of bounded traffic against immutable lab targets."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.application.scenario_runner.catalog import ScenarioCatalog


class ScenarioConflictError(RuntimeError):
    pass


def _is_internal_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        return False
    hostname = parsed.hostname
    try:
        return ipaddress.ip_address(hostname).is_private
    except ValueError:
        return "." not in hostname or hostname.endswith(
            (".svc", ".svc.cluster.local")
        )


@dataclass
class RunRecord:
    run_id: str
    scenario_id: str
    sensor_id: str
    parameters: dict[str, int]
    status: str
    started_at: float
    finished_at: float | None = None
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_get(
    url: str, timeout_seconds: float, *, headers: dict[str, str] | None = None
) -> bool:
    request = Request(
        url,
        headers={"User-Agent": "FedKube-Lab-Runner/1", **(headers or {})},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read(1024)
            return 200 <= int(response.status) < 500
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


class ScenarioExecutor:
    def __init__(
        self,
        *,
        catalog: ScenarioCatalog,
        target_url: str,
        scan_urls: list[str],
    ) -> None:
        if not _is_internal_http_url(target_url):
            raise ValueError("demo target must be an internal HTTP URL")
        if len(scan_urls) != 6 or any(not _is_internal_http_url(url) for url in scan_urls):
            raise ValueError("exactly six internal port-probe URLs are required")
        self.catalog = catalog
        self.target_url = target_url.rstrip("/")
        self.scan_urls = tuple(scan_urls)
        self._record: RunRecord | None = None
        self._task: asyncio.Task[None] | None = None
        self._cancel = asyncio.Event()
        self._lock = asyncio.Lock()
        self._start_gate = asyncio.Event()
        self._start_gate.set()

    async def start(
        self,
        scenario_id: str,
        raw_parameters: dict[str, Any],
        *,
        gated: bool = False,
    ) -> RunRecord:
        scenario = self.catalog.scenario(scenario_id)
        parameters = scenario.validate_parameters(raw_parameters)
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise ScenarioConflictError("another lab scenario is already running")
            self._cancel = asyncio.Event()
            self._start_gate = asyncio.Event()
            if not gated:
                self._start_gate.set()
            self._record = RunRecord(
                run_id=f"demo-{uuid.uuid4().hex[:12]}",
                scenario_id=scenario_id,
                sensor_id=self.catalog.sensor_id,
                parameters=parameters,
                status="running",
                started_at=time.time(),
            )
            self._task = asyncio.create_task(self._execute_after_gate(self._record))
            return self._record

    def release(self) -> None:
        """Release a gated run after collector correlation is registered."""
        self._start_gate.set()

    async def stop(self) -> RunRecord | None:
        self._cancel.set()
        self._start_gate.set()
        if self._task is not None and not self._task.done():
            await self._task
        return self._record

    def current(self) -> RunRecord | None:
        return self._record

    async def _request(self, url: str, timeout: float = 15.0) -> None:
        if self._record is None or self._cancel.is_set():
            return
        self._record.attempted += 1
        ok = await asyncio.to_thread(
            _bounded_get,
            url,
            timeout,
            headers={
                "X-FedKube-Demo-Run": self._record.run_id,
                "X-FedKube-Demo-Scenario": self._record.scenario_id,
            },
        )
        if ok:
            self._record.succeeded += 1
        else:
            self._record.failed += 1

    async def _pause(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._cancel.wait(), timeout=max(0.0, seconds))
        except TimeoutError:
            pass

    async def _execute(self, record: RunRecord) -> None:
        try:
            params = record.parameters
            if record.scenario_id == "benign-browsing":
                for index in range(params["request_count"]):
                    if self._cancel.is_set():
                        break
                    await self._request(f"{self.target_url}/payload/{128 + index % 4 * 64}")
                    await self._pause(params["interval_ms"] / 1000)
            elif record.scenario_id == "connection-burst":
                for _ in range(params["bursts"]):
                    if self._cancel.is_set():
                        break
                    await asyncio.gather(
                        *(self._request(f"{self.target_url}/payload/256") for _ in range(params["concurrency"]))
                    )
                    await self._pause(params["pause_ms"] / 1000)
            elif record.scenario_id == "request-flood":
                for _ in range(params["duration_seconds"]):
                    if self._cancel.is_set():
                        break
                    started = time.monotonic()
                    await asyncio.gather(
                        *(self._request(f"{self.target_url}/payload/64") for _ in range(params["requests_per_second"]))
                    )
                    await self._pause(max(0.0, 1.0 - (time.monotonic() - started)))
            elif record.scenario_id == "slow-connections":
                delay = params["delay_ms"]
                await asyncio.gather(
                    *(self._request(f"{self.target_url}/slow/{delay}", delay / 1000 + 5) for _ in range(params["connections"]))
                )
            elif record.scenario_id == "port-probe":
                for url in self.scan_urls[: params["port_count"]]:
                    if self._cancel.is_set():
                        break
                    await self._request(f"{url}/probe")
            elif record.scenario_id == "periodic-beacon":
                for _ in range(params["events"]):
                    if self._cancel.is_set():
                        break
                    await self._request(f"{self.target_url}/payload/32")
                    await self._pause(params["interval_ms"] / 1000)
            else:  # Catalog validation makes this unreachable.
                raise ValueError(f"unsupported scenario '{record.scenario_id}'")
            record.status = "cancelled" if self._cancel.is_set() else "completed"
        except Exception as exc:  # Preserve a bounded public error, never a target URL.
            record.status = "failed"
            record.error = type(exc).__name__
        finally:
            record.finished_at = time.time()

    async def _execute_after_gate(self, record: RunRecord) -> None:
        await self._start_gate.wait()
        await self._execute(record)


def scan_urls_from_json(raw: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("DEMO_SCAN_URLS must be a JSON string list")
    return value
