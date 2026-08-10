"""Single-run executor for fixed private traffic profiles."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import struct
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from src.application.traffic_agent.catalog import (
    TrafficProfile,
    TrafficProfileCatalog,
    TrafficTargetCatalog,
)


class TrafficRunConflictError(RuntimeError):
    pass


class TrafficProfileDisabledError(RuntimeError):
    pass


def _checksum(payload: bytes) -> int:
    if len(payload) % 2:
        payload += b"\0"
    value = sum(struct.unpack(f"!{len(payload) // 2}H", payload))
    while value >> 16:
        value = (value & 0xFFFF) + (value >> 16)
    return (~value) & 0xFFFF


def send_tcp_packet(
    *,
    source: str,
    destination: str,
    destination_port: int,
    sequence: int,
    flags: int,
    corrupt_checksum: bool = False,
) -> bool:
    """Send one header-only TCP packet; the caller owns fixed-target authority."""
    source_bytes = socket.inet_aton(source)
    destination_bytes = socket.inet_aton(destination)
    source_port = 40000 + sequence % 20000
    tcp = struct.pack(
        "!HHLLBBHHH",
        source_port,
        destination_port,
        sequence,
        1 if flags & 0x10 else 0,
        5 << 4,
        flags,
        64240,
        0,
        0,
    )
    pseudo = struct.pack(
        "!4s4sBBH", source_bytes, destination_bytes, 0, socket.IPPROTO_TCP, len(tcp)
    )
    tcp_checksum = _checksum(pseudo + tcp)
    if corrupt_checksum:
        tcp_checksum ^= 0xFFFF
    tcp = tcp[:16] + struct.pack("!H", tcp_checksum) + tcp[18:]
    total_length = 20 + len(tcp)
    ip_header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_length,
        sequence % 65535,
        0,
        64,
        socket.IPPROTO_TCP,
        0,
        source_bytes,
        destination_bytes,
    )
    ip_checksum = _checksum(ip_header)
    ip_header = ip_header[:10] + struct.pack("!H", ip_checksum) + ip_header[12:]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_RAW) as raw:
            raw.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            raw.sendto(ip_header + tcp, (destination, destination_port))
        return True
    except OSError:
        return False


def send_http_get(endpoint: str) -> bool:
    request = Request(endpoint, headers={"User-Agent": "FedKube-Traffic-Agent/1"})
    try:
        with urlopen(request, timeout=10) as response:
            response.read(1024)
            return 200 <= int(response.status) < 500
    except (HTTPError, URLError, OSError, TimeoutError):
        return False


def _read_line(connection: socket.socket, limit: int = 512) -> bytes:
    payload = bytearray()
    while len(payload) < limit:
        value = connection.recv(1)
        if not value:
            break
        payload.extend(value)
        if value == b"\n":
            break
    return bytes(payload)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        value = connection.recv(size - len(payload))
        if not value:
            break
        payload.extend(value)
    return bytes(payload)


def _ssh_string(value: bytes) -> bytes:
    return struct.pack("!I", len(value)) + value


def _ssh_kexinit_packet(total_size: int, marker: bytes) -> bytes:
    """Build one syntactically valid, size-controlled SSH2 KEXINIT packet."""
    fields = [
        b"curve25519-sha256",
        b"ssh-ed25519",
        b"aes128-ctr",
        b"aes128-ctr",
        b"hmac-sha2-256",
        b"hmac-sha2-256",
        b"none",
        b"none",
        b"",
        b"",
    ]

    def payload(filler: bytes) -> bytes:
        values = [fields[0] + filler, *fields[1:]]
        return b"\x14" + marker[:1] * 16 + b"".join(map(_ssh_string, values)) + b"\0" * 5

    base = payload(b"")
    filler_size = total_size - 9 - len(base)
    if total_size % 8 or filler_size < 0:
        raise ValueError("SSH KEXINIT target size is invalid.")
    filler = b"," + b"x" * (filler_size - 1) if filler_size else b""
    body = payload(filler)
    padding = b"\0" * 4
    packet_length = 1 + len(body) + len(padding)
    packet = struct.pack("!IB", packet_length, len(padding)) + body + padding
    if len(packet) != total_size:
        raise ValueError("SSH KEXINIT packet size drifted.")
    return packet


def _send_chunks(connection: socket.socket, payload: bytes, count: int) -> None:
    for index in range(count):
        start = len(payload) * index // count
        end = len(payload) * (index + 1) // count
        connection.sendall(payload[start:end])
        time.sleep(0.2)


def _drain_socket(connection: socket.socket) -> None:
    while True:
        try:
            if not connection.recv(1024):
                return
        except TimeoutError:
            return


def send_tcp_session(
    *,
    source: str,
    destination: str,
    destination_port: int,
    protocol: str,
    complete: bool = True,
) -> bool:
    """Run one bounded protocol exchange against a catalogued private target."""
    try:
        with socket.create_connection(
            (destination, destination_port),
            timeout=8,
            source_address=(source, 0),
        ) as connection:
            connection.settimeout(5)
            banner = _read_line(connection)
            if protocol == "ssh":
                if not banner.startswith(b"SSH-"):
                    return False
                connection.sendall(b"SSH-2.0-OpenSSH_8.9_FedKube_Lab\r\n")
                _send_chunks(connection, _ssh_kexinit_packet(552, b"c"), 6)
                header = _read_exact(connection, 4)
                if len(header) != 4:
                    return False
                packet_length = struct.unpack("!I", header)[0]
                return len(_read_exact(connection, packet_length)) == packet_length
            if protocol == "irc":
                connection.sendall(b"NICK fedkube-lab\r\n")
                if not _read_line(connection):
                    return False
                if not complete:
                    time.sleep(3.1)
                else:
                    connection.sendall(b"USER fedkube 0 * :IoT23 bounded lab\r\n")
                    if not _read_line(connection):
                        return False
                    connection.sendall(b"PING :fedkube\r\n")
                    if b"PONG" not in _read_line(connection):
                        return False
                    connection.sendall(b"QUIT :bounded-run-complete\r\n")
                connection.shutdown(socket.SHUT_WR)
                _drain_socket(connection)
                return True
            return False
    except (OSError, TimeoutError):
        return False


def _endpoint_host(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    value = parsed.hostname if parsed.scheme else endpoint
    return str(ipaddress.ip_address(str(value)))


@dataclass
class TrafficRunRecord:
    run_id: str
    profile_id: str
    reference_class: str
    reference_digest: str
    scientific_status: str
    events: int
    interval_ms: int
    status: str
    started_at: float
    released_at: float | None = None
    finished_at: float | None = None
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    error: str | None = None

    def public(self) -> dict:
        return asdict(self)


class TrafficExecutor:
    def __init__(
        self,
        *,
        catalog: TrafficProfileCatalog,
        targets: TrafficTargetCatalog,
        packet_sender: Callable[..., bool] = send_tcp_packet,
        http_sender: Callable[[str], bool] = send_http_get,
        session_sender: Callable[..., bool] = send_tcp_session,
        release_timeout_seconds: float = 30.0,
        interval_scale: float = 1.0,
    ) -> None:
        if not 0 < interval_scale <= 1:
            raise ValueError("Interval scale must be in (0, 1].")
        for profile in catalog.profiles:
            if profile.target_group not in targets.groups:
                raise ValueError(
                    f"Profile '{profile.id}' references unknown target group."
                )
            endpoint_count = len(targets.groups[profile.target_group].endpoints)
            if profile.mechanism.endswith("round-robin") and endpoint_count < 2:
                raise ValueError(
                    f"Profile '{profile.id}' requires multiple fixed targets."
                )
        self.catalog = catalog
        self.targets = targets
        self.packet_sender = packet_sender
        self.http_sender = http_sender
        self.session_sender = session_sender
        self.release_timeout_seconds = release_timeout_seconds
        self.interval_scale = interval_scale
        self._record: TrafficRunRecord | None = None
        self._task: asyncio.Task[None] | None = None
        self._gate = asyncio.Event()
        self._cancel = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(
        self,
        profile_id: str,
        *,
        events: int | None = None,
        interval_ms: int | None = None,
    ) -> TrafficRunRecord:
        profile = self.catalog.profile(profile_id)
        if not profile.execution_enabled:
            raise TrafficProfileDisabledError(
                f"Profile '{profile.id}' is {profile.scientific_status}."
            )
        selected_events, selected_interval = profile.resolve_run_controls(
            events=events,
            interval_ms=interval_ms,
        )
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise TrafficRunConflictError("Another traffic run is active.")
            self._gate = asyncio.Event()
            self._cancel = asyncio.Event()
            self._record = TrafficRunRecord(
                run_id=f"traffic-{uuid.uuid4().hex[:12]}",
                profile_id=profile.id,
                reference_class=profile.reference_class,
                reference_digest=self.catalog.reference_digest,
                scientific_status=profile.scientific_status,
                events=selected_events,
                interval_ms=selected_interval,
                status="waiting-for-release",
                started_at=time.time(),
            )
            self._task = asyncio.create_task(self._execute_after_release(profile))
            return self._record

    def release(self, run_id: str) -> TrafficRunRecord:
        if self._record is None or self._record.run_id != run_id:
            raise KeyError(run_id)
        if self._record.status != "waiting-for-release":
            raise TrafficRunConflictError("Traffic run is not waiting for release.")
        self._record.status = "running"
        self._record.released_at = time.time()
        self._gate.set()
        return self._record

    async def stop(self) -> TrafficRunRecord | None:
        self._cancel.set()
        self._gate.set()
        if self._task is not None and not self._task.done():
            await self._task
        return self._record

    def current(self) -> TrafficRunRecord | None:
        return self._record

    async def _wait_interval(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._cancel.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _send(
        self,
        profile: TrafficProfile,
        endpoints: list[str],
        index: int,
    ) -> bool:
        endpoint = endpoints[index % len(endpoints)]
        if profile.mechanism == "http-get":
            return await asyncio.to_thread(self.http_sender, endpoint)
        if profile.mechanism == "ssh-session":
            return await asyncio.to_thread(
                self.session_sender,
                source=self.targets.source_ipv4,
                destination=_endpoint_host(endpoint),
                destination_port=profile.destination_port,
                protocol="ssh",
            )
        if profile.mechanism == "irc-mixed":
            mode = index % 3
            if mode:
                return await asyncio.to_thread(
                    self.session_sender,
                    source=self.targets.source_ipv4,
                    destination=_endpoint_host(endpoints[-1]),
                    destination_port=profile.destination_port,
                    protocol="irc",
                    complete=mode == 2,
                )
            endpoint = endpoints[0]
        flags = 0x02 if profile.mechanism.startswith("syn-only") else 0x10
        if profile.mechanism == "irc-mixed":
            flags = 0x02
        return await asyncio.to_thread(
            self.packet_sender,
            source=self.targets.source_ipv4,
            destination=_endpoint_host(endpoint),
            destination_port=profile.destination_port,
            sequence=index + 1,
            flags=flags,
            corrupt_checksum=profile.mechanism == "ack-only",
        )

    async def _execute(self, profile: TrafficProfile) -> None:
        assert self._record is not None
        endpoints = self.targets.groups[profile.target_group].endpoints
        try:
            for index in range(self._record.events):
                if self._cancel.is_set():
                    break
                self._record.attempted += 1
                if await self._send(profile, endpoints, index):
                    self._record.succeeded += 1
                else:
                    self._record.failed += 1
                if index + 1 < self._record.events:
                    await self._wait_interval(
                        self._record.interval_ms / 1000 * self.interval_scale
                    )
            self._record.status = "cancelled" if self._cancel.is_set() else "completed"
        except Exception as exc:
            self._record.status = "failed"
            self._record.error = type(exc).__name__
        finally:
            self._record.finished_at = time.time()

    async def _execute_after_release(self, profile: TrafficProfile) -> None:
        try:
            await asyncio.wait_for(
                self._gate.wait(), timeout=self.release_timeout_seconds
            )
        except TimeoutError:
            assert self._record is not None
            self._record.status = "failed"
            self._record.error = "release-timeout"
            self._record.finished_at = time.time()
            return
        if self._cancel.is_set():
            assert self._record is not None
            self._record.status = "cancelled"
            self._record.finished_at = time.time()
            return
        await self._execute(profile)
