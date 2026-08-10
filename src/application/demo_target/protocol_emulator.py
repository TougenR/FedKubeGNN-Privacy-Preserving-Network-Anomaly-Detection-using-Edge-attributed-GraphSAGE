"""Fixed SSH/IRC protocol emulator for private, bounded IoT-23 lab traffic."""

from __future__ import annotations

import argparse
import asyncio
import struct
from contextlib import suppress


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError, TimeoutError):
        await asyncio.wait_for(writer.wait_closed(), timeout=2)


def _ssh_string(value: bytes) -> bytes:
    return struct.pack("!I", len(value)) + value


def _ssh_kexinit_packet(total_size: int) -> bytes:
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
        return b"\x14" + b"s" * 16 + b"".join(map(_ssh_string, values)) + b"\0" * 5

    base = payload(b"")
    filler_size = total_size - 9 - len(base)
    if total_size % 8 or filler_size < 0:
        raise ValueError("SSH KEXINIT target size is invalid.")
    filler = b"," + b"x" * (filler_size - 1) if filler_size else b""
    body = payload(filler)
    packet_length = 1 + len(body) + 4
    packet = struct.pack("!IB", packet_length, 4) + body + b"\0" * 4
    if len(packet) != total_size:
        raise ValueError("SSH KEXINIT packet size drifted.")
    return packet


async def _send_chunks(
    writer: asyncio.StreamWriter,
    payload: bytes,
    count: int,
) -> None:
    for index in range(count):
        start = len(payload) * index // count
        end = len(payload) * (index + 1) // count
        writer.write(payload[start:end])
        await writer.drain()
        await asyncio.sleep(0.3)


async def handle_ssh(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Exchange SSH identification and size-controlled KEXINIT packets."""
    try:
        writer.write(b"SSH-2.0-FedKube_IoT23_Lab\r\n")
        await writer.drain()
        banner = await asyncio.wait_for(reader.readline(), timeout=5)
        if not banner.startswith(b"SSH-"):
            return
        header = await asyncio.wait_for(reader.readexactly(4), timeout=5)
        packet_length = struct.unpack("!I", header)[0]
        packet = await asyncio.wait_for(reader.readexactly(packet_length), timeout=5)
        padding_length = packet[0]
        if not packet[1 : packet_length - padding_length].startswith(b"\x14"):
            return
        await _send_chunks(writer, _ssh_kexinit_packet(1776), 8)
    except (ConnectionError, TimeoutError, asyncio.IncompleteReadError):
        pass
    finally:
        await _close(writer)


async def handle_irc(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Serve a minimal IRC-like dialogue on the fixed lab listener."""
    try:
        writer.write(b":fedkube NOTICE AUTH :IoT-23 bounded IRC lab\r\n")
        await writer.drain()
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=6)
            if not line or line.startswith(b"QUIT"):
                break
            if line.startswith(b"PING"):
                token = line.partition(b":")[2].strip() or b"fedkube"
                writer.write(b"PONG :" + token + b"\r\n")
            elif line.startswith(b"USER"):
                writer.write(b":fedkube 001 fedkube-lab :Welcome\r\n")
            else:
                writer.write(b":fedkube NOTICE fedkube-lab :bounded\r\n")
            await writer.drain()
    except (ConnectionError, TimeoutError):
        pass
    finally:
        await _close(writer)


async def serve(*, host: str, ssh_port: int, irc_port: int) -> None:
    ssh = await asyncio.start_server(handle_ssh, host, ssh_port)
    irc = await asyncio.start_server(handle_irc, host, irc_port)
    async with ssh, irc:
        await asyncio.gather(ssh.serve_forever(), irc.serve_forever())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ssh-port", type=int, default=10022)
    parser.add_argument("--irc-port", type=int, default=16667)
    args = parser.parse_args()
    asyncio.run(serve(host=args.host, ssh_port=args.ssh_port, irc_port=args.irc_port))


if __name__ == "__main__":
    main()
