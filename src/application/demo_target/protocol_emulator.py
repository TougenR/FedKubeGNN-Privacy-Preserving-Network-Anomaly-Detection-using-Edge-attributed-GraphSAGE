"""Fixed SSH/IRC protocol emulator for private, bounded IoT-23 lab traffic."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError, TimeoutError):
        await asyncio.wait_for(writer.wait_closed(), timeout=2)


async def handle_ssh(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Exchange one SSH identification and twelve bounded payload blocks."""
    try:
        writer.write(b"SSH-2.0-FedKube_IoT23_Lab\r\n")
        await writer.drain()
        banner = await asyncio.wait_for(reader.readline(), timeout=5)
        if not banner.startswith(b"SSH-"):
            return
        for index in range(12):
            payload = await asyncio.wait_for(reader.read(256), timeout=5)
            if not payload:
                break
            writer.write(bytes([97 + index % 26]) * 96)
            await writer.drain()
    except (ConnectionError, TimeoutError):
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
