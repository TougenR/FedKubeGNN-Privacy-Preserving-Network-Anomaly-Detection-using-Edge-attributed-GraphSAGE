from __future__ import annotations

import asyncio
import unittest

from src.application.demo_target.protocol_emulator import handle_irc, handle_ssh
from src.application.traffic_agent.executor import send_tcp_session


class ProtocolEmulatorTests(unittest.IsolatedAsyncioTestCase):
    async def _exercise(self, handler, protocol: str, *, complete: bool = True) -> None:
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = int(server.sockets[0].getsockname()[1])
        async with server:
            result = await asyncio.to_thread(
                send_tcp_session,
                source="127.0.0.1",
                destination="127.0.0.1",
                destination_port=port,
                protocol=protocol,
                complete=complete,
            )
        self.assertTrue(result)

    async def test_ssh_exchange_completes(self) -> None:
        await self._exercise(handle_ssh, "ssh")

    async def test_irc_complete_and_partial_exchanges_complete(self) -> None:
        await self._exercise(handle_irc, "irc")
        await self._exercise(handle_irc, "irc", complete=False)


if __name__ == "__main__":
    unittest.main()
