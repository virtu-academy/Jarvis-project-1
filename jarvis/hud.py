"""HUD — Jarvis's visual face, served as a local web page.

A tiny aiohttp server hosts an Iron-Man-style HUD (jarvis/static/hud.html)
and pushes live events to it over a WebSocket: state changes (listening /
thinking / speaking), the conversation transcript, and tool calls.
"""

from __future__ import annotations

import asyncio
import webbrowser
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

STATIC_DIR = Path(__file__).parent / "static"


class HUD:
    def __init__(self, port: int = 8899):
        self.port = port
        self.clients: set[web.WebSocketResponse] = set()
        self.history: list[dict[str, Any]] = []  # replayed to newly connected pages
        self.app = web.Application()
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/ws", self._ws)
        self._runner: web.AppRunner | None = None

    async def start(self, open_browser: bool = True) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        url = f"http://127.0.0.1:{self.port}"
        print(f"[hud] visual interface online at {url}")
        if open_browser:
            webbrowser.open(url)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    # ---- event API (fire-and-forget, called from the main loop) ----

    def emit(self, type_: str, **data: Any) -> None:
        event = {"type": type_, **data}
        if type_ in ("user", "jarvis", "tool"):  # persistent transcript items
            self.history.append(event)
        elif type_ == "jarvis_done" and data.get("text"):
            # store the completed streamed reply so reconnects see it
            self.history.append({"type": "jarvis", "text": data["text"]})
        self.history = self.history[-100:]
        for ws in list(self.clients):
            asyncio.ensure_future(self._send(ws, event))

    async def _send(self, ws: web.WebSocketResponse, event: dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:
            self.clients.discard(ws)

    # ---- http handlers ----

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC_DIR / "hud.html")

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            for event in self.history[-50:]:
                await ws.send_json(event)
            async for msg in ws:
                if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            self.clients.discard(ws)
        return ws


class NullHUD:
    """Used with --no-hud: swallows all events."""

    async def start(self, open_browser: bool = True) -> None:
        pass

    async def stop(self) -> None:
        pass

    def emit(self, type_: str, **data: Any) -> None:
        pass
