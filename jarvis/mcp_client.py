"""Generic MCP client manager.

Connects to any MCP servers listed in mcp_servers.json (HTTP or stdio
transports), exposes their tools in Anthropic tool format, and dispatches
tool calls back to the right server.

To plug in Higgsfield (or anything else): add its entry to
mcp_servers.json, set "enabled": true, and restart Jarvis.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # older mcp versions
    streamablehttp_client = None


class MCPManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.sessions: dict[str, ClientSession] = {}
        self.tool_index: dict[str, str] = {}  # tool name -> server name
        self._stack = AsyncExitStack()

    async def connect_all(self) -> None:
        if not self.config_path.exists():
            return
        config = json.loads(self.config_path.read_text())
        for name, server in config.get("servers", {}).items():
            if not server.get("enabled", False):
                continue
            try:
                await self._connect(name, server)
                print(f"[mcp] connected to '{name}'")
            except Exception as e:
                print(f"[mcp] failed to connect to '{name}': {e}")

    async def _connect(self, name: str, server: dict[str, Any]) -> None:
        transport = server.get("transport", "http")
        if transport == "stdio":
            params = StdioServerParameters(
                command=server["command"],
                args=server.get("args", []),
                env=server.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        elif transport == "http":
            if streamablehttp_client is None:
                raise RuntimeError("mcp package too old for HTTP transport — pip install -U mcp")
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(server["url"], headers=server.get("headers"))
            )
        else:
            raise ValueError(f"unknown transport '{transport}'")

        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = session

        tools = await session.list_tools()
        for tool in tools.tools:
            self.tool_index[tool.name] = name

    async def anthropic_tools(self) -> list[dict[str, Any]]:
        """All connected servers' tools, in Anthropic tool-definition format."""
        defs: list[dict[str, Any]] = []
        for name, session in self.sessions.items():
            result = await session.list_tools()
            for tool in result.tools:
                defs.append(
                    {
                        "name": tool.name,
                        "description": tool.description or f"Tool from MCP server '{name}'",
                        "input_schema": tool.inputSchema,
                    }
                )
        return defs

    async def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        server_name = self.tool_index.get(tool_name)
        if server_name is None:
            return f"Error: no MCP server provides a tool named '{tool_name}'."
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, tool_input)
        parts = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "(tool returned no content)"

    async def close(self) -> None:
        await self._stack.aclose()
