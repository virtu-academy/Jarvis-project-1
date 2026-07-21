"""Brain — Claude drives the conversation and calls MCP tools.

Runs the agentic loop manually: stream a response, speak text deltas as
they arrive, execute any tool calls via the MCP manager, feed results
back, repeat until Claude is done.
"""

from __future__ import annotations

from typing import Any, Callable

from anthropic import AsyncAnthropic

from .config import JARVIS_SYSTEM_PROMPT, Config
from .mcp_client import MCPManager


class Brain:
    def __init__(self, config: Config, mcp: MCPManager):
        self.config = config
        self.mcp = mcp
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key)
        self.messages: list[dict[str, Any]] = []

    async def respond(self, user_text: str, on_text: Callable[[str], None]) -> str:
        """Process one user turn. Calls on_text(delta) as speech-ready text
        streams in; returns the full assistant text for the turn."""
        self.messages.append({"role": "user", "content": user_text})
        tools = await self.mcp.anthropic_tools()
        spoken: list[str] = []

        while True:
            kwargs: dict[str, Any] = dict(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": JARVIS_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                messages=self.messages,
            )
            if tools:
                kwargs["tools"] = tools

            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        on_text(event.delta.text)
                        spoken.append(event.delta.text)
                response = await stream.get_final_message()

            # Preserve the full content (incl. thinking + tool_use blocks)
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"\n[brain] calling tool: {block.name}")
                        try:
                            result = await self.mcp.call_tool(block.name, block.input)
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result,
                                }
                            )
                        except Exception as e:
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": f"Tool error: {e}",
                                    "is_error": True,
                                }
                            )
                self.messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "pause_turn":
                continue  # server-side pause — resend to resume

            return "".join(spoken)
