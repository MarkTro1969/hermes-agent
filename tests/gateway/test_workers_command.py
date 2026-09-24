from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_busy import GatewayBusySessionMixin
from gateway.session import SessionSource
from hermes_cli.commands import gateway_help_lines, resolve_command
from hermes_cli.commands_platforms import slack_subcommand_map, telegram_bot_commands


def _event(platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text="/workers",
        source=SessionSource(
            platform=platform,
            chat_id="chat-1",
            user_id="user-1",
            user_name="tester",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
async def test_workers_gateway_handler_uses_shared_formatter(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "hermes_cli.kanban_workers.format_kanban_workers",
        lambda: "🟢 **Kanban workers: 1 working**",
    )
    runner = object.__new__(GatewayRunner)
    runner._handle_agents_command = AsyncMock(return_value="Hermes agent detail")

    result = await runner._handle_workers_command(_event())

    assert result == (
        "**Hermes agents**\nHermes agent detail\n\n"
        "**Kanban workers**\n🟢 **Kanban workers: 1 working**"
    )
    runner._handle_agents_command.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agents", "kanban"),
    [
        ("1 active agent", "**Kanban workers: none working**"),
        ("No active agents", "🟢 **Kanban workers: 1 working**"),
        ("1 active agent", "🟢 **Kanban workers: 1 working**"),
        ("No active agents", "**Kanban workers: none working**"),
        ("No active agents", "⚠️ **1 running card without a live worker**"),
    ],
    ids=("local-only", "kanban-only", "both", "neither", "stale-kanban"),
)
async def test_workers_combines_truthful_source_states(monkeypatch, agents, kanban):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("hermes_cli.kanban_workers.format_kanban_workers", lambda: kanban)
    runner = object.__new__(GatewayRunner)
    runner._handle_agents_command = AsyncMock(return_value=agents)

    result = await runner._handle_workers_command(_event())

    assert "**Hermes agents**" in result
    assert agents in result
    assert "**Kanban workers**" in result
    assert kanban in result


@pytest.mark.asyncio
async def test_workers_keeps_healthy_section_when_one_source_fails(monkeypatch):
    from gateway.run import GatewayRunner

    def fail_kanban():
        raise RuntimeError("board unavailable")

    monkeypatch.setattr("hermes_cli.kanban_workers.format_kanban_workers", fail_kanban)
    runner = object.__new__(GatewayRunner)
    runner._handle_agents_command = AsyncMock(return_value="1 active agent")

    result = await runner._handle_workers_command(_event())

    assert "1 active agent" in result
    assert "Could not read Kanban worker status." in result


@pytest.mark.asyncio
async def test_workers_keeps_kanban_section_when_agents_source_fails(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "hermes_cli.kanban_workers.format_kanban_workers",
        lambda: "🟢 **Kanban workers: 1 working**",
    )
    runner = object.__new__(GatewayRunner)
    runner._handle_agents_command = AsyncMock(side_effect=RuntimeError("agent state unavailable"))

    result = await runner._handle_workers_command(_event())

    assert "Could not read Hermes agent status." in result
    assert "🟢 **Kanban workers: 1 working**" in result


def test_workers_combined_output_is_bounded_and_preserves_both_sections():
    from gateway.slash_commands import render_worker_sections

    result = render_worker_sections("agent row\n" * 1000, "kanban row\n" * 1000)

    assert len(result) <= 3900
    assert result.startswith("**Hermes agents**")
    assert "**Kanban workers**" in result
    assert "use `/agents` for full details" in result
    assert "use `/kanban` for full details" not in result
    assert result.endswith("…section truncated.")


@pytest.mark.parametrize(
    ("platform", "expected_limit"),
    [
        (Platform.DISCORD, 1900),
        (Platform.TELEGRAM, 3900),
        (Platform.SLACK, 39000),
    ],
)
def test_workers_uses_single_message_platform_limits(platform, expected_limit):
    from gateway.slash_commands import _worker_output_limit, render_worker_sections

    result = render_worker_sections(
        "agent row\n" * 10000,
        "kanban row\n" * 10000,
        limit=_worker_output_limit(_event(platform)),
    )

    assert len(result) <= expected_limit
    assert "**Hermes agents**" in result
    assert "**Kanban workers**" in result


def test_workers_is_registered_for_busy_dispatch():
    command = resolve_command("workers")
    assert command is not None
    assert command.description == "Show Hermes agents and Kanban workers"
    assert command.gateway_only is True
    assert command.busy_policy == "dispatch"
    assert "workers" in GatewayBusySessionMixin._PLAIN_COMMANDS
    assert any("/workers" in line and command.description in line for line in gateway_help_lines())
    assert ("workers", command.description) in telegram_bot_commands()
    assert slack_subcommand_map()["workers"] == "/workers"


@pytest.mark.asyncio
async def test_busy_workers_command_dispatches_without_interrupting():
    handler = AsyncMock(return_value="workers ok")

    class Runner(GatewayBusySessionMixin):
        class _Scope:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *_args):
                return None

        def _async_profile_scope_for_source(self, _source):
            return self._Scope()

        def _gateway_plain_command_handlers(self):
            return {"workers": handler}

    runner = Runner()
    result = await runner._dispatch_busy_slash_command(
        _event(), resolve_command("workers"), "session-key", object()
    )

    assert result == "workers ok"
    handler.assert_awaited_once()
