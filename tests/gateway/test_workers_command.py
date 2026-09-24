from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run_busy import GatewayBusySessionMixin
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command


def _event() -> MessageEvent:
    return MessageEvent(
        text="/workers",
        source=SessionSource(
            platform=Platform.TELEGRAM,
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

    result = await runner._handle_workers_command(_event())

    assert result == "🟢 **Kanban workers: 1 working**"


def test_workers_is_registered_for_busy_dispatch():
    command = resolve_command("workers")
    assert command is not None
    assert command.gateway_only is True
    assert command.busy_policy == "dispatch"
    assert "workers" in GatewayBusySessionMixin._PLAIN_COMMANDS


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
