"""Delivery-only context/allowance/compaction notices on a finished gateway turn.

Contract: the context notice fires only on a provider-measured occupancy, the allowance notice only on
real rate-limit header data, the compaction confirmation only on a real in-place compaction event; a
failed turn never alerts; each condition warns once per session while it holds (re-arming after
recovery); the notice rides the turn's own reply, never opens a send of its own, and is never written
to the transcript.
"""

import sys
import threading
import time
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from agent.rate_limit_tracker import RateLimitBucket, RateLimitState
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run_turn_usage_alerts import COMPACTION_NOTICE, GatewayTurnUsageAlertsMixin
from gateway.session import SessionEntry, SessionSource

SESSION_KEY = "agent:main:telegram:group:-1001:12345"
SESSION_ID = "sess-usage-alerts"


def _compressor(used, maximum=100_000, *, measured=True):
    # context_display_source: a built-in ledger whose shown figure differs from the real one is a seed.
    return SimpleNamespace(last_prompt_tokens=used, context_length=maximum,
                           last_real_prompt_tokens=used if measured else used - 1)


def _rate_state(used_pct=None):
    if used_pct is None:
        return RateLimitState()
    now = time.time()
    bucket = RateLimitBucket(limit=100, remaining=100 - used_pct, reset_seconds=600, captured_at=now)
    return RateLimitState(requests_hour=bucket, captured_at=now, provider="test")


def _agent(context_used=0, *, measured=True, rate=None):
    state = rate if rate is not None else RateLimitState()
    return SimpleNamespace(context_compressor=_compressor(context_used, measured=measured),
                           get_rate_limit_state=lambda: state)


def _compacting(attempt_id, used=0):
    agent = _agent(used)
    agent._compression_attempt_id = attempt_id
    return agent


class _Host(GatewayTurnUsageAlertsMixin):
    pass


def _alert(host, agent, session=SESSION_ID, result=None, failed=False):
    return host._hmwa_usage_alert_text(agent, session, result or {}, failed)


# ── decision logic ──────────────────────────────────────────────────────


def test_estimated_context_never_alerts():
    assert _alert(_Host(), _agent(95_000, measured=False)) == ""


def test_measured_context_uses_legacy_text_once_and_rearms_after_recovery():
    host = _Host()
    assert _alert(host, _agent(79_999)) == ""
    assert _alert(host, _agent(85_000)) == (
        "⚠️ Estimated context occupancy is 85% (85,000 of 100,000 tokens). Consider starting a fresh topic."
    )
    assert _alert(host, _agent(90_000)) == ""
    assert _alert(host, _agent(40_000)) == ""  # recovery
    assert "82%" in _alert(host, _agent(82_000))


def test_once_per_session_not_global():
    host = _Host()
    assert _alert(host, _agent(85_000), "a")
    assert _alert(host, _agent(85_000), "b")


def test_allowance_requires_real_header_data_and_uses_legacy_text():
    host = _Host()
    assert _alert(host, _agent(rate=_rate_state(None))) == ""
    assert _alert(host, _agent(rate=_rate_state(81))) == (
        "⚠️ test API requests/hour allowance is 19% remaining. Wait for reset or reduce requests."
    )
    assert _alert(host, _agent(rate=_rate_state(95))) == ""
    assert _alert(host, _agent(rate=_rate_state(80))) == ""  # 20% remaining is recovery
    assert _alert(host, _agent(rate=_rate_state(81)))


def test_allowance_without_provider_name_is_silent():
    state = _rate_state(95)
    state.provider = ""
    assert _alert(_Host(), _agent(rate=state)) == ""


def test_expired_rate_limit_window_is_not_current_usage():
    stale = _rate_state(95)
    stale.requests_hour.captured_at = time.time() - 3600
    assert _alert(_Host(), _agent(rate=stale)) == ""


def test_compaction_confirms_each_real_event_once():
    host = _Host()
    done = {"compacted_in_place": True}
    assert _alert(host, _compacting("attempt-1"), result=done) == COMPACTION_NOTICE
    assert _alert(host, _compacting("attempt-1"), result=done) == ""
    assert _alert(host, _compacting("attempt-2"), result=done) == COMPACTION_NOTICE


def test_compaction_needs_in_place_flag_and_a_genuine_event_id():
    host = _Host()
    assert _alert(host, _compacting("attempt-1"), result={"compacted_in_place": False}) == ""
    assert _alert(host, _compacting("attempt-1")) == ""
    # No event id: the session id must never stand in for one.
    assert _alert(host, _compacting(None), result={"compacted_in_place": True, "session_id": SESSION_ID}) == ""


def test_one_notice_per_turn_context_first():
    agent = _compacting("attempt-1", used=85_000)
    agent.get_rate_limit_state = lambda: _rate_state(95)
    text = _alert(_Host(), agent, result={"compacted_in_place": True})
    assert text.startswith("⚠️ Estimated context occupancy") and "\n" not in text


def test_failed_turn_neither_alerts_nor_rearms():
    host = _Host()
    assert _alert(host, _agent(85_000))
    # A failed turn reporting recovery must not re-arm the warning...
    assert _alert(host, _agent(10_000), failed=True) == ""
    # ...so the condition still holding afterwards stays silent.
    assert _alert(host, _agent(85_000)) == ""
    assert _alert(_Host(), _agent(85_000), failed=True) == ""


# ── real turn handler: delivered with the reply, never persisted ────────


def _runner(monkeypatch, tmp_path, agent):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    # Keep this focused test independent of first-contact home-channel onboarding.
    runner._hmwa_first_contact_notes = AsyncMock(return_value=None)
    runner._session_db = None
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._agent_cache = {SESSION_KEY: (agent, "sig", 0)}
    runner._agent_cache_lock = threading.Lock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY, session_id=SESSION_ID, created_at=datetime.now(),
        updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.transcript_tail_role.return_value = "user"
    runner.session_store.append_to_transcript = MagicMock()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 100_000)
    return runner


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="12345")


def _event(n=1):
    return MessageEvent(text="hello", source=_source(), message_id=f"msg-{n}")


def _result(**extra):
    return {"final_response": "Hi there!", "history_offset": 0, "last_prompt_tokens": 85_000,
            "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi there!"}],
            **extra}


def _persisted_text(runner):
    return "\n".join(str(c.args[1]) for c in runner.session_store.append_to_transcript.call_args_list)


@pytest.mark.asyncio
async def test_alert_is_delivered_but_never_persisted(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, _agent(85_000, rate=_rate_state(90)))
    runner._run_agent = AsyncMock(return_value=_result())

    reply = await runner._handle_message_with_agent(_event(1), _source(), SESSION_KEY, 1)

    assert reply.startswith("Hi there!\n\n⚠️ Estimated context occupancy is 85%")
    assert "allowance" not in reply  # one notice per turn: context outranks allowance
    persisted = _persisted_text(runner)
    assert "Hi there!" in persisted
    assert "⚠️" not in persisted

    # Threshold deduplication is exercised directly above. Re-entering the full
    # gateway without completing its delivery phase deliberately retains the turn
    # lease, so a second call here would test lease cleanup rather than alert state.


@pytest.mark.asyncio
async def test_in_place_compaction_is_confirmed_in_the_reply(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, _compacting("attempt-1", used=10_000))
    runner._run_agent = AsyncMock(return_value=_result(compacted_in_place=True))

    reply = await runner._handle_message_with_agent(_event(), _source(), SESSION_KEY, 1)

    assert reply == f"Hi there!\n\n{COMPACTION_NOTICE}"
    assert COMPACTION_NOTICE not in _persisted_text(runner)


@pytest.mark.asyncio
async def test_estimated_context_turn_delivers_no_alert(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, _agent(95_000, measured=False))
    runner._run_agent = AsyncMock(return_value=_result())
    assert await runner._handle_message_with_agent(_event(), _source(), SESSION_KEY, 1) == "Hi there!"


@pytest.mark.asyncio
async def test_failed_turn_delivers_no_alert(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, _agent(85_000, rate=_rate_state(90)))
    runner._run_agent = AsyncMock(return_value=_result(failed=True, error="boom"))
    reply = await runner._handle_message_with_agent(_event(), _source(), SESSION_KEY, 1)
    assert "⚠️" not in (reply or "")


def _streaming(monkeypatch, runner, footer):
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._streaming_tts_turn_completed = lambda *_a, **_k: False
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda _source: adapter)
    monkeypatch.setattr(runner, "_deliver_media_from_response", AsyncMock())
    monkeypatch.setattr(runner, "_hmwa_runtime_footer_line", lambda *_a, **_k: footer)
    runner._run_agent = AsyncMock(return_value=_result(already_sent=True))
    return adapter


@pytest.mark.asyncio
async def test_streamed_turn_appends_alert_to_existing_footer(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path, _agent(85_000))
    adapter = _streaming(monkeypatch, runner, "model · 1s")

    assert await runner._handle_message_with_agent(_event(), _source(), SESSION_KEY, 1) is None
    sent = [c.args[1] for c in adapter.send.call_args_list]
    assert len(sent) == 1 and sent[0].startswith("model · 1s\n\n⚠️ Estimated context occupancy is 85%")
    assert "⚠️" not in _persisted_text(runner)


@pytest.mark.asyncio
async def test_streamed_turn_without_footer_sends_nothing_and_keeps_alert_armed(monkeypatch, tmp_path):
    agent = _agent(85_000)
    runner = _runner(monkeypatch, tmp_path, agent)
    adapter = _streaming(monkeypatch, runner, "")

    assert await runner._handle_message_with_agent(_event(), _source(), SESSION_KEY, 1) is None
    adapter.send.assert_not_called()
    # Not latched while undelivered: the next reply that can carry it still gets it.
    assert "85%" in runner._hmwa_usage_alert_text(agent, SESSION_KEY, {}, False)
