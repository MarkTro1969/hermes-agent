import asyncio
from types import SimpleNamespace
from typing import Any

from gateway.context_usage_alerts import ContextUsageAlerts, compose_turn_alert, inject_turn_alert
from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource


def _state(provider: str = "OpenAI", *, remaining: Any = 19, limit: Any = 100, has_data: bool = True):
    bucket = SimpleNamespace(remaining=remaining, limit=limit)
    return SimpleNamespace(
        provider=provider, has_data=has_data,
        requests_min=bucket, requests_hour=None, tokens_min=None, tokens_hour=None,
    )


def test_context_alert_crosses_dedupes_and_resets():
    alerts = ContextUsageAlerts()
    assert alerts.context_alert("topic", used_tokens=799, context_length=1000) is None
    assert alerts.context_alert("topic", used_tokens=800, context_length=1000) == (
        "⚠️ Estimated context occupancy is 80% (800 of 1,000 tokens). "
        "Consider starting a fresh topic."
    )
    assert alerts.context_alert("topic", used_tokens=900, context_length=1000) is None
    assert alerts.context_alert("topic", used_tokens=700, context_length=1000) is None
    assert alerts.context_alert("topic", used_tokens=900, context_length=1000) is not None


def test_unavailable_context_telemetry_is_silent():
    alerts = ContextUsageAlerts()
    assert alerts.context_alert("topic", used_tokens=None, context_length=1000) is None
    assert alerts.context_alert("topic", used_tokens=900, context_length=None) is None
    assert alerts.context_alert("other", used_tokens=900, context_length=0) is None


def test_real_provider_header_allowance_dedupes_and_resets():
    alerts = ContextUsageAlerts()
    assert alerts.provider_rate_limit_alert("topic", _state()) == (
        "⚠️ OpenAI API requests/min allowance is 19% remaining. "
        "Wait for reset or reduce requests."
    )
    assert alerts.provider_rate_limit_alert("topic", _state()) is None
    assert alerts.provider_rate_limit_alert("topic", _state(remaining=20)) is None
    assert alerts.provider_rate_limit_alert("topic", _state()) is not None


def test_provider_allowance_requires_real_header_measurements():
    alerts = ContextUsageAlerts()
    assert alerts.provider_rate_limit_alert("topic", None) is None
    assert alerts.provider_rate_limit_alert("topic", _state(has_data=False)) is None
    assert alerts.provider_rate_limit_alert("topic", _state(provider="")) is None
    assert alerts.provider_rate_limit_alert("topic", _state(remaining=None)) is None


def test_compaction_dedupes_a_real_event_not_the_session():
    alerts = ContextUsageAlerts()
    expected = "ℹ️ Conversation history was shortened to preserve context space. Continue normally or start a fresh topic."
    assert alerts.compaction_alert("topic", "attempt-1") == expected
    assert alerts.compaction_alert("topic", "attempt-1") is None
    assert alerts.compaction_alert("topic", "attempt-2") == expected
    assert alerts.compaction_alert("topic", None) is None


def test_real_alert_composition_and_injection_avoids_duplicate_credit_and_error_notices():
    alerts = ContextUsageAlerts()
    credit_state = SimpleNamespace(has_data=True, used_fraction=0.95)
    failed = {
        "last_prompt_tokens": 100,
        "context_length": 1000,
        "credits_state": credit_state,
        "failed": True,
        "error": "429 rate_limit_exceeded secret=never-shown",
    }
    # Existing credits and provider-failure paths own those user-visible notices.
    assert compose_turn_alert(alerts, "topic", failed) is None

    compaction = {"compacted_in_place": True, "compaction_event_id": "attempt-1"}
    notice = compose_turn_alert(alerts, "topic", compaction)
    assert notice is not None
    assert compose_turn_alert(alerts, "topic", compaction) is None
    assert compose_turn_alert(alerts, "topic", {**compaction, "compaction_event_id": "attempt-2"}) == notice

    body, footer = inject_turn_alert("answer", notice, already_sent=False, intentional_silence=False)
    assert body == f"answer\n\n{notice}"
    assert footer is None
    body, footer = inject_turn_alert("streamed answer", notice, already_sent=True, intentional_silence=False)
    assert body == "streamed answer"
    assert footer == notice
    assert inject_turn_alert("answer", notice, already_sent=False, intentional_silence=True) == ("answer", None)


def test_streamed_alert_footer_is_delivered_by_real_gateway_delivery_path():
    sent = []

    class Adapter:
        def _streaming_tts_turn_completed(self, *_args):
            return False

        async def send(self, chat_id, content, **kwargs):
            sent.append((chat_id, content, kwargs))

    adapter = Adapter()
    gateway = object.__new__(GatewayTurnMixin)
    gateway._delivery_adapter_for = lambda _source: adapter
    gateway._should_send_voice_reply = lambda *_args, **_kwargs: False
    gateway._event_thread_metadata = lambda _event, _source: {"thread_id": "thread"}
    event = SimpleNamespace()
    source = SessionSource(platform=Platform.SLACK, chat_id="chat", user_id="user")
    session = SimpleNamespace(session_id="session")
    result = {"already_sent": True, "failed": False, "media_already_delivered": True}

    delivered = asyncio.run(gateway._hmwa_deliver_turn_response(
        event, source, session, "session-key", 1, result, [], "streamed answer", "context notice", False,
    ))

    assert delivered is None
    assert sent == [("chat", "context notice", {"metadata": {"thread_id": "thread"}})]
    assert event._streamed_final_response == "streamed answer"
