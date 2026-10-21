"""Delivery-only context/allowance/compaction notices for a finished gateway turn; inherited by GatewayTurnMixin.

Truthfulness rules: the context notice reads only a provider-measured occupancy
(``context_usage_fields`` with ``context_estimated`` False); the allowance notice reads only a
``RateLimitState`` that actually carries header data; the compaction confirmation needs the runner's
``compacted_in_place`` AND the agent's real per-attempt ``_compression_attempt_id`` (a session id is not
an event id: keying on it would hide every later in-place compaction). A failed turn neither alerts nor
re-arms. Each condition warns once per session while it holds and re-arms after recovery.

At most one notice per turn (context, then allowance, then compaction). The caller appends it to the
turn's own reply AFTER transcript persistence — delivered, never saved — and never opens a send of its
own: with no carrier this turn, state is left untouched so the notice is not latched undelivered.
"""
from __future__ import annotations

from typing import Any, Optional

CONTEXT_ALERT_THRESHOLD = 80
LOW_PROVIDER_ALLOWANCE_THRESHOLD = 20

COMPACTION_NOTICE = (
    "ℹ️ Conversation history was shortened to preserve context space. Continue normally or start a fresh topic."
)
_BUCKET_LABELS = (
    ("requests_min", "requests/min"),
    ("requests_hour", "requests/hour"),
    ("tokens_min", "tokens/min"),
    ("tokens_hour", "tokens/hour"),
)


def measured_context(agent: Any) -> Optional[tuple[int, int]]:
    """``(used_tokens, context_length)`` from a provider-measured prompt, or None when only an estimate exists."""
    from agent.context_breakdown import context_usage_fields
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return None
    fields = context_usage_fields(compressor)
    if not fields or fields.get("context_estimated") is not False:
        return None
    return fields["context_used"], fields["context_max"]


def measured_allowance(agent: Any) -> Optional[tuple[str, str, float]]:
    """``(provider, label, remaining_pct)`` of the lowest live header window, or None without real data."""
    getter = getattr(agent, "get_rate_limit_state", None)
    state = getter() if callable(getter) else None
    provider = getattr(state, "provider", None)
    if state is None or not getattr(state, "has_data", False) or not isinstance(provider, str) or not provider:
        return None
    lowest: Optional[tuple[str, float]] = None
    for attr, label in _BUCKET_LABELS:
        bucket = getattr(state, attr, None)
        if bucket is None or bucket.limit <= 0:
            continue
        # A window whose reset has already passed no longer describes current usage.
        if bucket.reset_seconds > 0 and bucket.remaining_seconds_now <= 0:
            continue
        percent = max(0.0, bucket.remaining * 100 / bucket.limit)
        if lowest is None or percent < lowest[1]:
            lowest = (label, percent)
    return (provider, *lowest) if lowest else None


class GatewayTurnUsageAlertsMixin:
    def _usage_alert_state(self) -> dict[str, set]:
        state = getattr(self, "_usage_alerts_sent", None)
        if state is None:
            state = self._usage_alerts_sent = {}
        return state

    def _hmwa_usage_alert_text(
        self, agent: Any, alert_session: Optional[str], agent_result: dict, turn_failed: bool,
    ) -> str:
        """One notice for this turn ('' for none). Call only when the reply can carry it."""
        if turn_failed or agent is None or not alert_session:
            return ""
        sent = self._usage_alert_state().setdefault(alert_session, set())
        notice = (
            self._context_notice(agent, sent)
            or self._allowance_notice(agent, sent)
            or self._compaction_notice(agent, agent_result, sent)
        )
        if not sent:
            self._usage_alert_state().pop(alert_session, None)
        return notice

    @staticmethod
    def _context_notice(agent: Any, sent: set) -> str:
        measured = measured_context(agent)
        if measured is None:
            return ""
        used, maximum = measured
        occupancy = used * 100 / maximum
        if occupancy < CONTEXT_ALERT_THRESHOLD:
            sent.discard("context")
            return ""
        if "context" in sent:
            return ""
        sent.add("context")
        return (
            f"⚠️ Estimated context occupancy is {occupancy:.0f}% "
            f"({used:,} of {maximum:,} tokens). Consider starting a fresh topic."
        )

    @staticmethod
    def _allowance_notice(agent: Any, sent: set) -> str:
        measured = measured_allowance(agent)
        if measured is None:
            return ""
        provider, label, percent = measured
        if percent >= LOW_PROVIDER_ALLOWANCE_THRESHOLD:
            sent.difference_update({k for k in sent if isinstance(k, tuple) and k[:2] == ("allowance", provider)})
            return ""
        key = ("allowance", provider, label)
        if key in sent:
            return ""
        sent.add(key)
        return f"⚠️ {provider} API {label} allowance is {percent:.0f}% remaining. Wait for reset or reduce requests."

    @staticmethod
    def _compaction_notice(agent: Any, agent_result: dict, sent: set) -> str:
        if not agent_result.get("compacted_in_place"):
            return ""
        event_id = getattr(agent, "_compression_attempt_id", None)
        if not isinstance(event_id, str) or not event_id:
            return ""
        key = ("compaction", event_id)
        if key in sent:
            return ""
        # Attempt ids are unique uuids, so only the newest needs remembering.
        sent.difference_update({k for k in sent if isinstance(k, tuple) and k[0] == "compaction"})
        sent.add(key)
        return COMPACTION_NOTICE
