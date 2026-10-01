"""Low-noise, provider-neutral context and usage alert policy.

The gateway supplies only measurements the active agent has actually exposed.  This
module never invents a context window, account balance, or reset time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONTEXT_ALERT_THRESHOLD = 80
LOW_PROVIDER_ALLOWANCE_THRESHOLD = 20


@dataclass
class ContextUsageAlerts:
    """Threshold latches keyed by session and measured provider window."""

    context_alerted: set[str] = field(default_factory=set)
    allowance_alerted: set[tuple[str, str, str]] = field(default_factory=set)
    compaction_alerted: set[tuple[str, str]] = field(default_factory=set)

    def context_alert(self, session_key: str, *, used_tokens: int | None, context_length: int | None) -> str | None:
        """Return one crossing alert; missing telemetry is intentionally silent."""
        if (
            isinstance(used_tokens, bool) or isinstance(context_length, bool)
            or not isinstance(used_tokens, int) or not isinstance(context_length, int)
            or used_tokens < 0 or context_length <= 0
        ):
            return None
        occupancy = used_tokens * 100 / context_length
        if occupancy < CONTEXT_ALERT_THRESHOLD:
            self.context_alerted.discard(session_key)
            return None
        if session_key in self.context_alerted:
            return None
        self.context_alerted.add(session_key)
        return (
            f"⚠️ Estimated context occupancy is {occupancy:.0f}% "
            f"({used_tokens:,} of {context_length:,} tokens). Consider starting a fresh topic."
        )

    def provider_rate_limit_alert(self, session_key: str, state: Any) -> str | None:
        """Alert once on a real provider header window below 20%; reset on recovery."""
        provider = getattr(state, "provider", None)
        if not isinstance(provider, str) or not provider or not bool(getattr(state, "has_data", False)):
            return None
        buckets = (
            ("requests/min", getattr(state, "requests_min", None)),
            ("requests/hour", getattr(state, "requests_hour", None)),
            ("tokens/min", getattr(state, "tokens_min", None)),
            ("tokens/hour", getattr(state, "tokens_hour", None)),
        )
        lowest: tuple[str, float] | None = None
        for label, bucket in buckets:
            limit, remaining = getattr(bucket, "limit", None), getattr(bucket, "remaining", None)
            if isinstance(limit, bool) or isinstance(remaining, bool) or not isinstance(limit, int) or not isinstance(remaining, int) or limit <= 0:
                continue
            percent = max(0.0, remaining * 100 / limit)
            if lowest is None or percent < lowest[1]:
                lowest = (label, percent)
        if lowest is None or lowest[1] >= LOW_PROVIDER_ALLOWANCE_THRESHOLD:
            self.allowance_alerted = {
                key for key in self.allowance_alerted if key[:2] != (session_key, provider)
            }
            return None
        key = (session_key, provider, lowest[0])
        if key in self.allowance_alerted:
            return None
        self.allowance_alerted.add(key)
        return f"⚠️ {provider} API {lowest[0]} allowance is {lowest[1]:.0f}% remaining. Wait for reset or reduce requests."

    def compaction_alert(self, session_key: str, compaction_id: object) -> str | None:
        """Confirm each persisted compaction once, including native/Codex compaction."""
        if not isinstance(compaction_id, str) or not compaction_id:
            # A session id is not an event id: using it would hide every later in-place compaction.
            return None
        key = (session_key, compaction_id)
        if key in self.compaction_alerted:
            return None
        self.compaction_alerted.add(key)
        return "ℹ️ Conversation history was shortened to preserve context space. Continue normally or start a fresh topic."


def compose_turn_alert(alerts: ContextUsageAlerts, session_key: str, result: dict[str, Any]) -> str | None:
    """Choose one supplementary notice from successful, measured turn telemetry.

    Credits and provider failures already emit their own user-facing notices.  Adding
    another footer here would repeat the same warning alongside the response.
    """
    alert = alerts.context_alert(
        session_key,
        used_tokens=result.get("last_prompt_tokens"),
        context_length=result.get("context_length"),
    )
    alert = alert or alerts.provider_rate_limit_alert(session_key, result.get("rate_limit_state"))
    if result.get("compacted_in_place"):
        alert = alert or alerts.compaction_alert(session_key, result.get("compaction_event_id"))
    return alert


def inject_turn_alert(response: str, alert: str | None, *, already_sent: bool, intentional_silence: bool) -> tuple[str, str | None]:
    """Attach a notice to a normal reply or return a trailing footer for streamed replies."""
    if not alert or intentional_silence:
        return response, None
    if already_sent:
        return response, alert
    return (f"{response}\n\n{alert}" if response else alert), None
