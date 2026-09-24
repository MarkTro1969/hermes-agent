"""What a finite one-shot session (``hermes chat -q`` / ``--oneshot``, ``hermes -z``) does NOT do.

A one-shot run has no later session in its HERMES_HOME to learn for: the process answers one query and
exits. The interactive self-improvement loop is pure overhead there, and a measured one — across 21
one-shot benchmark trajectories the agent authored 7 new skills and patched a bundled one mid-task, 37 of
~215 tool calls were ``skill_view``/``skill_manage``, and skill text was 34% of every tool-result byte fed
back into context. Three rules follow, all keyed on the same session marker the approval gate and the
delegation dispatcher already read (``HERMES_SINGLE_QUERY_SESSION``), so interactive sessions are untouched:

* ``skill_manage`` is not offered (``skills_list``/``skill_view`` stay: reading a domain skill can still win);
* the ## Skills prompt drops the "record it / patch it / offer to save" coaching and the "load process skills
  even for tasks you already know" push, keeping only "load a skill when it adds knowledge you lack";
* delegation is capped per session (``delegation.oneshot_max_children``): subagents each re-pay a cold
  system prompt and re-explore the repo, and the observed spawns were mostly "independent review of my own
  work" rather than parallel work.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

ONESHOT_HIDDEN_TOOLS = frozenset({"skill_manage"})


def is_single_query_session() -> bool:
    """The finite ``-q`` marker, read through the session env so gateway-bound sessions never see it."""
    try:
        from gateway.session_context import get_session_env
    except Exception:
        import os
        get_session_env = os.environ.get
    return str(get_session_env("HERMES_SINGLE_QUERY_SESSION", "") or "") == "1"


def prune_oneshot_tools(tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """*tools* minus ``ONESHOT_HIDDEN_TOOLS``; identity when the session is not one-shot."""
    tools = list(tools)
    if not is_single_query_session():
        return tools
    return [t for t in tools if (t.get("function") or {}).get("name") not in ONESHOT_HIDDEN_TOOLS]


ONESHOT_SKILLS_LOAD_GUIDANCE = (
    "## Skills\n"
    "Use the catalog selectively. For a named integration, specialized workflow, or project convention, load "
    "the best matching skill with skill_view(name) before acting. Load at most one initially; load a second only "
    "if the first explicitly lacks the procedure you need. For ordinary questions and standard work you already "
    "know how to do, answer directly instead of loading general process skills. Use skills_list when the names "
    "below are not enough to identify the right skill. Do not create or edit skills: this is a one-shot run with "
    "no later session to reuse them.\n"
)
