"""Read-only Kanban worker liveness projection for the ``/workers`` gateway command."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing
from hermes_cli.kanban_db_dispatch import _STALE_HEARTBEAT_GAP_SECONDS, _worker_alive


@dataclass(frozen=True)
class WorkerSnapshot:
    board: str
    task_id: str
    title: str
    profile: str
    started_at: Optional[int]
    heartbeat_at: Optional[int]
    worker_pid: Optional[int]
    live: bool
    stalling: bool


def _as_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def collect_board_workers(
    conn,
    board: str,
    *,
    now: Optional[int] = None,
    alive_check: Callable[[Optional[int], object], bool] = _worker_alive,
) -> list[WorkerSnapshot]:
    """Return every ``running`` card on one board with verified process liveness."""
    current = int(time.time()) if now is None else int(now)
    rows = conn.execute(
        """
        SELECT id, title, assignee, started_at, worker_pid,
               last_heartbeat_at, worker_started_at
          FROM tasks
         WHERE status = 'running'
         ORDER BY started_at ASC, created_at ASC, id ASC
        """
    ).fetchall()
    snapshots: list[WorkerSnapshot] = []
    for row in rows:
        pid = _as_int(row["worker_pid"])
        started_at = _as_int(row["started_at"])
        heartbeat_at = _as_int(row["last_heartbeat_at"])
        live = bool(alive_check(pid, row["worker_started_at"]))
        progress_at = heartbeat_at if heartbeat_at is not None else started_at
        stalling = bool(
            live
            and progress_at is not None
            and current - progress_at >= _STALE_HEARTBEAT_GAP_SECONDS
        )
        snapshots.append(
            WorkerSnapshot(
                board=board,
                task_id=str(row["id"]),
                title=str(row["title"] or "(untitled)"),
                profile=str(row["assignee"] or "unassigned"),
                started_at=started_at,
                heartbeat_at=heartbeat_at,
                worker_pid=pid,
                live=live,
                stalling=stalling,
            )
        )
    return snapshots


def collect_kanban_workers(*, now: Optional[int] = None) -> tuple[list[WorkerSnapshot], int]:
    """Collect running-card liveness across every non-archived board.

    The second return value is the number of boards that could not be read. Duplicate
    database paths are visited once so ``HERMES_KANBAN_DB`` cannot duplicate workers.
    """
    snapshots: list[WorkerSnapshot] = []
    failures = 0
    seen_paths: set[str] = set()
    for meta in kb.list_boards(include_archived=False):
        board = str(meta.get("slug") or kb.DEFAULT_BOARD)
        try:
            db_path = str(kb.kanban_db_path(board=board).expanduser().resolve())
            if db_path in seen_paths:
                continue
            seen_paths.add(db_path)
            with connect_closing(board=board) as conn:
                snapshots.extend(collect_board_workers(conn, board, now=now))
        except Exception:
            failures += 1
    return snapshots, failures


def _duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h" + (f" {minutes}m" if minutes else "")
    days, hours = divmod(hours, 24)
    return f"{days}d" + (f" {hours}h" if hours else "")


def _worker_line(worker: WorkerSnapshot, now: int) -> str:
    elapsed = _duration(now - worker.started_at) if worker.started_at is not None else "unknown"
    heartbeat = (
        f"heartbeat {_duration(now - worker.heartbeat_at)} ago"
        if worker.heartbeat_at is not None
        else "no heartbeat"
    )
    return (
        f"• **{worker.profile}** — {worker.title}\n"
        f"  `{worker.board}` · `{worker.task_id}` · {elapsed} · {heartbeat}"
    )


def render_kanban_workers(
    workers: Iterable[WorkerSnapshot],
    *,
    board_failures: int = 0,
    now: Optional[int] = None,
    limit: int = 10,
) -> str:
    """Render a compact Telegram-safe worker report."""
    current = int(time.time()) if now is None else int(now)
    items = list(workers)
    working = [item for item in items if item.live and not item.stalling]
    stalling = [item for item in items if item.live and item.stalling]
    stale = [item for item in items if not item.live]

    if working:
        lines = [f"🟢 **Kanban workers: {len(working)} working**"]
    elif stalling:
        lines = [f"🟡 **Kanban workers: 0 working, {len(stalling)} stalling**"]
    else:
        lines = ["**Kanban workers: none working**"]

    shown = 0
    for item in working:
        if shown >= limit:
            break
        lines.append(_worker_line(item, current))
        shown += 1

    if stalling:
        lines.append(f"🟡 **Stalling: {len(stalling)}**")
        for item in stalling[: max(0, limit - shown)]:
            lines.append(_worker_line(item, current))
            shown += 1

    if stale:
        noun = "card" if len(stale) == 1 else "cards"
        lines.append(f"⚠️ **{len(stale)} running {noun} without a live worker**")
        for item in stale[:3]:
            lines.append(f"• `{item.board}` · `{item.task_id}` — {item.title}")

    hidden = max(0, len(working) + len(stalling) - shown)
    if hidden:
        lines.append(f"…and {hidden} more live worker{'s' if hidden != 1 else ''}")
    if board_failures:
        lines.append(f"⚠️ Could not read {board_failures} board{'s' if board_failures != 1 else ''}.")
    return "\n".join(lines)


def format_kanban_workers() -> str:
    workers, failures = collect_kanban_workers()
    return render_kanban_workers(workers, board_failures=failures)
