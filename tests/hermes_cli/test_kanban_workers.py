from __future__ import annotations

from contextlib import contextmanager
import sqlite3

from hermes_cli.kanban_workers import (
    collect_board_workers,
    collect_kanban_workers,
    render_kanban_workers,
)


def _db(*rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id TEXT, title TEXT, assignee TEXT, status TEXT,
            started_at INTEGER, created_at INTEGER, worker_pid INTEGER,
            last_heartbeat_at INTEGER, worker_started_at TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return conn


def test_collect_board_workers_reconciles_process_and_heartbeat():
    now = 10_000
    conn = _db(
        ("t_live", "Live build", "default", "running", 9_400, 9_000, 101, 9_990, "fp-101"),
        ("t_stalling", "Wedged build", "sv", "running", 4_000, 3_000, 102, 6_000, "fp-102"),
        ("t_dead", "Dead build", "personal", "running", 9_500, 9_000, 103, 9_990, "fp-103"),
        ("t_done", "Finished", "default", "done", 9_000, 8_000, None, None, None),
    )

    workers = collect_board_workers(
        conn,
        "jarvis-operations",
        now=now,
        alive_check=lambda pid, _fingerprint: pid in {101, 102},
    )

    assert [item.task_id for item in workers] == ["t_stalling", "t_live", "t_dead"]
    by_id = {item.task_id: item for item in workers}
    assert by_id["t_live"].live is True
    assert by_id["t_live"].stalling is False
    assert by_id["t_stalling"].live is True
    assert by_id["t_stalling"].stalling is True
    assert by_id["t_dead"].live is False


def test_render_empty_live_and_stale_states():
    assert render_kanban_workers([], now=10_000) == "**Kanban workers: none working**"

    conn = _db(
        ("t_live", "Live build", "default", "running", 9_400, 9_000, 101, 9_990, "fp"),
        ("t_dead", "Dead build", "sv", "running", 9_000, 8_000, 102, 9_900, "fp"),
    )
    workers = collect_board_workers(
        conn, "ops", now=10_000, alive_check=lambda pid, _fp: pid == 101
    )
    text = render_kanban_workers(workers, now=10_000)
    assert "1 working" in text
    assert "**default** — Live build" in text
    assert "`ops` · `t_live` · 10m · heartbeat 10s ago" in text
    assert "1 running card without a live worker" in text
    assert "t_dead" in text


def test_collect_kanban_workers_reads_multiple_boards(monkeypatch, tmp_path):
    import hermes_cli.kanban_workers as module

    dbs = {
        "alpha": _db(("t_a", "Alpha", "default", "running", 90, 80, 1, 99, "fp")),
        "beta": _db(("t_b", "Beta", "sv", "running", 80, 70, 2, 98, "fp")),
    }
    monkeypatch.setattr(
        module.kb,
        "list_boards",
        lambda include_archived=False: [{"slug": "alpha"}, {"slug": "beta"}],
    )
    monkeypatch.setattr(module.kb, "kanban_db_path", lambda board: tmp_path / f"{board}.db")

    @contextmanager
    def fake_connect(*, board):
        yield dbs[board]

    monkeypatch.setattr(module, "connect_closing", fake_connect)
    monkeypatch.setattr(module, "collect_board_workers", lambda conn, board, now=None: [board])

    workers, failures = collect_kanban_workers(now=100)
    assert workers == ["alpha", "beta"]
    assert failures == 0
