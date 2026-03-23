import json
import sqlite3
from typing import Any, Dict, List, Optional

from config import TASK_DB_PATH

DB_PATH = TASK_DB_PATH


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                message TEXT,
                error TEXT,
                summary_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                current_file TEXT,
                done_chunks INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 1,
                finished_files INTEGER NOT NULL DEFAULT 0,
                total_files INTEGER NOT NULL DEFAULT 0,
                request_json TEXT
            )
            """
        )
        conn.commit()


def create_task(task: Dict[str, Any]):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO tasks (
                id, status, progress, message, error, summary_json, created_at, updated_at,
                current_file, done_chunks, total_chunks, finished_files, total_files, request_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task["id"],
                task.get("status", "pending"),
                float(task.get("progress", 0.0)),
                task.get("message"),
                task.get("error"),
                json.dumps(task.get("summary"), ensure_ascii=False)
                if task.get("summary") is not None
                else None,
                task["created_at"],
                task["updated_at"],
                task.get("current_file", ""),
                int(task.get("done_chunks", 0)),
                int(task.get("total_chunks", 1)),
                int(task.get("finished_files", 0)),
                int(task.get("total_files", 0)),
                json.dumps(task.get("request"), ensure_ascii=False)
                if task.get("request") is not None
                else None,
            ),
        )
        conn.commit()


def update_task(task_id: str, fields: Dict[str, Any]):
    if not fields:
        return
    payload = dict(fields)
    if "summary" in payload:
        payload["summary_json"] = (
            json.dumps(payload.pop("summary"), ensure_ascii=False)
            if payload.get("summary") is not None
            else None
        )
    if "request" in payload:
        payload["request_json"] = (
            json.dumps(payload.pop("request"), ensure_ascii=False)
            if payload.get("request") is not None
            else None
        )

    columns = []
    values = []
    for key, value in payload.items():
        columns.append(f"{key} = ?")
        values.append(value)
    values.append(task_id)

    sql = f"UPDATE tasks SET {', '.join(columns)} WHERE id = ?"
    with _connect() as conn:
        conn.execute(sql, values)
        conn.commit()


def _row_to_task(row: sqlite3.Row) -> Dict[str, Any]:
    task = dict(row)
    task["summary"] = json.loads(task["summary_json"]) if task.get("summary_json") else None
    task["request"] = json.loads(task["request_json"]) if task.get("request_json") else None
    task.pop("summary_json", None)
    task.pop("request_json", None)
    return task


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None


def list_tasks(limit: int = 100) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY datetime(created_at) DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_task(r) for r in rows]
