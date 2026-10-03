"""Transactional local job registry. SQLite is authoritative; JSON is an export."""

from contextlib import contextmanager
import json
import sqlite3
import time
from .config import settings

TERMINAL = {"completed", "failed", "cancelled"}


@contextmanager
def database():
    settings.job_artifacts_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(settings.job_artifacts_dir / ".state.sqlite3", timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=10000")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
        created TEXT NOT NULL, meta TEXT NOT NULL, lease TEXT, expires REAL,
        attempt INTEGER NOT NULL DEFAULT 0);
      CREATE INDEX IF NOT EXISTS jobs_owner_status ON jobs(owner,status);
      CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY, at REAL NOT NULL, job TEXT, name TEXT NOT NULL, value REAL NOT NULL);
    """)
    db.execute("BEGIN IMMEDIATE")
    if "bytes_used" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
        db.execute("ALTER TABLE jobs ADD COLUMN bytes_used INTEGER NOT NULL DEFAULT 0")
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def save(identifier, meta):
    with database() as db:
        db.execute(
            """INSERT INTO jobs(id,owner,status,created,meta) VALUES(?,?,?,?,?)
          ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,status=excluded.status,meta=excluded.meta""",
            (
                identifier,
                meta.get("owner_id", "local"),
                meta.get("status", "queued"),
                meta.get("created_at", ""),
                json.dumps(meta),
            ),
        )


def read(identifier):
    if not (settings.job_artifacts_dir / ".state.sqlite3").exists():
        return None
    with database() as db:
        row = db.execute("SELECT meta FROM jobs WHERE id=?", (identifier,)).fetchone()
        return json.loads(row[0]) if row else None


def list_jobs(owner=None, limit=200):
    with database() as db:
        rows = db.execute(
            "SELECT meta FROM jobs WHERE (? IS NULL OR owner=?) ORDER BY created DESC LIMIT ?",
            (owner, owner, limit),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]


def claim(identifier, token, seconds=60):
    with database() as db:
        # Never steal an expired running job: its native worker could still own files.
        row = db.execute(
            "UPDATE jobs SET lease=?,expires=?,attempt=attempt+1 WHERE id=? AND lease IS NULL AND status='queued'",
            (token, time.time() + seconds, identifier),
        )
        return row.rowcount == 1


def heartbeat(identifier, token, seconds=60):
    with database() as db:
        return (
            db.execute(
                "UPDATE jobs SET expires=? WHERE id=? AND lease=?",
                (time.time() + seconds, identifier, token),
            ).rowcount
            == 1
        )


def release(identifier, token):
    with database() as db:
        db.execute(
            "UPDATE jobs SET lease=NULL,expires=NULL WHERE id=? AND lease=?", (identifier, token)
        )


def recover_interrupted():
    with database() as db:
        # Startup holds the single-dispatcher file lock. Any old lease belongs to
        # the previous dispatcher. Remote renderer locks still drain native work.
        db.execute("UPDATE jobs SET lease=NULL,expires=NULL")


def remove(identifier):
    with database() as db:
        row = db.execute("SELECT status FROM jobs WHERE id=?", (identifier,)).fetchone()
        if row and row[0] not in TERMINAL:
            raise ValueError("job is active")
        db.execute("DELETE FROM jobs WHERE id=?", (identifier,))
        db.execute("DELETE FROM metrics WHERE job=?", (identifier,))


def metric(job, name, value):
    with database() as db:
        db.execute(
            "INSERT INTO metrics(at,job,name,value) VALUES(?,?,?,?)",
            (time.time(), job, name, float(value)),
        )
        db.execute("DELETE FROM metrics WHERE at<?", (time.time() - 30 * 86400,))


def metrics(owner):
    import math

    with database() as db:
        groups = {}
        for row in db.execute(
            "SELECT m.name,m.value FROM metrics m JOIN jobs j ON j.id=m.job WHERE j.owner=? ORDER BY m.value",
            (owner,),
        ):
            groups.setdefault(row["name"], []).append(row["value"])
        return [
            {
                "name": name,
                "count": len(values),
                "mean": sum(values) / len(values),
                "maximum": values[-1],
                "p50": values[math.ceil(len(values) * 0.5) - 1],
                "p95": values[math.ceil(len(values) * 0.95) - 1],
            }
            for name, values in groups.items()
        ]


def set_usage(identifier, size):
    with database() as db:
        db.execute("UPDATE jobs SET bytes_used=? WHERE id=?", (max(0, size), identifier))


def usage(owner):
    with database() as db:
        row = db.execute(
            "SELECT COALESCE(SUM(bytes_used),0), COALESCE(SUM(status NOT IN ('completed','failed','cancelled')),0) FROM jobs WHERE owner=?",
            (owner,),
        ).fetchone()
        return {"bytes": row[0], "active": row[1]}


def active_job_ids():
    with database() as db:
        return [
            row[0]
            for row in db.execute(
                "SELECT id FROM jobs WHERE status NOT IN ('completed','failed','cancelled')"
            )
        ]
