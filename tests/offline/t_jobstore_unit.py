"""Job lease ownership: heartbeat recovery must not kill another instance."""

import os
import socket
import sqlite3
import tempfile
from datetime import datetime, timedelta

import _paths  # noqa: F401
import jobstore

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


old_db = jobstore.DB_PATH
with tempfile.TemporaryDirectory() as td:
    jobstore.configure(os.path.join(td, "jobs.db"))
    try:
        jobstore.init()

        def run_job(jid, worker, heartbeat=None):
            job = {"id": jid, "status": "running", "mode": "ask",
                   "created_at": "2026-10-21T10:00:00", "worker": worker,
                   "heartbeat_at": heartbeat or datetime.now().isoformat(timespec="seconds")}
            jobstore.save(job)
            return job

        now = datetime.now()
        run_job("fresh_other", "other-host:123")
        run_job("dead_local", f"{socket.gethostname()}:99999999")
        run_job("stale_other", "other-host:456")
        stale_at = (now - timedelta(seconds=600)).isoformat(timespec="seconds")
        with jobstore._lock:
            jobstore._connect().execute(
                "UPDATE jobs SET heartbeat_at=? WHERE id='stale_other'", (stale_at,))
        run_job("legacy", None)
        jobstore.save({"id": "queued", "status": "queued", "mode": "ask",
                       "created_at": "2026-10-21T10:00:00"})

        settled = set(jobstore.recover(stale_after=300))
        check("a fresh other-host run is left alone", "fresh_other" not in settled, settled)
        check("a dead same-host process is settled", "dead_local" in settled, settled)
        check("an old other-host heartbeat is settled", "stale_other" in settled, settled)
        check("an ownerless legacy running row is settled", "legacy" in settled, settled)
        check("a queued job is never settled", "queued" not in settled, settled)
        check("the fresh run is still running",
              jobstore.get("fresh_other")["status"] == "running")
        check("the dead run is marked error",
              jobstore.get("dead_local")["status"] == "error")
        check("the queued job is untouched",
              jobstore.get("queued")["status"] == "queued")

        claimed = jobstore.claim_next("this-host:777")
        check("claim_next returns the queued job", claimed and claimed["id"] == "queued",
              claimed)
        check("claim_next records its worker", claimed.get("worker") == "this-host:777", claimed)
        check("claim_next starts a heartbeat", bool(claimed.get("heartbeat_at")), claimed)
        saved = jobstore.get("queued")
        check("the worker/heartbeat persist in the table",
              saved["worker"] == "this-host:777" and bool(saved["heartbeat_at"]), saved)

        # A database created by the previous schema must be migrated in place.
        old_path = os.path.join(td, "old.db")
        raw = sqlite3.connect(old_path)
        raw.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, "
                    "mode TEXT, owner TEXT, created_at TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL, payload TEXT NOT NULL, "
                    "progress TEXT NOT NULL DEFAULT '{}')")
        raw.commit()
        raw.close()
        jobstore.configure(old_path)
        jobstore.init()
        with jobstore._lock:
            columns = {row["name"] for row in
                       jobstore._connect().execute("PRAGMA table_info(jobs)")}
        check("init adds worker/heartbeat to an old table",
              {"owner", "worker", "heartbeat_at"} <= columns, columns)
    finally:
        jobstore.configure(old_db)


print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all checks passed")
