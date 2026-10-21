"""Durable task table (stdlib sqlite3), shared by every web process.

The web layer used to keep every job in a process-local dict. Two things
followed from that. A restart lost all queued and running work -- ``GET
/status`` answered 404 for anything the current process had not started -- and
because nothing was ever removed from the dict, the process grew with every run
it had ever served, each finished job still holding its log, papers, exports
and answer.

Jobs now live in one SQLite file in WAL mode, which is what makes the queue
survive a restart and (later) lets more than one worker process take jobs from
the same table. What stays in memory is only the job currently running, for a
live log tail; every state change is written through.

Two columns instead of one, because the two halves change at very different
rates: ``payload`` holds the job definition and its result (rewritten a handful
of times per run) while ``progress`` holds the streaming stage/log lines
(rewritten every couple of seconds). Writing the papers and exports out again
on every log line would be megabytes of pointless I/O.
"""

import json
import os
import socket
import sqlite3
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("JOBSTORE_DB") or os.path.join(BASE_DIR, "outputs", "jobs.db")
STALE_AFTER_S = float(os.environ.get("JOBSTORE_STALE_S", "300"))

# the streaming half of a job: cheap to rewrite, and the only part that changes
# while a run is in flight
PROGRESS_KEYS = ("stage", "log", "stages")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    mode       TEXT,
    owner      TEXT,
    worker     TEXT,
    heartbeat_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload    TEXT NOT NULL,
    progress   TEXT NOT NULL DEFAULT '{}'
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created_at);
"""

_conn = None
_lock = threading.RLock()


def _instance_id():
    return f"{socket.gethostname()}:{os.getpid()}"


def configure(path):
    """Point the store at another file. Call it before anything else uses it."""
    global DB_PATH, _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        DB_PATH = path


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _connect():
    """The single connection, opened on first use.

    One connection behind one lock, not one per thread. ThreadingHTTPServer
    starts a thread per request, and a connection opened in a request thread
    outlives that thread -- measured 49 open handles on jobs.db after 300
    requests -- so a per-thread connection leaks file descriptors with every
    request served. Every query here takes microseconds, so serialising them
    costs nothing worth measuring.
    """
    global _conn
    if _conn is None:
        directory = os.path.dirname(os.path.abspath(DB_PATH))
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None,
                               check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        _conn = conn
    return _conn


def init():
    """Create the table and migrate columns added after the first schema."""
    with _lock:
        conn = _connect()
        conn.executescript(_SCHEMA)
        # the owner column arrived after the first tables did, and an index on a
        # column that does not exist yet is an error -- so migrate before index
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "owner" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN owner TEXT")
        if "worker" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN worker TEXT")
        if "heartbeat_at" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT")
        conn.executescript(_INDEXES)


def _split(job):
    """(definition+result, streaming progress) -- private keys stay in memory."""
    progress = {k: job[k] for k in PROGRESS_KEYS if k in job}
    body = {k: v for k, v in job.items()
            if k not in PROGRESS_KEYS and not k.startswith("_")}
    return body, progress


def save(job):
    """Write a job out in full (definition, result and progress)."""
    with _lock:
        _save(job)


def _save(job):
    body, progress = _split(job)
    body.setdefault("created_at", _now())
    jid = str(job["id"])
    now = _now()
    running = str(job.get("status") or "") == "running"
    _connect().execute(
        "INSERT INTO jobs (id, status, mode, owner, worker, heartbeat_at, "
        "                  created_at, updated_at, payload, progress) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET status=excluded.status, mode=excluded.mode, "
        # COALESCE: a caller that saved a job dict without an owner (an old record
        # read back, say) must not blank the ownership out from under it
        "owner=COALESCE(excluded.owner, jobs.owner), "
        "worker=COALESCE(excluded.worker, jobs.worker), "
        "heartbeat_at=CASE WHEN excluded.status='running' "
        "                  THEN COALESCE(excluded.heartbeat_at, jobs.heartbeat_at) "
        "                  ELSE jobs.heartbeat_at END, "
        "updated_at=excluded.updated_at, payload=excluded.payload, "
        "progress=excluded.progress",
        (jid, str(job.get("status") or "queued"), str(job.get("mode") or ""),
         (job.get("owner") or None), (job.get("worker") or None),
         now if running else (job.get("heartbeat_at") or None),
         body["created_at"], now,
         json.dumps(body, ensure_ascii=False),
         json.dumps(progress, ensure_ascii=False)),
    )


def save_progress(jid, job):
    """Write out only the streaming half, for the log tail of a live run."""
    _, progress = _split(job)
    with _lock:
        now = _now()
        _connect().execute(
            "UPDATE jobs SET progress=?, updated_at=?, "
            "heartbeat_at=CASE WHEN status='running' THEN ? ELSE heartbeat_at END "
            "WHERE id=?",
            (json.dumps(progress, ensure_ascii=False), now, now, str(jid)),
        )


def get(jid):
    """One job as a flat dict, or None.

    The payload is a snapshot of the job as it was last saved, so its own
    status field can be a beat out of date -- claiming and settling a job
    update the column. The columns therefore win over whatever the payload
    says, and progress wins over both.
    """
    with _lock:
        row = _connect().execute(
            "SELECT id, status, mode, owner, worker, heartbeat_at, "
            "       created_at, updated_at, payload, progress "
            "FROM jobs WHERE id=?", (str(jid),)).fetchone()
    if row is None:
        return None
    job = json.loads(row["payload"])
    job.update(json.loads(row["progress"] or "{}"))
    job.update({"id": row["id"], "status": row["status"], "created_at": row["created_at"],
                "updated_at": row["updated_at"], "owner": row["owner"] or "",
                "worker": row["worker"] or "",
                "heartbeat_at": row["heartbeat_at"] or ""})
    if row["mode"]:
        job["mode"] = row["mode"]
    return job


def claim_next(instance=None):
    """Take the oldest queued job and mark it running, atomically.

    Returns the job dict, or None when the queue is empty. The transaction is
    what lets several workers share this table later: whoever gets the row is
    the one that runs it.
    """
    instance = instance or _instance_id()
    with _lock:
        conn = _connect()
        conn.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            row = conn.execute("SELECT id FROM jobs WHERE status='queued' "
                               "ORDER BY created_at, id LIMIT 1").fetchone()
            if row is None:
                return None
            jid = row["id"]
            now = _now()
            changed = conn.execute(
                "UPDATE jobs SET status='running', updated_at=?, worker=?, "
                "heartbeat_at=? WHERE id=? AND status='queued'",
                (now, instance, now, jid)).rowcount
            if changed != 1:
                return None
            saved = conn.execute(
                "SELECT payload, progress, worker, heartbeat_at "
                "FROM jobs WHERE id=?", (jid,)).fetchone()
            conn.execute("COMMIT")
            committed = True
        finally:
            # every early return has to unwind the transaction, or the write
            # lock stays held and the next claim (or another process) blocks
            if not committed:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
    job = json.loads(saved["payload"])
    job.update(json.loads(saved["progress"] or "{}"))
    job.update({"status": "running", "started_at": now,
                "worker": saved["worker"] or instance,
                "heartbeat_at": saved["heartbeat_at"] or now,
                "stage": job.get("stage") or "排队中…"})
    return job


def _finish_running(conn, row, note):
    job = json.loads(row["payload"])
    job.update({"status": "error", "stage": "❌ 平台重启中断", "error": note,
                "finished_at": _now()})
    body, progress = _split(job)
    conn.execute(
        "UPDATE jobs SET status='error', updated_at=?, payload=?, progress=? WHERE id=?",
        (_now(), json.dumps(body, ensure_ascii=False),
         json.dumps(progress, ensure_ascii=False), row["id"]))


def _heartbeat_age(heartbeat_at, now=None):
    try:
        then = datetime.fromisoformat(str(heartbeat_at))
    except (TypeError, ValueError):
        return None
    return ((now or datetime.now()) - then).total_seconds()


def _worker_process_alive(worker):
    """True/False for a worker on this host, None when it is another host."""
    try:
        host, raw_pid = str(worker or "").rsplit(":", 1)
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    if host != socket.gethostname() or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _should_recover(row, stale_after, now=None):
    """(settle?, reason) for one running row owned by another instance."""
    worker = str(row["worker"] or "")
    if not worker:
        return True, "升级前的任务没有实例记录（断点未持久化，请重新提问）"
    alive = _worker_process_alive(worker)
    if alive is False:
        return True, "任务所在实例已退出（断点未持久化，请重新提问）"
    if alive is True:
        return False, ""
    age = _heartbeat_age(row["heartbeat_at"], now=now)
    if age is None or age > stale_after:
        return True, "任务所在实例心跳超时（断点未持久化，请重新提问）"
    return False, ""


def recover(instance=None, stale_after=None):
    """Settle only running jobs whose worker is known to be gone.

    A run cannot survive its process: LangGraph checkpoints live in memory. A
    second web instance must not, however, fail the first instance's live
    jobs. Rows carry ``worker`` (host:pid) and ``heartbeat_at``; this routine
    settles ownerless legacy rows, dead same-host processes, and other-host
    workers whose heartbeat is older than ``stale_after``. A fresh heartbeat
    from another host is left untouched. Queued jobs are never touched.
    """
    del instance  # the caller identity is used by claim_next(); kept in API
    stale_after = STALE_AFTER_S if stale_after is None else float(stale_after)
    now = datetime.now()
    with _lock:
        conn = _connect()
        conn.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            rows = conn.execute(
                "SELECT id, payload, worker, heartbeat_at FROM jobs "
                "WHERE status='running'").fetchall()
            victims = []
            for row in rows:
                settle, reason = _should_recover(row, stale_after, now=now)
                if not settle:
                    continue
                _finish_running(conn, row, reason)
                victims.append(row["id"])
            conn.execute("COMMIT")
            committed = True
            return victims
        finally:
            if not committed:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass


def import_history(directory):
    """Adopt the snapshots written before this table existed, so ``/status``
    answers for old runs too. Idempotent: ids already stored are skipped."""
    if not directory or not os.path.isdir(directory):
        return 0
    added = 0
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        jid = name[:-5]
        if get(jid) is not None:
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                snap = json.load(f)
        except Exception:
            continue
        if not isinstance(snap, dict):
            continue
        snap.setdefault("id", jid)
        snap.setdefault("mode", "ask")
        snap.setdefault("status", "done")
        snap.setdefault("created_at", _now())
        save(snap)
        added += 1
    return added


# ------------------------------------------------------- who owns a run

def list_recent(owner, limit=20):
    """Newest runs belonging to `owner`, as light index rows.

    Read with json_extract instead of loading the payloads: the same list built
    from the snapshot files parsed ~500KB per row just to print a title.

    Rows that only exist as somebody's turn are skipped (a follow-up or a
    deepening step records `run_id` pointing at the run it belongs to), because
    their content belongs to that run's own entry.
    """
    with _lock:
        rows = _connect().execute(
            "SELECT id, created_at, mode, status, "
            "       json_extract(payload,'$.question')      AS question, "
            "       json_extract(payload,'$.sort_label')    AS sort_label, "
            "       json_extract(payload,'$.filters_label') AS filters_label, "
            "       json_extract(payload,'$.duration_s')    AS duration_s, "
            "       json_extract(payload,'$.counts')        AS counts, "
            "       json_array_length(payload,'$.papers')   AS paper_count, "
            "       json_extract(payload,'$.run_id')        AS run_id "
            "FROM jobs WHERE owner=? AND status NOT IN ('queued','running') "
            "  AND (run_id IS NULL OR run_id = id) "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (str(owner), int(limit))).fetchall()
    items = []
    for row in rows:
        item = {k: row[k] for k in ("id", "created_at", "mode", "question",
                                    "sort_label", "filters_label", "duration_s",
                                    "paper_count")}
        try:
            item["counts"] = json.loads(row["counts"]) if row["counts"] else None
        except ValueError:
            item["counts"] = None
        items.append(item)
    return items


def adopt_legacy(owner):
    """Hand every ownerless run (the ones that predate this column) to `owner`.

    Runs are ownerless until someone adopts them, and an ownerless run is
    readable by nobody: that is what makes the isolation hold for the history
    written before it existed, instead of leaving it shared.
    """
    with _lock:
        cur = _connect().execute(
            "UPDATE jobs SET owner=?, updated_at=? WHERE owner IS NULL OR owner=''",
            (str(owner), _now()))
        return cur.rowcount


def legacy_count():
    """How many runs nobody owns yet (what /owner offers to adopt)."""
    with _lock:
        row = _connect().execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE owner IS NULL OR owner=''").fetchone()
    return row["n"]


# ------------------------------------------------------------ small queries

def active_chat_run(run_id):
    """Is a follow-up turn for this run already queued or running?"""
    with _lock:
        row = _connect().execute(
            "SELECT 1 FROM jobs WHERE status IN ('queued','running') AND mode='chat' "
            "AND json_extract(payload, '$.run_id')=? LIMIT 1",
            (str(run_id),)).fetchone()
    return row is not None


def queue_depth():
    with _lock:
        row = _connect().execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='queued'").fetchone()
    return row["n"]


def counts():
    with _lock:
        rows = _connect().execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}
