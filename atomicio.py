"""Atomic file writes, stdlib only.

Several writers share one directory: every finished run saves a history
snapshot, the retraction cache is rewritten by any run that screened papers,
and the scoring settings are shared by every browser. A plain
``open(path, "w")`` truncates first and writes second, so a crash -- or a
second writer -- in between leaves a half-written JSON file that no reader can
parse and no later write repairs.

So every writer here builds the bytes, writes them to a scratch file in the
target directory, then ``os.replace``s it onto the target. ``os.replace`` is
atomic on POSIX: a reader sees the old file or the new one, never a torn one.

The scratch name comes from ``tempfile.mkstemp`` instead of the fixed
``path + ".tmp"`` this project used before, because two threads saving the
same session must not write into the same scratch file.
"""

import json
import os
import tempfile


def write_bytes(path, data):
    """Write `data` (bytes) to `path`, atomically."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text(path, text, encoding="utf-8"):
    write_bytes(path, text.encode(encoding))


def write_json(path, obj, **kwargs):
    """Write `obj` as JSON. Extra keyword arguments go to ``json.dumps``."""
    kwargs.setdefault("ensure_ascii", False)
    write_bytes(path, json.dumps(obj, **kwargs).encode("utf-8"))
