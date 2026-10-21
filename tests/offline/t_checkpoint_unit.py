"""Checkpoint pruning keeps recent threads and protects live runs."""

import threading

import _paths  # noqa: F401
import graph

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


class FakeSerde:
    def loads_typed(self, pair):
        return pair


class FakeSaver:
    def __init__(self, storage):
        self.storage = storage
        self.serde = FakeSerde()
        self.deleted = []

    def delete_thread(self, thread_id):
        self.deleted.append(thread_id)
        self.storage.pop(thread_id, None)


def entry(ts):
    return {"ts": ts}


def storage():
    return {
        "old": {"": {"cp": (entry("2020-01-01T00:00:00+00:00"), None, None)}},
        "mid": {"": {"cp": (entry("2021-01-01T00:00:00+00:00"), None, None)}},
        "new": {"": {"cp": (entry("2022-01-01T00:00:00+00:00"), None, None)}},
    }


old_checkpointer = graph._checkpointer
old_enabled = graph.CHECKPOINT_ENABLED
old_used = graph._checkpoint_used
old_cancel = graph._cancel_events
old_hooks = graph._hooks
try:
    graph.CHECKPOINT_ENABLED = True
    graph._checkpoint_used = {}
    graph._cancel_events = {}
    graph._hooks = {}

    fake = FakeSaver(storage())
    graph._checkpointer = fake
    dropped = graph.prune_checkpoints(keep=2)
    check("prune drops the oldest thread", dropped == 1 and fake.deleted == ["old"],
          (dropped, fake.deleted))
    check("the newest threads remain", set(fake.storage) == {"mid", "new"}, fake.storage)

    fake = FakeSaver(storage())
    graph._checkpointer = fake
    graph._cancel_events = {"old": threading.Event()}
    dropped = graph.prune_checkpoints(keep=1)
    check("an active old thread is never dropped", "old" in fake.storage, fake.storage)
    check("a non-active thread is still evicted", "mid" in fake.deleted, fake.deleted)
    check("prune keeps the newest plus the active run", set(fake.storage) == {"old", "new"},
          fake.storage)
finally:
    graph._checkpointer = old_checkpointer
    graph.CHECKPOINT_ENABLED = old_enabled
    graph._checkpoint_used = old_used
    graph._cancel_events = old_cancel
    graph._hooks = old_hooks


print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all checks passed")
