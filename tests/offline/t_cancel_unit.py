"""Phase 0: the cancel registry is per-run, and cleans up after itself."""
import sys, threading, time
import _paths  # noqa: F401
import graph

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + str(extra)) if extra and not cond else ""))
    if not cond:
        fails.append(name)

# ---- a thread with no run registered is never cancelled --------------------
check("no run registered -> no cancel", graph.cancel_requested() is False)
graph._check_cancel()                      # must not raise
check("no run registered -> no raise", True)

# ---- two concurrent runs: stopping one leaves the other alone ---------------
out = {}
started = threading.Barrier(3, timeout=5)
def work(key, hold):
    done = graph.begin_run(key)
    try:
        started.wait()
        t0 = time.time()
        while time.time() - t0 < hold:
            try:
                graph._check_cancel()
            except graph.JobCancelled:
                out[key] = "cancelled"
                return
            time.sleep(0.01)
        out[key] = "completed"
    finally:
        done()

a = threading.Thread(target=work, args=("A", 1.5))
b = threading.Thread(target=work, args=("B", 0.4))
a.start(); b.start()
started.wait()                              # both registered before the stop
graph.request_cancel("A")
a.join(); b.join()
check("stopped run reports cancelled", out.get("A") == "cancelled", out)
check("other run is untouched", out.get("B") == "completed", out)

# ---- the registry does not leak -------------------------------------------
check("registry empty after both runs", graph._cancel_events == {}, graph._cancel_events)

# ---- a reused key starts clean --------------------------------------------
graph.begin_run("K")
graph.request_cancel("K")
check("cancel survives being set", graph.cancel_requested("K") is True)
graph.end_run("K")
graph.begin_run("K")
check("reused key starts uncancelled", graph.cancel_requested("K") is False)
graph.end_run("K")
check("registry empty again", graph._cancel_events == {}, graph._cancel_events)

# ---- a stop asked for before the run starts is still honoured -------------
graph.request_cancel("LATE")
graph.begin_run("LATE")
check("pre-run stop is honoured", graph.cancel_requested("LATE") is True)
graph.end_run("LATE")
check("registry empty at the end", graph._cancel_events == {}, graph._cancel_events)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
