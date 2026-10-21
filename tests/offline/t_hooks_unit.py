"""Two runs in one process must not share progress/log routing."""
import sys, threading, time
import _paths  # noqa: F401
import graph

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond: fails.append(name)

seen = {"A": [], "B": []}
started = threading.Barrier(2)

def run(tag, exit_after=None):
    done = graph.begin_run(tag)
    graph.install_hooks(lambda t, tag=tag: seen[tag].append(t), None)
    try:
        started.wait(timeout=5)
        for i in range(40):
            graph._emit(f"{tag}-stage-{i}")
            if exit_after and i == exit_after:
                return
            time.sleep(0.001)
    finally:
        graph.clear_hooks()
        done()

a = threading.Thread(target=run, args=("A",))
b = threading.Thread(target=run, args=("B",))
a.start(); b.start(); a.join(); b.join()

check("A saw only its own stages", all(s.startswith("A-") for s in seen["A"]), seen["A"][:3])
check("B saw only its own stages", all(s.startswith("B-") for s in seen["B"]), seen["B"][:3])
check("A got all its stages", len(seen["A"]) == 40, len(seen["A"]))
check("B got all its stages", len(seen["B"]) == 40, len(seen["B"]))
check("hook registry is empty afterwards", not graph._hooks, graph._hooks)
check("cancel registry is empty afterwards",
      not graph._cancel_events, graph._cancel_events)

# one run stopping must not stop the other
graph.begin_run("X"); graph.begin_run("Y")
graph.install_hooks(None, None)
check("X is not cancelled", not graph.cancel_requested("X"))
graph.request_cancel("X")
check("X is cancelled", graph.cancel_requested("X"))
check("Y is untouched", not graph.cancel_requested("Y"))
graph.end_run("X"); graph.end_run("Y")

# the CLI path (no begin_run) still uses the plain global hook
got = []
graph._stage_hook = got.append
graph._emit("cli-line")
check("CLI fallback hook still fires", got == ["cli-line"], got)
graph._stage_hook = None

print()
print("FAILURES: %d" % len(fails))
sys.exit(1 if fails else 0)
