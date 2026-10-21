"""A missing judge verdict is not a negative verdict -- offline."""
import json

import _paths  # noqa: F401
import llm
import sentences
import verify
import web

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


PAPERS = [{"title": "Paper one", "year": 2020, "abstract": "An abstract.",
           "pdf_url": ""}]
CLAIMS = [
    {"id": 1, "text": "claim one [1]", "citations": [1]},
    {"id": 2, "text": "claim two [1]", "citations": [1]},
]

_real_collect = sentences._collect_evidence
_real_chat = sentences.llm.chat
_real_verify_chat = verify.llm.chat
sentences._collect_evidence = lambda *a, **k: {1: ([], []), 2: ([], [])}


def run_judge(reply):
    sentences.llm.chat = reply
    try:
        return sentences.judge_claims(CLAIMS, PAPERS)
    finally:
        sentences.llm.chat = _real_chat


try:
    # 1) the call failed: every claim is unjudged, never unsupported
    lines, structured, failures, _ = run_judge(
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("judge down")))
    check("a failed judge call is recorded", bool(failures), failures)
    check("every claim is unjudged", all(c["judged"] is False for c in structured), structured)
    check("none of them is unsupported",
          all(c["badge"] == "unjudged" for c in structured), structured)
    check("they are not counted as supported", all(not c["supported"] for c in structured))
    check("the log warns instead of calling them unsupported",
          all(l[0] == "warn" for l in lines if l[1].startswith("Claim ")), lines)

    # 2) a verdict without a supported field is unjudged, not false
    _, structured, _, _ = run_judge(
        lambda *a, **k: '{"verdicts":[{"id":1,"reason":"no field"},'
                        '{"id":2,"supported":false,"reason":"not covered"}]}')
    check("a verdict missing supported is unjudged",
          structured[0]["judged"] is False and structured[0]["badge"] == "unjudged",
          structured[0])
    check("a real false verdict stays unsupported",
          structured[1]["judged"] is True and structured[1]["badge"] == "unsupported",
          structured[1])

    # 3) a complete verdict is judged normally
    _, structured, _, _ = run_judge(
        lambda *a, **k: '{"verdicts":[{"id":1,"supported":true,"supporting":[1],"reason":"yes"},'
                        '{"id":2,"supported":false,"reason":"no"}]}')
    check("a supported verdict keeps judged=true",
          structured[0]["judged"] is True and structured[0]["supported"] is True,
          structured[0])
    check("a false verdict keeps judged=true",
          structured[1]["judged"] is True and structured[1]["supported"] is False,
          structured[1])
finally:
    sentences._collect_evidence = _real_collect
    sentences.llm.chat = _real_chat

# ---- parser fallback -------------------------------------------------------
parsed = llm.extract_verdicts('{"verdicts": [{"n": 1, "reason": "no field"}]}')
check("the tolerant parser leaves a missing supported field out",
      parsed and "supported" not in parsed[0], parsed)
parsed = llm.extract_verdicts('"id": 1, "reason": "x" (no supported key)', id_field="id")
check("the regex fallback also leaves it out",
      parsed and "supported" not in parsed[0], parsed)

# ---- the citation-level LLM audit -----------------------------------------
verify.llm.chat = lambda *a, **k: '{"verdicts":[{"n":1,"reason":"no field"}]}'
try:
    audit = verify.llm_audit("A sentence [1].", PAPERS)
finally:
    verify.llm.chat = _real_verify_chat
check("the citation audit marks the missing verdict",
      audit["verdicts"][0]["judged"] is False and audit["verdicts"][0]["supported"] is False,
      audit["verdicts"])
check("its line is a warning, not a failure",
      any(status == "warn" and "UNJUDGED" in msg for status, msg in audit["lines"]),
      audit["lines"])

# ---- counts and badges -----------------------------------------------------
claim = {"supported": False, "judged": False, "badge": "unjudged",
         "badge_label": "无法核对（判决未完成）", "evidence": []}
check("the UI badge stays unjudged", web._claim_badge(claim) == "unjudged")
counts = web._count_claims({"verdicts": [claim, {"supported": False, "judged": True,
                                                "badge": "unsupported", "evidence": []}]})
check("the count separates unjudged from unsupported",
      counts["unjudged"] == 1 and counts["unsupported"] == 1, counts)
check("the total still counts both", counts["total"] == 2, counts)

print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all checks passed")
