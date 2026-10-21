"""Survey claim splitting, the "no source text" verdict and the survey rules -- offline."""
import glob, json, re, sys
from _paths import FIXTURES
import sentences, verify, web

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + str(extra)))
    if not cond: fails.append(name)

SNAP = FIXTURES / "history" / "20261015_154020_081495.json"
snap = json.load(open(SNAP, encoding="utf-8"))
claim4 = [c for c in snap["audits"]["claims"]["verdicts"] if c["id"] == 4][0]

# ---- 1. "we never got the text" is not "unsupported" ------------------------
# the saved verdict verbatim: judged 不支持 while its only evidence said the
# PDF download failed
badge, label, note = sentences._support_badge(
    False, [], claim4["evidence"], )
check("a failed fetch makes it 无法核对, not 不支持", badge == "unfetched", (badge, label))
check("the reason travels with it", "download failed" in note, note)
check("the label says what it means", "无法核对" in label, label)

ok = [{"n": 5, "source": "abstract", "note": "abstract only"}]
check("a plain abstract verdict stays 不支持",
      sentences._support_badge(False, [], ok)[0] == "unsupported")
check("no evidence at all stays 不支持",
      sentences._support_badge(False, [], [])[0] == "unsupported")
check("a paper with no open full text stays 不支持",
      sentences._support_badge(False, [], [{"n": 5, "source": "abstract",
                                           "note": "no full-text URL"}])[0] == "unsupported")
check("a supported claim with a failed fetch is unaffected",
      sentences._support_badge(True, [5], [{"n": 5, "source": "abstract",
                                            "note": "download failed: x"}])[0] == "abstract")
check("every failure note fulltext.get_fulltext can produce is recognised",
      all(sentences.fetch_failed(n) for n in
          ["download failed: HTTP Error 403: Forbidden", "text extraction timed out after 20s",
           "text extraction failed (12 chars)", "extraction busy",
           "full-text stage deadline reached", "skipped, unavailable in a recent run: x"]),
      [n for n in ["download failed", "timed out", "extraction busy", "deadline reached",
                   "unavailable in a recent run"] if not sentences.fetch_failed(n)])

# ---- 2. the report is split per section ------------------------------------
REPORT = (
    "## 摘要\n本次覆盖 3 个方向 [1]。\n\n"
    "## 方向甲（alpha direction）\n方法 A 报告了结果 [1]。它优于 B [2]。还有第三句 [3]。\n\n"
    "## 方向乙（beta direction）\n方法 C 的结论 [4]。\n\n"
    "## 方向丙（gamma direction）\n这一节还没有写引用。\n\n"
    "## 争议与空白\n甲与乙互相矛盾 [1][4]。\n\n"
    "## 参考文献\n[1] Paper one (2020)\n[2] Paper two (2021)\n"
)
sent_llm_calls = []
def fake_chat(messages, **kw):
    sent_llm_calls.append({"system": messages[0]["content"], "user": messages[1]["content"]})
    cap = int(re.search(r"at most (\d+) claims", messages[0]["content"]).group(1))
    body = messages[1]["content"]
    claims = [{"id": i + 1, "text": f"{body[:8]}#{i + 1} [1]", "citations": [1]}
              for i in range(cap)]
    # the model also offers statements it lifted without a citation; those are
    # not checkable and must not become 不支持 noise
    claims.append({"id": cap + 1, "text": "本次检索未覆盖湿实验验证。", "citations": []})
    claims.append({"id": cap + 2, "text": "报告聚焦一个核心问题", "citations": []})
    return json.dumps({"claims": claims})
real_chat = sentences.llm.chat
sentences.llm.chat = fake_chat
claims = sentences.split_report_claims(REPORT)
sentences.llm.chat = real_chat

titles = [t for t, b in verify.split_sections(REPORT) if t]
check("one splitter call per citable section (6 sections - 方向丙 - 参考文献)",
      len(sent_llm_calls) == 4, (len(sent_llm_calls), titles))
check("the bibliography is never sent to the splitter",
      not any("Paper one" in c["user"] for c in sent_llm_calls))
check("a section with no [n] is not split", not any("还没有写引用" in c["user"] for c in sent_llm_calls))
check("caps add up to at most MAX_CLAIMS_TOTAL", len(claims) <= sentences.MAX_CLAIMS_TOTAL, len(claims))
check("every section with citations is represented",
      sorted({c["section"] for c in claims}) == sorted([t for t in titles if t != "方向丙（gamma direction）"
                                                        and t != "参考文献"]),
      sorted({c["section"] for c in claims}))
check("4 sections x 4 claims = the whole per-section budget", len(claims) == 16, len(claims))
check("ids are unique and sequential", [c["id"] for c in claims] == list(range(1, len(claims) + 1)),
      [c["id"] for c in claims])
check("claims keep their section title", all(c.get("section") for c in claims))
check("statements the model lifted without a citation are dropped",
      all(re.search(r"\[\d+\]", c["text"]) for c in claims), [c["text"] for c in claims])
check("the prompt is told the per-section cap",
      all("at most 4 claims" in c["system"] for c in sent_llm_calls))

# long report: 9 sections must not blow the total ceiling
LONG = "".join(f"## 方向{s}（direction {s}）\n结论 [1]。\n\n" for s in "ABCDEFGHI")
sentences.llm.chat = fake_chat
long_claims = sentences.split_report_claims(LONG)
sentences.llm.chat = real_chat
check("the total ceiling holds however many sections there are",
      len(long_claims) <= sentences.MAX_CLAIMS_TOTAL, len(long_claims))
check("and every section still gets its turn",
      len({c["section"] for c in long_claims}) == 9, len({c["section"] for c in long_claims}))

# ---- 3. a sectionless answer keeps the flat pass ---------------------------
sentences.llm.chat = fake_chat
flat = sentences.split_report_claims("这是一段没有小标题的回答 [1]，还有一句 [2]。")
sentences.llm.chat = real_chat
check("no headings -> the old flat cap", len(flat) == sentences.MAX_CLAIMS, len(flat))
check("flat claims carry no section", all(c.get("section") is None for c in flat))

# ---- 4. a splitter that will not answer falls back locally -----------------
def dead_chat(messages, **kw):
    raise RuntimeError("HTTP Error 405: Method Not Allowed")
sentences.llm.chat = dead_chat
sentences.time.sleep = lambda s: None
try:
    rescued = sentences.split_report_claims(REPORT)
finally:
    sentences.llm.chat = real_chat
check("a dead splitter still yields claims", len(rescued) > 0, len(rescued))
check("the fallback is the section's own cited sentences",
      all(re.search(r"\[\d+\]", c["text"]) for c in rescued), [c["text"] for c in rescued])
check("the fallback keeps the citation numbers",
      all(c["citations"] for c in rescued))
check("the fallback respects the per-section cap", len(rescued) == 4, len(rescued))

# ---- 5. the counters and the UI read the new badge -------------------------
counts = web._count_claims(snap["audits"]["claims"])
check("the saved run counts one unchecked claim, not one failure",
      counts["unfetched"] == 1 and counts["unsupported"] == 0, counts)
check("the total is the claim count", counts["total"] == 8, counts.get("total"))
check("a supported-but-unread claim is not an unfetched one",
      web._claim_badge({"supported": True, "evidence": [{"note": "download failed"}]}) == "unknown")
check("the badge survives a round trip through the UI table",
      all(k in web.BADGE_KEYS for k in
          ("cross", "fulltext", "abstract", "unknown", "unsupported", "unfetched")))

# ---- 6. survey rules: quiet on the real reports, loud when they drift ------
surveys = [json.load(open(f, encoding="utf-8")) for f in
           sorted(glob.glob(str(FIXTURES / "history" / "*.json")))]
surveys = [d for d in surveys if d.get("mode") == "survey"]
check("ten saved surveys are analysed", len(surveys) == 10, len(surveys))
check("no rule fires on any saved survey",
      all(not verify.survey_items(d["answer"], d.get("directions") or []) for d in surveys),
      [(d["id"], verify.survey_items(d["answer"], d.get("directions") or [])) for d in surveys][:2])
big = [d for d in surveys if len(d.get("directions") or []) > 3][0]
dropped = "## 等变图神经网络折叠（equivariant graph neural networks for RNA）"
check("the sample report really has that section", dropped in big["answer"])
cut = big["answer"].replace(dropped, "## 无关小节（unrelated）")
items = verify.survey_items(cut, big["directions"])
check("a renamed direction is reported as missing",
      any("报告缺少方向" in it["message"] for it in items), items)
check("and the unknown section is named",
      any("目录外的小节" in it["message"] for it in items), items)
empty = re.sub(r"\[\d+\]", "", big["answer"])
check("a direction section with no citation is flagged",
      any("没有任何 [n] 引用" in it["message"] for it in verify.survey_items(empty, big["directions"])),
      verify.survey_items(empty, big["directions"])[:3])
check("ask mode passes no directions and gets no survey rules",
      verify.survey_items(big["answer"], []) == [])

print()
print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
