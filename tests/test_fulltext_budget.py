"""Offline checks for claim-driven full-text budget allocation."""

import contextlib
import io
import unittest

import sentences


def paper(title, abstract="An abstract.", pdf=True):
    return {
        "title": title,
        "abstract": abstract,
        "pdf_url": f"https://example.test/{title.replace(' ', '-')}.pdf" if pdf else "",
        "url": f"https://example.test/{title.replace(' ', '-')}",
        "year": 2024,
    }


class FulltextBudgetTest(unittest.TestCase):
    def setUp(self):
        self.old_budget = sentences.FULLTEXT_BUDGET

    def tearDown(self):
        sentences.FULLTEXT_BUDGET = self.old_budget

    def test_auto_budget_scales_with_claims_and_is_bounded(self):
        sentences.FULLTEXT_BUDGET = "auto"
        self.assertEqual(sentences._fulltext_budget_limit(8, 20), 12)
        self.assertEqual(sentences._fulltext_budget_limit(13, 30), 17)
        self.assertEqual(sentences._fulltext_budget_limit(30, 40), 24)

    def test_explicit_budget_values(self):
        sentences.FULLTEXT_BUDGET = "0"
        self.assertEqual(sentences._fulltext_budget_limit(8, 20), 0)
        sentences.FULLTEXT_BUDGET = "-1"
        self.assertEqual(sentences._fulltext_budget_limit(8, 20), 20)
        sentences.FULLTEXT_BUDGET = "3"
        self.assertEqual(sentences._fulltext_budget_limit(8, 20), 3)

    def test_sole_and_uncovered_claims_get_the_first_slots(self):
        papers = [
            paper("rare widget calibration", abstract="", pdf=True),
            paper("multi evidence alpha"),
            paper("multi evidence beta"),
            paper("transformers language models"),
            paper("no address paper", pdf=False),
        ]
        # Paper 1 is in the pool as itself: that must not count as independent
        # coverage for claim 1.
        pool = {
            "rare widget calibration": papers[0],
            "transformers language models": papers[3],
            "independent transformers language models survey": paper(
                "independent transformers language models survey"),
        }
        claims = [
            {"id": 1, "text": "rare widget calibration [1]", "citations": [1]},
            {"id": 2, "text": "transformers language models [4]", "citations": [4]},
            {"id": 3, "text": "multi evidence [2][3]", "citations": [2, 3]},
            {"id": 4, "text": "no address paper [5]", "citations": [5]},
        ]

        sentences.FULLTEXT_BUDGET = "2"
        plan = sentences.plan_fulltext_budget(claims, papers, pool)
        self.assertEqual(plan["considered"], 5)
        self.assertEqual(plan["fetchable"], 4)
        self.assertEqual(plan["no_address"], 1)
        self.assertEqual(plan["selected"], 2)
        self.assertEqual(plan["deferred"], 2)
        # Claim 1: sole citation, no abstract, no independent pool support.
        # Claim 3: two citations, but no pool support. Both outrank claim 2,
        # whose sole paper is independently covered by the pool.
        self.assertEqual(plan["selected_indices"], [1, 2])
        self.assertNotIn(4, plan["selected_indices"])

    def test_three_slots_give_every_claim_one_source(self):
        papers = [paper("p1"), paper("p2"), paper("p3"), paper("p4")]
        claims = [
            {"id": 1, "text": "claim one [1]", "citations": [1]},
            {"id": 2, "text": "claim two [4]", "citations": [4]},
            {"id": 3, "text": "claim three [2][3]", "citations": [2, 3]},
        ]
        sentences.FULLTEXT_BUDGET = "3"
        plan = sentences.plan_fulltext_budget(claims, papers, {})
        self.assertEqual(len(plan["selected_indices"]), 3)
        self.assertTrue({1, 4}.issubset(plan["selected_indices"]))
        self.assertTrue({2, 3} & set(plan["selected_indices"]))

    def test_plan_is_deterministic(self):
        papers = [paper("a"), paper("b"), paper("c")]
        claims = [
            {"id": 1, "text": "one [1]", "citations": [1]},
            {"id": 2, "text": "two [2][3]", "citations": [2, 3]},
        ]
        sentences.FULLTEXT_BUDGET = "2"
        one = sentences.plan_fulltext_budget(claims, papers, {})
        two = sentences.plan_fulltext_budget(claims, papers, {})
        self.assertEqual(one, two)

    def test_deferred_papers_are_not_fetched_during_collection(self):
        papers = [paper("deferred")]
        claims = [{"id": 1, "text": "only [1]", "citations": [1]}]
        seen = []

        def fake_evidence(claim, p, n, allow_fetch=True, in_answer=True):
            seen.append(allow_fetch)
            return None, []

        original = sentences._paper_evidence
        sentences._paper_evidence = fake_evidence
        try:
            sentences._collect_evidence(claims, papers, 1, fulltext_indices=[])
        finally:
            sentences._paper_evidence = original
        self.assertEqual(seen, [False])

    def test_run_prefetches_and_judges_only_selected_papers(self):
        papers = [paper("first"), paper("second")]
        claims = [
            {"id": 1, "text": "first claim [1]", "citations": [1]},
            {"id": 2, "text": "second claim [2]", "citations": [2]},
        ]
        original = (
            sentences.split_report_claims, sentences.retrieve_evidence_many,
            sentences._offer_alternative_urls, sentences.fulltext.prefetch_fulltexts,
            sentences.judge_claims,
        )
        prefetched, judged = [], []
        try:
            sentences.split_report_claims = lambda answer: claims
            sentences.retrieve_evidence_many = lambda cs, pool: {1: 0, 2: 0}
            sentences._offer_alternative_urls = lambda ps: None
            sentences.FULLTEXT_BUDGET = "1"
            sentences.fulltext.prefetch_fulltexts = (
                lambda ps: prefetched.extend(ps) or len(prefetched))

            def fake_judge(cs, index, answer_count=None, fulltext_indices=None):
                judged.extend(fulltext_indices)
                return [], [], [], 0

            sentences.judge_claims = fake_judge
            with contextlib.redirect_stdout(io.StringIO()):
                sentences._run("answer", papers, {})
        finally:
            (sentences.split_report_claims, sentences.retrieve_evidence_many,
             sentences._offer_alternative_urls, sentences.fulltext.prefetch_fulltexts,
             sentences.judge_claims) = original
        self.assertEqual([p["title"] for p in prefetched], ["first"])
        self.assertEqual(judged, [1])


if __name__ == "__main__":
    unittest.main()
