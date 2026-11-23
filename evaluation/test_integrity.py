import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import json
from src.config import manifest, corpus, excerpt, RAW, load_json
from src.retrieve import retrieve, check_story
from src.answer import generate, inputs
from evaluation.run import benchmark, read_csv, write_csv
from src.config import ROOT


class Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = manifest()
        cls.tables = corpus(cls.meta["fingerprint"])

    def test_sources_and_gold(self):
        turns = self.tables["turns"]
        self.assertEqual({int(c):len(episodes) for c, episodes in self.meta["campaigns"].items()}, {1:113,2:46})
        for ep in self.meta["episode_ids"]:
            source = load_json(RAW / (ep+".json"))["TURNS"]
            for n, raw in enumerate(source):
                row = turns[f"{ep}:T{n:04}"]
                self.assertEqual((row["speakers"],row["text"]), (raw["NAMES"]," ".join(raw["UTTERANCES"])))
        for split,n in [("dev",10),("test",34)]:
            cases = read_csv(ROOT/f"evaluation/{split}.csv")
            self.assertEqual(len({r["question_id"] for r in cases}),n)
            for row in cases:
                for span in row["acceptable_evidence_ranges"].replace("|",";").split(";"):
                    if not span:
                        continue
                    e, positions = span.split(":")
                    self.assertLessEqual(int(e),int(row["cutoff"]))
                    for pos in positions.split("-"):
                        self.assertIn(f"C{row['campaign_id']}E{int(e):03}:T{int(pos):04}",turns)

    def test_source_windows_and_empty_cutoff(self):
        for row in self.tables["chunks"].values():
            src = excerpt(row,self.tables["turns"])
            self.assertTrue(src)
            self.assertTrue(all(t["episode_id"]==row["episode_id"] for t in src))
        for version in ("v2","v3","v4"):
            self.assertEqual(retrieve("Kima",version,1,0,self.meta["fingerprint"])["final"],[])

    def test_future_rejected_before_prompt(self):
        with self.assertRaises(ValueError):
            inputs("question",[{"campaign_id":1,"episode_no":2}],1,1,"English")

    def test_invalid_citations_do_not_escape(self):
        from langchain_core.runnables import RunnableLambda
        fake = SimpleNamespace(content=json.dumps({"parts":[{"claim":"Invented detail","status":"established",
                               "evidence_ids":["C1E115:P9999"],"support_quotes":["invented"]}]}),usage_metadata={})
        model = RunnableLambda(lambda _:fake)
        with patch("src.answer.chat",return_value=SimpleNamespace(bind=lambda **kw:model)):
            r = generate("What happened?",[self.tables["chunks"]["F001"]],1,1,"English")
        self.assertEqual(r["retries"],1)
        self.assertEqual(r["parts"][0]["status"],"unresolved")
        self.assertEqual(r["parts"][0]["evidence_ids"],[])

    def test_gate_skips_generation(self):
        ranking = [{"id":"F001","score":-1.0}]
        found = dict(initial=ranking, final=ranking, retrieval_ms=1, rerank_ms=1)
        with patch("src.retrieve.retrieve",return_value=found), patch("src.answer.generate") as model:
            result = check_story("Test", "v4", 1, 1, self.meta["fingerprint"])
        model.assert_not_called()
        self.assertEqual(result["prompt_ids"], [])
        self.assertEqual(result["answer"]["parts"][0]["status"], "unresolved")

    def test_unresolved_claim_cannot_publish_an_uncited_location(self):
        from langchain_core.runnables import RunnableLambda
        fake = SimpleNamespace(content=json.dumps({"parts":[{"claim":"The inn was in an invented city.",
                               "status":"unresolved","evidence_ids":[],"support_quotes":[]}]}),usage_metadata={})
        with patch("src.answer.chat",return_value=SimpleNamespace(bind=lambda **kw:RunnableLambda(lambda _:fake))):
            result = generate("Where was the inn?",[self.tables["chunks"]["F001"]],1,1,"English")
        self.assertNotIn("invented city", result["parts"][0]["claim"])
        self.assertIn("Campaign 1, episode 1", result["parts"][0]["claim"])

    def test_valid_id_with_invented_quote_is_rejected(self):
        from langchain_core.runnables import RunnableLambda
        fake = SimpleNamespace(content=json.dumps({"parts":[{"claim":"Unsupported detail","status":"established",
                               "evidence_ids":["F001"],"support_quotes":["This invented supporting line is absent."]}]}),
                               usage_metadata={})
        with patch("src.answer.chat",return_value=SimpleNamespace(bind=lambda **kw:RunnableLambda(lambda _:fake))):
            result = generate("What happened?",[self.tables["chunks"]["F001"]],1,1,"English")
        self.assertEqual(result["parts"][0]["status"], "unresolved")
        self.assertEqual(result["parts"][0]["support_quotes"], [])

    def test_service_outage_is_separate_from_low_relevance(self):
        with patch("src.answer.generate") as model:
            for score in (-1, 1):
                candidates = [{"id":"F001","score":score}]
                ranking = dict(initial=candidates, final=candidates, retrieval_ms=1, rerank_ms=1)
                with patch("src.retrieve.retrieve",return_value=ranking):
                    result = check_story("Test", "v4", 1, 1, self.meta["fingerprint"], generation_paused=True)
                self.assertEqual(result["prompt_ids"], [])
                if score < 0:
                    self.assertEqual(result["answer"]["parts"][0]["status"], "unresolved")
                else:
                    self.assertEqual(result["generation_skipped"], "ProviderCooldown")
                    self.assertNotIn("answer", result)
            model.assert_not_called()

    def test_rate_limit_retry(self):
        from src.answer import invoke
        from unittest.mock import Mock
        err = RuntimeError("limited")
        err.status_code = 429
        err.response = SimpleNamespace(headers={"retry-after":"7"})
        chain = Mock()
        chain.invoke.side_effect = [err, "ok"]
        with patch("src.answer.sleep") as pause:
            self.assertEqual(invoke(chain, {}), ("ok", 1))
        pause.assert_called_once_with(7)

    def test_failed_generation_preserves_evidence_and_timing(self):
        row = read_csv(ROOT/"evaluation/dev.csv")[0]
        rank = [{"id":"F001", "score":1.0}]
        found = dict(initial=rank, final=rank, retrieval_ms=15.0, rerank_ms=0)
        with tempfile.TemporaryDirectory() as tmp:
            cases, output = Path(tmp)/"case.csv", Path(tmp)/"run.csv"
            write_csv(cases, [row])
            with patch("evaluation.run.retrieve"), patch("src.retrieve.retrieve", return_value=found), \
                 patch("src.answer.generate", side_effect=TimeoutError()), \
                 patch("evaluation.run.chat_settings", return_value={"model":"test"}):
                recorded = benchmark(cases, ["v2"], output)[0]
        self.assertIn("TimeoutError", recorded["error"])
        self.assertEqual(recorded["hit5"], 1)
        self.assertEqual(json.loads(recorded["final"]), rank)
        self.assertEqual(recorded["retrieval_ms"], 15)


if __name__ == "__main__":
    unittest.main()

