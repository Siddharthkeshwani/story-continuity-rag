"""Evaluate fixed question sets. Reference answers never enter the search or chat prompt."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, sleep
from uuid import uuid4
import numpy as np
from src.config import ROOT, corpus, manifest, sha, chat_settings
from src.retrieve import check_story, retrieve
from src.answer import PROMPT_VERSION

RESULTS = ROOT / "results"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def matches(passage, span, campaign):
    ep, rng = span.split(":")
    ends = [int(n) for n in rng.split("-")]
    return (passage["campaign_id"] == campaign and passage["episode_no"] == int(ep) and passage["turn_start"] <= ends[0]
            and passage["turn_end"] >= ends[-1])


def benchmark(cases_path, versions, out, evidence_only=False):
    cases = read_csv(cases_path)
    model_id = "evidence-only" if evidence_only else chat_settings()["model"]
    meta = manifest()
    rev = meta["fingerprint"]
    chunks = corpus(rev)["chunks"]
    rows = []
    traces_on = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    digest = sha(cases_path)
    for version in versions:
        start = perf_counter()
        # Load search components with an independent warmup question.
        retrieve("Which passages describe Kraghammer?", version, 1, 1, rev)
        cold_ms = round((perf_counter()-start)*1000, 1)
        for case in cases:
            if rows and not evidence_only:
                sleep(7)
            run_id = str(uuid4())
            cutoff, campaign = int(case["cutoff"]), int(case["campaign_id"])
            row = dict(**case, version=version, corpus=rev, cases_sha256=digest,
                       chat_model=model_id, guardrail="",
                       prompt_version=PROMPT_VERSION, run_id=run_id, trace_id=run_id if traces_on else "",
                       warmup_ms=cold_ms, retrieval_ms="", rerank_ms="", elapsed_ms="", initial="[]",
                       final="[]", prompt_ids="[]", response="[]", citations="[]", input_tokens=0,
                       output_tokens=0, retries=0, hit5="", complete="", source_leak=0,
                       future_text_flag=0, error="", review_key="")
            start = perf_counter()
            try:
                result = check_story(case["question"], version, campaign, cutoff, rev, case["language"], evidence_only,
                                     langsmith_extra={"run_id": run_id})
                answer = result.get("answer", {})
                row["guardrail"] = result.get("guardrail", "")
                parts = answer.get("parts", [])
                cited = [cid for p in parts for cid in p["evidence_ids"]]
                for name in ("initial","final","prompt_ids"):
                    row[name] = json.dumps(result[name], ensure_ascii=False)
                row.update(response=json.dumps(parts, ensure_ascii=False), citations=json.dumps(cited),
                           retrieval_ms=result["retrieval_ms"], rerank_ms=result["rerank_ms"])
                for key in ("input_tokens","output_tokens","retries"):
                    row[key] = answer.get(key, 0)
                examined = [c["id"] for c in result["initial"]] + result["prompt_ids"] + cited
                row["source_leak"] = int(any(chunks[i]["episode_no"] > cutoff or chunks[i]["campaign_id"] != campaign for i in examined))
                forbidden = case["forbidden_future_fact"].casefold()
                row["future_text_flag"] = int(bool(forbidden and forbidden in " ".join(p["claim"] for p in parts).casefold()))
            except Exception as err:
                row["error"] = type(err).__name__ + " " + str(getattr(err, "code", "") or "")
                retrieved = getattr(err, "retrieval_result", {})
                for key in ("initial", "final", "prompt_ids"):
                    row[key] = json.dumps(retrieved.get(key, []))
                for key in ("retrieval_ms", "rerank_ms"):
                    row[key] = retrieved.get(key, "")
                examined = [item["id"] for item in retrieved.get("initial", [])] + retrieved.get("prompt_ids", [])
                row["source_leak"] = int(any(chunks[i]["episode_no"] > cutoff or chunks[i]["campaign_id"] != campaign for i in examined))
            target = case["acceptable_evidence_ranges"]
            if target:
                final = [chunks[c["id"]] for c in json.loads(row["final"])]
                hits = [any(matches(p, s, campaign) for p in final for s in alternatives.split(";"))
                        for alternatives in target.split("|")]
                row["hit5"] = int(any(hits))
                if len(hits) > 1:
                    row["complete"] = int(all(hits))
            row["elapsed_ms"] = round((perf_counter()-start)*1000,2)
            identity = [rev, digest, model_id, PROMPT_VERSION, version, case["question_id"], row["response"], row["final"]]
            row["review_key"] = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            rows.append(row)
            write_csv(out, rows)
            print(version, case["question_id"], row["elapsed_ms"], row["error"][:100], flush=True)
    return rows


def summarize(path):
    rows = read_csv(path)
    review_file = path.with_name(path.stem + "_reviews.csv")
    reviews = {r["review_key"]:r for r in read_csv(review_file)} if review_file.exists() else {}
    summaries = []
    for version in sorted({r["version"] for r in rows}):
        group = [r for r in rows if r["version"] == version]
        reviewed = [reviews[r["review_key"]] for r in group if r["review_key"] in reviews]
        times = [float(r["elapsed_ms"]) for r in group]
        def rate(name):
            valid = [int(r[name]) for r in group if r[name] != ""]
            return f"{sum(valid)}/{len(valid)} ({np.mean(valid)*100:.0f}%)" if valid else "n/a"
        claims = sum(int(r["factual_claims"]) for r in reviewed)
        supported = sum(int(r["supported_claims"]) for r in reviewed)
        summaries.append(dict(version=version, cases=len(group), reviewed=len(reviewed),
            accuracy=(f"{sum(int(r['correct']) for r in reviewed)}/{len(group)}" if len(reviewed)==len(group) else "unreviewed"),
            hit5=rate("hit5"), complete_multi=rate("complete"),
            citation_support=f"{supported}/{claims}" if claims else "n/a",
            source_leaks=sum(int(r["source_leak"]) for r in group),
            text_leaks=sum(int(r["text_leak"]) for r in reviewed) if len(reviewed)==len(group) else "unreviewed",
            warm_median_ms=round(float(np.median(times)),1), warm_p95_ms=round(float(np.percentile(times,95)),1),
            retrieval_median_ms=round(float(np.median([float(r["retrieval_ms"]) for r in group if r["retrieval_ms"]] or [0])),1),
            rerank_median_ms=round(float(np.median([float(r["rerank_ms"]) for r in group if r["rerank_ms"]] or [0])),1),
            warmup_ms=group[0]["warmup_ms"], input_tokens=sum(int(r["input_tokens"]) for r in group),
            output_tokens=sum(int(r["output_tokens"]) for r in group), retries=sum(int(r["retries"]) for r in group),
            failures=sum(bool(r["error"]) for r in group)))
    target = path.with_name("comparison.csv" if path.stem=="runs" else path.stem+"_comparison.csv")
    write_csv(target, summaries)
    print(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, default=ROOT/"evaluation/test.csv")
    p.add_argument("--versions", nargs="+", default=["v2","v3","v4"], choices=["v2","v3","v4"])
    p.add_argument("--summarize", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--evidence-only", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize(a.summarize)
    else:
        path = a.output or RESULTS / ("runs.csv" if a.cases.stem=="test" else a.cases.stem+"_runs.csv")
        benchmark(a.cases, a.versions, path, a.evidence_only)
        summarize(path)

