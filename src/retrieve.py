import argparse
import json
from functools import lru_cache
from time import perf_counter
import numpy as np
from langsmith import traceable
from .config import RERANKER, corpus, manifest, model_source, load_json
from .indexes import build, collection, encode

# The pinned multilingual reranker returns raw relevance logits.
MIN_RERANK_SCORE = 0.0


@lru_cache(maxsize=4)
def backend_store(version, revision):
    backend = "faiss" if version == "v2" else "chroma"
    path = build(backend, revision)
    if backend == "chroma":
        return collection(path, revision)
    import faiss
    faiss.omp_set_num_threads(2)
    shards = {tuple(map(int, p.stem[1:].split("E"))): faiss.read_index(str(p))
              for p in sorted((path / "faiss").glob("*.faiss"))}
    return shards, load_json(path / "ids.json")


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    source, opts = model_source(RERANKER)
    return CrossEncoder(source, **opts)


@traceable(name="Eligible passage search", run_type="retriever")
def search(question, version, campaign, cutoff, revision):
    rows = corpus(revision)["chunks"]
    k = 20 if version == "v4" else 5
    count = sum(p["campaign_id"] == campaign and p["episode_no"] <= cutoff for p in rows.values())
    if not count:
        return []
    vector = encode([question], "query: ")
    db = backend_store("v2" if version == "v2" else "v3", revision)
    scores = []
    if version == "v2":
        shards, ids = db
        for (shard_campaign, episode), shard in shards.items():
            if shard_campaign == campaign and episode <= cutoff:
                distances, labels = shard.search(vector, min(k, shard.ntotal))
                scores.extend((ids[i], float(s)) for i,s in zip(labels[0], distances[0]) if i >= 0)
    else:
        where = {"$and": [{"campaign_id": {"$eq": campaign}}, {"episode_no": {"$lte": cutoff}}]}
        found = db.query(query_embeddings=vector, where=where, n_results=min(k,count), include=["distances"])
        scores = [(i, 1-d) for i,d in zip(found["ids"][0], found["distances"][0])]
    return [{"id": i, "score": s} for i,s in sorted(scores, key=lambda x:(-x[1], x[0]))[:k]]


@traceable(name="Rerank candidate passages")
def rerank(question, candidates, revision):
    rows = corpus(revision)["chunks"]
    model = reranker()
    # Limit the question separately to leave the full 248-token passage room in a 512-token pair.
    tok = model.tokenizer
    question = tok.decode(tok.encode(question, add_special_tokens=False)[:128], skip_special_tokens=True)
    pairs = [(question, rows[c["id"]]["text"]) for c in candidates]
    scores = np.asarray(model.predict(pairs, batch_size=8, show_progress_bar=False)).reshape(-1)
    ranked = [dict(id=c["id"], score=float(s)) for c,s in zip(candidates,scores)]
    return sorted(ranked, key=lambda r:(-r["score"], r["id"]))[:5]


def retrieve(question, version, campaign, cutoff, revision):
    if not question.strip() or version not in ("v2","v3","v4"):
        raise ValueError("Enter a question and a supported version.")
    start = perf_counter()
    initial = search(question, version, campaign, cutoff, revision)
    split = perf_counter()
    final = rerank(question, initial, revision) if version == "v4" and initial else initial[:5]
    rows = corpus(revision)["chunks"]
    if any(rows[c["id"]]["episode_no"] > cutoff or rows[c["id"]]["campaign_id"] != campaign for c in initial + final):
        raise RuntimeError("Cutoff filter failed.")
    return dict(initial=initial, final=final, retrieval_ms=round((split-start)*1000,2),
                rerank_ms=round((perf_counter()-split)*1000,2) if version=="v4" else 0)


@traceable(name="Check story")
def check_story(question, version, campaign, cutoff, revision, language="English", evidence_only=False):
    result = retrieve(question, version, campaign, cutoff, revision)
    rows = corpus(revision)["chunks"]
    evidence = [rows[c["id"]] for c in result["final"]]
    weak = bool(version == "v4" and evidence and result["final"][0]["score"] < MIN_RERANK_SCORE)
    result["guardrail"] = "low_relevance" if weak else ""
    result["prompt_ids"] = [] if evidence_only or weak else [p["chunk_id"] for p in evidence]
    if weak and not evidence_only:
        message = {"Hindi": "पर्याप्त प्रमाण नहीं मिला।", "Hinglish": "Is sawaal ke liye paryaapt saboot nahi mila."}.get(
            language, "Insufficient evidence. The retrieved dialogue does not closely match this question.")
        result["answer"] = dict(parts=[dict(claim=message, status="unresolved", evidence_ids=[], support_quotes=[])],
                                retries=0, input_tokens=0, output_tokens=0)
        return result
    if not evidence_only and evidence:
        from .answer import generate
        try:
            result["answer"] = generate(question, evidence, campaign, cutoff, language)
        except Exception as err:
            # Preserve the completed retrieval for diagnostics if the provider fails.
            err.retrieval_result = result
            raise
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--version", choices=["v2","v3","v4"], default="v4")
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--campaign", type=int, default=1)
    p.add_argument("--language", default="English")
    p.add_argument("--evidence-only", action="store_true")
    args = p.parse_args()
    revision = manifest()["fingerprint"]
    result = check_story(args.question, args.version, args.campaign, args.cutoff, revision, args.language, args.evidence_only)
    result["passages"] = [corpus(revision)["chunks"][c["id"]] for c in result["final"]]
    print(json.dumps(result, indent=2, ensure_ascii=True))

