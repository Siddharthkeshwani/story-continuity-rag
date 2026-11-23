import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer
from .config import (BUILD, CORPUS_LOCK, DATA, E5, FACTS, PROCESSED, RAW,
                     REVISIONS, load_json, model_dir, sha, turn_id)

BUDGET = 248
OVERLAP = 2


def units_for(turn, tokenizer):
    label = ", ".join(turn["speakers"]) + ": "
    text = label + turn["text"]
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded.ids) <= BUDGET:
        yield turn["turn_no"], text, len(encoded.ids)
        return
    # Split using original character offsets, retaining the original speaker and turn.
    encoded = tokenizer.encode(turn["text"], add_special_tokens=False)
    size = BUDGET - len(tokenizer.encode(label, add_special_tokens=False).ids) - 3
    for offset in range(0, len(encoded.ids), size):
        span = encoded.offsets[offset:offset + size]
        piece = label + turn["text"][span[0][0]:span[-1][1]]
        yield turn["turn_no"], piece, len(tokenizer.encode(piece, add_special_tokens=False).ids)


def windows(units):
    start = 0
    while start < len(units):
        end, tokens = start, 0
        while end < len(units) and tokens + units[end][2] <= BUDGET:
            tokens += units[end][2]
            end += 1
        end = max(end, start + 1)
        yield units[start:end]
        if end == len(units):
            break
        start = max(start + 1, end - OVERLAP)
        while start < end and sum(u[2] for u in units[start:end+1]) > BUDGET:
            start += 1


def write_line(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def episode_key(path):
    return tuple(map(int, path.stem[1:].split("E")))


def build(raw, target=PROCESSED):
    files = sorted((p for p in raw.glob("C*E*.json") if re.fullmatch(r"C\d+E\d+", p.stem)),
                   key=episode_key)
    if not files:
        raise ValueError(f"No episodes selected from {raw}")
    tokenizer_path = model_dir(E5) / "tokenizer.json"
    if not tokenizer_path.exists():
        from huggingface_hub import hf_hub_download
        tokenizer_path = hf_hub_download(E5, "tokenizer.json", revision=REVISIONS[E5])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    with FACTS.open(encoding="utf-8-sig", newline="") as source:
        facts = list(csv.DictReader(source))
    ids, counts, episode_sizes, sources = set(), Counter(), {}, {}
    campaigns = {}
    with tempfile.TemporaryDirectory(dir=BUILD) as staging:
        stage = Path(staging)
        with (stage / "turns.jsonl").open("w", encoding="utf8", newline="\n") as turns, (stage / "chunks.jsonl").open("w", encoding="utf8", newline="\n") as chunks:
            for path in files:
                ep = path.stem
                campaign, number = episode_key(path)
                campaigns.setdefault(str(campaign), []).append(number)
                raw_turns = load_json(path)["TURNS"]
                sources[path.name] = sha(path)
                episode_sizes[ep] = len(raw_turns)
                units = []
                for i, row in enumerate(raw_turns):
                    if int(row.get("NUMBER", i)) != i:
                        raise ValueError(f"Unexpected source numbering at {ep} turn {i}")
                    turn = dict(turn_id=turn_id(ep, i), episode_id=ep, turn_no=i,
                                speakers=row["NAMES"], text=" ".join(row["UTTERANCES"]))
                    write_line(turns, turn)
                    units.extend(units_for(turn, tokenizer))
                base = dict(campaign_id=campaign, episode_id=ep, episode_no=number,
                            source_file="baseline/data/cleaned data/" + path.name, language="en")
                def emit(cid, start, end, kind, text):
                    if cid in ids or not 0 <= start <= end < len(raw_turns):
                        raise ValueError(f"Invalid source record {cid}")
                    ids.add(cid)
                    counts[kind] += 1
                    write_line(chunks, dict(base, chunk_id=cid, turn_start=start, turn_end=end,
                                           document_type=kind, text=text))
                for j, window in enumerate(windows(units)):
                    emit(f"{ep}:P{j:04}", window[0][0], window[-1][0], "dialogue",
                         "\n".join(u[1] for u in window))
                for fact in facts:
                    if int(fact["campaign_id"]) == campaign and int(fact["episode_no"]) == number:
                        emit(fact["fact_id"], int(fact["source_turn_start"]), int(fact["source_turn_end"]),
                             "editorial", fact["fact_text"])
        hashes = {k: sha(stage / (k + ".jsonl")) for k in ("chunks", "turns")}
        meta = dict(episode_ids=list(episode_sizes), episodes=episode_sizes, campaigns=campaigns,
                    gaps={c:sorted(set(range(1,max(nums)+1))-set(nums)) for c,nums in campaigns.items()},
                    document_counts=counts, turn_count=sum(episode_sizes.values()), source_files=sources,
                    tokenizer=E5, tokenizer_revision=REVISIONS[E5], token_budget=BUDGET,
                    overlap_turns_max=OVERLAP, sha256=hashes)
        meta["fingerprint"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        (stage / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
        with CORPUS_LOCK:
            target.mkdir(exist_ok=True, parents=True)
            for file in ("turns.jsonl", "chunks.jsonl", "manifest.json"):
                (stage / file).replace(target / file)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    meta = build(args.raw)
    print(meta["episode_ids"], meta["document_counts"], meta["fingerprint"])

