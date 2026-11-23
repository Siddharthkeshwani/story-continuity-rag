import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from filelock import FileLock
from .models import ROOT, E5, RERANKER, REVISIONS, model_dir

load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
CHAT_MODEL = os.getenv("CHAT_MODEL") or "gemini-3.8-flash"
INDEXES = ROOT / "indexes"
FACTS = DATA / "editorial" / "character_facts.csv"
RAW = DATA / "raw" / "baseline" / "data" / "cleaned data"
BUILD = ROOT / ".build"
BUILD.mkdir(exist_ok=True)
CORPUS_LOCK = FileLock(str(BUILD / "corpus.lock"))


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def manifest():
    with CORPUS_LOCK:
        return load_json(PROCESSED / "manifest.json")


@lru_cache(maxsize=2)
def corpus(revision):
    with CORPUS_LOCK:
        current = manifest()
        if revision != current["fingerprint"]:
            raise RuntimeError("The transcript edition changed. Refresh and repeat the search.")
        tables = {}
        for name in ("chunks", "turns"):
            path = PROCESSED / (name + ".jsonl")
            if sha(path) != current["sha256"][name]:
                raise ValueError(f"Checksum failed for {name}. Rebuild the corpus.")
            with path.open(encoding="utf8") as f:
                rows = [json.loads(line) for line in f]
            key = "chunk_id" if name == "chunks" else "turn_id"
            tables[name] = {row[key]: row for row in rows}
        return tables


def model_source(name):
    directory = model_dir(name)
    if (directory / "model.safetensors").is_file():
        return str(directory), {"local_files_only": True}
    return name, {"revision": REVISIONS[name]}


def chat_settings():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Google Gemini generation requires GEMINI_API_KEY in .env or Space Secrets.")
    return dict(model=CHAT_MODEL, api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def turn_id(ep, n):
    return f"{ep}:T{n:04d}"


def excerpt(passage, turns):
    return [turns[turn_id(passage["episode_id"], i)]
            for i in range(passage["turn_start"], passage["turn_end"] + 1)]

