import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from filelock import FileLock
from .config import (DATA, INDEXES, E5, REVISIONS, corpus, load_json, manifest,
                     model_source, sha)


@lru_cache(maxsize=1)
def encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(int(os.getenv("STORY_CPU_THREADS", "6")))
    source, opts = model_source(E5)
    return SentenceTransformer(source, **opts)


def encode(texts, prefix):
    return encoder().encode([prefix + s for s in texts], normalize_embeddings=True,
                            batch_size=24, show_progress_bar=len(texts) > 100,
                            convert_to_numpy=True).astype("float32")


def location(revision):
    return INDEXES / revision


def collection(folder, revision, create=False):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(folder / "chroma"), settings=Settings(anonymized_telemetry=False))
    name = "dialogue_" + revision[:20]
    if create:
        return client.get_or_create_collection(name=name, embedding_function=None,
                                               configuration={"hnsw": {"space": "cosine", "num_threads": 2}})
    return client.get_collection(name=name, embedding_function=None)


def build(backend, revision=None):
    revision = revision or manifest()["fingerprint"]
    rows = list(corpus(revision)["chunks"].values())
    ids = [r["chunk_id"] for r in rows]
    folder = location(revision)
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "build.lock")):
        meta_path = folder / "manifest.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        if not meta:
            bundle = DATA / "bundle" / revision
            with tempfile.TemporaryDirectory(dir=folder) as temp:
                stage = Path(temp)
                if (bundle / "vectors.npy").exists():
                    import shutil
                    spec = load_json(bundle / "manifest.json")
                    if spec["fingerprint"] != revision or sha(bundle / "vectors.npy") != spec["vectors_sha256"]:
                        raise ValueError("Packaged vector checksum failed.")
                    shutil.copyfile(bundle / "vectors.npy", stage / "vectors.npy")
                else:
                    np.save(stage / "vectors.npy", encode([r["text"] for r in rows], "passage: "))
                matrix = np.load(stage / "vectors.npy")
                if matrix.shape != (len(ids), 384) or matrix.dtype != np.float32 or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=0.001):
                    raise ValueError("Embedding count, dimension or normalization mismatch")
                meta = dict(fingerprint=revision, model=E5, model_revision=REVISIONS[E5], dimension=384,
                            count=len(ids), query_prefix="query: ", passage_prefix="passage: ",
                            vectors_sha256=sha(stage / "vectors.npy"), ready=[])
                (stage / "vectors.npy").replace(folder / "vectors.npy")
                (folder / "ids.json").write_text(json.dumps(ids), encoding="utf8")
                meta["ids_sha256"] = sha(folder / "ids.json")
        if meta["fingerprint"] != revision or load_json(folder / "ids.json") != ids:
            raise ValueError("Corpus and vector row order differ")
        if backend in meta["ready"]:
            return folder
        matrix = np.load(folder / "vectors.npy", mmap_mode="r")
        if sha(folder / "vectors.npy") != meta["vectors_sha256"]:
            raise ValueError("Vector checksum failed")
        if backend == "faiss":
            import faiss
            out = folder / "faiss"
            out.mkdir(exist_ok=True)
            groups = {}
            for i, row in enumerate(rows):
                groups.setdefault(row["episode_id"], []).append(i)
            for ep, positions in groups.items():
                index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
                index.add_with_ids(matrix[positions], np.array(positions, dtype=np.int64))
                faiss.write_index(index, str(out / (ep + ".faiss")))
                if index.ntotal != len(positions):
                    raise ValueError("Incomplete FAISS shard")
        elif backend == "chroma":
            db = collection(folder, revision, create=True)
            for i in range(0, len(rows), 256):
                batch = rows[i:i+256]
                db.upsert(ids=ids[i:i+256], embeddings=matrix[i:i+256],
                          documents=[r["text"] for r in batch],
                          metadatas=[{k:v for k,v in r.items() if k not in ("text","chunk_id")} for r in batch])
            if db.count() != len(ids):
                raise ValueError("Chroma record count mismatch")
        else:
            raise ValueError("Unknown index backend")
        meta["ready"].append(backend)
        tmp = folder / "manifest.next"
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf8")
        tmp.replace(meta_path)
    return folder


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["faiss", "chroma", "both"], default="both")
    backend = p.parse_args().backend
    for choice in (["faiss", "chroma"] if backend == "both" else [backend]):
        print(build(choice))

