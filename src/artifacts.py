"""Package and restore the exact corpus and both native index backends."""
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile
from filelock import FileLock
from huggingface_hub import hf_hub_download
from .config import BUILD, CORPUS_LOCK, ROOT, load_json, manifest, sha

ARCHIVE = "crd3_index_v1.zip"
RELEASE = ROOT / "deployment" / "release.json"
DATASET_REPO = "Siddharth156/story-continuity-index"


def ready(root, spec):
    revision = spec["fingerprint"]
    index = root / "indexes" / revision
    processed = root / "data" / "processed"
    try:
        return (load_json(processed / "manifest.json")["fingerprint"] == revision
                and all((processed / name).is_file() for name in ("turns.jsonl", "chunks.jsonl"))
                and all((index / name).is_file() for name in ("vectors.npy", "ids.json", "chroma/chroma.sqlite3"))
                and len(list((index / "faiss").glob("*.faiss"))) == spec["episodes"]
                and set(load_json(index / "manifest.json")["ready"]) == {"faiss", "chroma"})
    except (OSError, ValueError, KeyError):
        return False


def package():
    meta = manifest()
    rev = meta["fingerprint"]
    spec = dict(fingerprint=rev, episodes=len(meta["episode_ids"]), filename=ARCHIVE)
    if not ready(ROOT, spec):
        raise RuntimeError("Build both indexes before packaging.")
    path = ROOT / ARCHIVE
    tmp = BUILD / (ARCHIVE + ".tmp")
    base = ROOT / "indexes" / rev
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=3) as z:
        files = [*(ROOT/"data/processed").glob("*.json*"), ROOT/"data/CRD3-LICENSE", ROOT/"DATA_SOURCES.md",
                 base/"manifest.json", base/"ids.json", base/"vectors.npy", *(base/"faiss").glob("*.faiss"),
                 *(p for p in (base/"chroma").rglob("*") if p.is_file())]
        for file in sorted(files):
            z.write(file, file.relative_to(ROOT).as_posix())
    spec["sha256"] = sha(tmp)
    tmp.replace(path)
    RELEASE.parent.mkdir(exist_ok=True)
    RELEASE.write_text(json.dumps(spec, indent=2)+"\n", encoding="utf8")
    print(path, spec["sha256"])


def restore(archive, spec, root=ROOT):
    if sha(archive) != spec["sha256"]:
        raise ValueError("Dataset archive checksum failed.")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as staging:
        stage = Path(staging)
        with ZipFile(archive) as z:
            for member in z.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts or ":" in str(name) or "\\" in str(name):
                    raise ValueError("Invalid archive member path")
                if not (name.is_relative_to("data/processed") or name.is_relative_to("indexes/" + spec["fingerprint"])):
                    continue
                z.extract(member, stage)
        if not ready(stage, spec):
            raise ValueError("Incomplete archive")
        # Publish complete directories under a lock. Preserve the old directories on failure.
        paths = [Path("indexes")/spec["fingerprint"], Path("data/processed")]
        backups, installed = [], []
        with CORPUS_LOCK:
            try:
                for i, rel in enumerate(paths):
                    dest = root/rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        backup = stage/f"old-{i}"
                        dest.rename(backup)
                        backups.append((dest, backup))
                    (stage/rel).rename(dest)
                    installed.append((dest, stage/rel))
            except OSError:
                for dest, original in reversed(installed):
                    dest.rename(original)
                for dest, backup in reversed(backups):
                    backup.rename(dest)
                raise


def ensure_assets():
    spec = load_json(RELEASE)
    with FileLock(str(BUILD / "restore.lock")):
        if ready(ROOT, spec):
            return
        archive = ROOT / ARCHIVE
        if not archive.exists():
            dataset = os.getenv("HF_DATASET_REPO") or DATASET_REPO
            archive = Path(hf_hub_download(repo_id=dataset, repo_type="dataset", filename=ARCHIVE,
                                          revision=os.getenv("HF_DATASET_REVISION") or spec.get("dataset_revision") or "main"))
        restore(archive, spec)


if __name__ == "__main__":
    package()

