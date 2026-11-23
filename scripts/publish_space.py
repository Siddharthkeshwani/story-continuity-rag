"""Publish the verified corpus and a free CPU Docker Space from this checkout."""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationAdd
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def publish(dataset, space):
    api = HfApi()
    api.whoami()
    for key in ("GEMINI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.getenv(key):
            raise RuntimeError(f"Missing {key} for the Space")
    spec = json.loads((ROOT/"deployment/release.json").read_text())
    archive = ROOT/spec["filename"]
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError("Corpus archive checksum mismatch")
    api.create_repo(dataset, repo_type="dataset", exist_ok=True)
    card = "---\nlicense: cc-by-sa-4.0\nlanguage:\n- en\n---\n\n" + (ROOT/"DATA_SOURCES.md").read_text(encoding="utf8")
    result = api.create_commit(repo_id=dataset, repo_type="dataset", commit_message="Publish verified CRD3 search bundle",
        operations=[CommitOperationAdd(path_in_repo=archive.name, path_or_fileobj=str(archive)),
                    CommitOperationAdd(path_in_repo="README.md",path_or_fileobj=card.encode()),
                    CommitOperationAdd(path_in_repo="data/CRD3-LICENSE",path_or_fileobj=str(ROOT/"data/CRD3-LICENSE"))])
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    for key in ("GEMINI_API_KEY", "LANGSMITH_API_KEY"):
        api.add_space_secret(space, key, os.environ[key])
    variables = dict(CHAT_MODEL=os.getenv("CHAT_MODEL","gemini-3.8-flash"), LANGSMITH_TRACING="true",
                     LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT","story-continuity"),
                     HF_DATASET_REPO=dataset, HF_DATASET_REVISION=result.oid)
    for key,value in variables.items():
        api.add_space_variable(space,key,value)
    paths = ["README.md","DATA_SOURCES.md","Dockerfile","requirements.txt","app.py",
             ".streamlit/config.toml","deployment/release.json","data/CRD3-LICENSE"]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/"src").glob("*.py"))]
    api.create_commit(repo_id=space,repo_type="space",commit_message="Deploy the two-campaign Streamlit app",
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(ROOT/name)) for name in paths])
    print(f"https://huggingface.co/datasets/{dataset}\nhttps://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--space", required=True)
    args = parser.parse_args()
    publish(args.dataset, args.space)

