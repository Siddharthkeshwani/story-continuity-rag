import argparse
import json
from .answer import generate, smoke
from .config import RAW, load_json, corpus, manifest, turn_id

QUESTION = "Who asked Vox Machina to help find Lady Kima, and what was their relationship?"


def passage(turn_no=51):
    raw = RAW / "C1E001.json"
    if raw.exists():
        row = load_json(raw)["TURNS"][turn_no]
        text = ", ".join(row["NAMES"]) + ": " + " ".join(row["UTTERANCES"])
    else:
        turn = corpus(manifest()["fingerprint"])["turns"][turn_id("C1E001", turn_no)]
        text = ", ".join(turn["speakers"]) + ": " + turn["text"]
    return dict(chunk_id=f"C1E001:T{turn_no:04}", episode_id="C1E001", episode_no=1,
                campaign_id=1, turn_start=turn_no, turn_end=turn_no,
                text=text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--unrelated", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    context = [passage(12 if args.unrelated else 51)]
    result = (smoke if args.smoke else generate)(QUESTION, context, 1, 1, "English")
    print(json.dumps(result, ensure_ascii=True, indent=2))

