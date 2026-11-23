"""Attach reviewed scores to the matching private LangSmith runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid5
from langsmith import Client
from src.config import ROOT
from .run import read_csv


def publish(path):
    client = Client()
    reviews = {r["review_key"]:r for r in read_csv(path.with_name(path.stem + "_reviews.csv"))}
    def attach(row):
        review = reviews[row["review_key"]]
        trace = UUID(row["trace_id"])
        client.create_feedback(run_id=trace, trace_id=trace, key="grounded_answer_accuracy",
                               score=int(review["correct"]), comment=review["note"],
                               feedback_id=uuid5(trace, "grounded_answer_accuracy"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(attach, read_csv(path)))
    client.flush()
    print(f"Published reviewed scores for {path.name}.")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("runs", type=Path, nargs="+")
    for file in p.parse_args().runs:
        publish(file)

