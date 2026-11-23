"""Read back real LangSmith runs and save a small offline trace view."""
import html
import json
import os
from time import sleep
from uuid import uuid4
from langsmith import Client
from src.config import ROOT, manifest
from src.retrieve import check_story
from src.v1 import QUESTION


def flatten(run, depth=0):
    duration = None if not run.end_time else round((run.end_time-run.start_time).total_seconds()*1000)
    fields = dict(name=run.name, depth=depth, type=run.run_type, duration_ms=duration,
                  error=bool(run.error), tokens=run.total_tokens or 0)
    return [fields] + [row for child in run.child_runs or [] for row in flatten(child, depth+1)]


def record(client, label, question, cutoff):
    run_id = uuid4()
    try:
        result = check_story(question, "v4", 1, cutoff, manifest()["fingerprint"],
                             langsmith_extra={"run_id":run_id})
    except ValueError:
        if label != "Controlled error":
            raise
        result = {}
    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
    client.flush()
    for _ in range(12):
        try:
            run = client.read_run(run_id, load_child_runs=True)
            if run.end_time and (label=="Controlled error" or result.get("guardrail") or any(x["type"]=="llm" for x in flatten(run))):
                break
        except Exception:
            pass
        sleep(2)
    else:
        raise RuntimeError("The completed trace could not be read back")
    return dict(label=label, question=question, cutoff=cutoff, run_id=str(run_id),
                url=client.get_run_url(run=run), stages=flatten(run), result=result)


def main():
    client = Client()
    records = [record(client,"Answerable",QUESTION,1),
               record(client,"Early cutoff","How much gold did the group recover after the treasure chest exploded?",1),
               record(client,"Controlled error"," ",1)]
    out = ROOT/"results"
    (out/"diagnostics.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8")
    sections = []
    for r in records:
        rows = "".join(f"<tr><td style='padding-left:{12+14*x['depth']}px'>{html.escape(x['name'])}</td><td>{x['duration_ms']}</td><td>{x['tokens'] or '—'}</td><td>{'Error' if x['error'] else 'OK'}</td></tr>" for x in r["stages"])
        parts = r["result"].get("answer",{}).get("parts",[])
        claims = "".join(f"<p>{html.escape(p['claim'])} <small>{html.escape(', '.join(p['evidence_ids']))}</small></p>" for p in parts)
        sections.append(f"<section><h2>{r['label']}</h2><p>{html.escape(r['question']) or 'Blank question'}</p>{claims}<table><tr><th>Stage</th><th>ms</th><th>Tokens</th><th>Status</th></tr>{rows}</table><p><a href='{html.escape(r['url'])}'>Open this run in LangSmith</a></p></section>")
    page = """<!doctype html><meta charset='utf-8'><title>Continuity traces</title><style>
    body{max-width:900px;margin:40px auto;background:#faf8f1;color:#25382f;font:18px/1.45 system-ui}
    section{padding:8px 0 24px;border-top:1px solid #b1bcad}table{width:100%;border-collapse:collapse}
    th,td{text-align:left;padding:5px 12px;border-bottom:1px solid #dee2d8}th{background:#deeadf}
    h1{font:48px Georgia}h2{font:30px Georgia}a,small{color:#136453}</style>
    <h1>Story continuity traces</h1><p>Offline copy of stage timings read back from LangSmith. Parent times include child runs.</p>"""
    (out/"trace_report.html").write_text(page+"".join(sections),encoding="utf8")


if __name__ == "__main__":
    main()

