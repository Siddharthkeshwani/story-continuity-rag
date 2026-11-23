# What the continuity checks found

**The search found useful dialogue more often than the model produced a correct answer.** On the ten-episode edition, V4 found an annotated source on 21 of 25 answerable questions. Across all 30 held-out questions, 7 answers were fully correct and supported. Being able to open the original exchange matters for this prototype.

## Ten-episode comparison

| Measure | V2 · FAISS | V3 · Chroma | V4 · Chroma and reranker |
|---|---:|---:|---:|
| Fully correct and supported | 6/30 (20%) | 6/30 (20%) | 7/30 (23%) |
| Annotated source in final five | 20/25 (80%) | 20/25 (80%) | 21/25 (84%) |
| All required source spans on multi-part questions | 4/6 (67%) | 4/6 (67%) | 4/6 (67%) |
| Factual output claims with support | 9/17 (53%) | 9/18 (50%) | 13/23 (57%) |
| Cases with later-episode evidence or an observed future-fact disclosure | 0/30 | 0/30 | 0/30 |
| Median end-to-end time | 2.74 s | 1.87 s | 4.96 s |
| 95th percentile time | 4.16 s | 3.70 s | 8.10 s |
| Median search / rerank time | 37 ms / 0 ms | 62 ms / 0 ms | 63 ms / 2,220 ms |
| Input / output tokens, including retries | 65,898 / 3,120 | 64,433 / 3,039 | 63,326 / 3,229 |
| Validation retries / operational failures | 17 / 0 | 16 / 0 | 15 / 0 |

All 90 observations have a review decision and a reason. A question with an unsupported assertion, the wrong person, a missing requested fact or an unexplained abstention failed the accuracy check. A factual output part counted as supported when at least one of its supplied citations established the part. Location coverage and answer correctness were reviewed separately. The two wordings in the Percy pair concern the same fact, so the 30 rows do not represent 30 independent story facts. [Per-question records](runs.csv) · [Review notes](runs_reviews.csv) · [Scoring rules](../evaluation/README.md)

**Where reranking helped.** In the episode 7 question about Kima and Clarota, V4 found the recap and described the alliance as tenuous. The earlier versions returned a sentence about the existence of an alliance without saying what it was like. One additional correct answer on this small set is a modest difference.

**Where the answer went wrong.** For the statue question, the correct passage stayed at rank 1 before and after reranking. It records Clarota suggesting that the group leave Kima there for safekeeping. The generator instead cited the third-ranked passage, in which Liam suggests taking first watch. It assigned Liam's suggestion to Clarota. The search found the scene. The generator chose a different exchange and reversed the speaker. [Inspect the V4 run in LangSmith](https://smith.langchain.com/o/98abc327-28d4-45f2-bfae-9d1411fb60d6/projects/p/a2ecd954-e481-404a-8730-7ddb35e4b599/r/be788c21-8041-4c2d-b6e4-8ec92edd6aa2)

**What a cutoff protects.** Every logged initial candidate, final passage, prompt passage and citation stayed within its campaign and cutoff in the held-out runs. Review found no affirmative disclosure of the annotated later facts. The model still invented details from eligible passages and sometimes treated an example in the prompt as story content. In the diagnostic early-cutoff gold question it mentioned a letter while marking the answer unresolved. A generic citation failure also gives the writer little explanation of what is missing. [Offline trace view](trace_report.html)

**Cold start and cost.** A second Python installation with an empty index cache took 43.47 seconds to construct the ten-episode Chroma index from the packaged vectors, load the locally available retrieval models and return five ranked passages. F001 came first. The timer excludes package installation, model downloads and answer generation. The median and tail times above are warm end-to-end timings on this laptop. Other local verification tasks were running during parts of the benchmark, and the model server can reuse prompt prefixes. The latency differences are observed values for this run. Local inference incurred **zero per-token API charges**. Electricity and hardware cost were not measured. The separate OpenAI check failed before generation with HTTP 429, `credit_balance_exhausted`.

## Rerun after indexing the full campaign

The larger edition contains 113 episodes, 281,516 turns and 33,669 passages including four editorial facts. The same 30 locked questions produced another 90 reviewed observations.

| Measure | V2 | V3 | V4 |
|---|---:|---:|---:|
| Fully correct and supported | 6/30 | 6/30 | 7/30 |
| Annotated source in final five | 20/25 | 20/25 | 21/25 |
| Complete multi-part coverage | 4/6 | 4/6 | 4/6 |
| Factual claims with support | 9/18 | 9/18 | 13/22 |
| Median / 95th percentile | 2.20 / 4.37 s | 1.92 / 3.66 s | 5.95 / 9.49 s |
| Operational failures / observed future-fact leaks | 0 / 0 | 0 / 0 | 0 / 0 |

All locked cutoffs are at or before episode 10. Their rerun checks whether adding later material changes an early answer. It cannot measure general accuracy on the later storyline. Three additional V4 probes therefore opened the search through episodes 50, 100 and 115. Two retrieved the annotated turn. None gave a fully supported answer under the question-specific rubric. On the episode 50 question the answer named Thordak correctly, but cited an earlier recap for a claim about what episode 50 said. On episode 115 the target turn reached the final five and the answer withheld the fact. [Full-campaign results](campaign1_runs_comparison.csv) · [Late-episode probes](late_probes.csv) · [Late-probe review](late_probes_reviews.csv)

## Rehearsed demonstration

1. Open the local Streamlit screen at episode 1. Ask who requested help finding Kima. V4 cites the reviewed fact F001. Its source opens Matt's original turn 51.
2. Ask “What was the light Grog charged toward before the duergar ambush?” at episode 1. The recorded run withheld the answer. At episode 3, the answer named the campfire and cited passages containing Matt's recap at turn 74.
3. Open the trace report for an answerable request, an early-cutoff request and a controlled blank-question error. The normal trace includes the prompt and local chat call, and records 1,349 tokens. The controlled error stops before generation. Private trace links require project access. The offline report and screenshot work without that access. [Recorded demo](demo.json) · [Trace screenshot](../presentation/trace.png)

## Execution record

| Guide step | Result |
|---|---|
| 1. Problem statement | Six fields are in the app docstring and README. External form submission was deferred by request. |
| 2–4. Sources, environment and files | Frozen CRD3 input, model revisions and attribution are documented. Python 3.11.9 and pinned dependencies passed a clean installation check. |
| 5–6. First answer and manual baseline | Verified episode 1 turn 51 directly. Ran the free-text chain, structured answer and unrelated-turn negative control. |
| 7. Ingestion | Rebuilding the ten-episode corpus reproduced its manifest and hashes. Checks compared all 27,561 source turns to the frozen input and resolved all demo passage ranges. |
| 8–10. Search versions | Built both persistent backends from one ordered matrix per edition. Exercised cutoff search, reranking and the initial/final rank logs. |
| 11. Writer's screen | A fresh browser opened the original citation, changed cutoff, ran another query and exercised the blank-input message. |
| 12. Evaluation | Scored 30 development observations, 180 held-out observations over two corpus sizes and three late-episode probes. Published the held-out accuracy feedback to LangSmith. |
| 13. Monitoring | Read three completed diagnostic traces back from LangSmith and saved their stages and private links locally. |
| 14. Deployment | Verified local service and a fresh environment with an empty index cache. Community Cloud publication awaits the deferred GitHub upload. |
| 15–16. Presentation and hand-in | Six editable PowerPoint slides, source attribution, results, trace screenshot, demo questions and reproducible startup commands are in the project. |

