# Two-campaign verification

**The upgraded release covers 398,682 original dialogue turns.** Both backends use the same 47,578 passage embeddings. Each of the 159 FAISS files includes the campaign in its filename. A fresh restore from the release ZIP reopened 159 shards and 47,578 Chroma records.

## Search results

The original 30 Campaign 1 test questions were preserved. Four new Campaign 2 questions check early dialogue, an episode 46 recap and the same reveal at an early cutoff. A correct location must enclose an annotated source turn. Six questions have an expected unresolved answer.

| Search method | Annotated source found | All source spans for multi-part questions | Wrong campaign or later episode |
|---|---:|---:|---:|
| V2 · FAISS | 23/28 | 4/6 | 0/34 |
| V3 · Chroma | 22/28 | 4/6 | 0/34 |
| V4 · Chroma and reranking | 23/28 | 4/6 | 0/34 |

These are retrieval-only runs. V4's median retrieval-plus-reranking request took 222 ms on the test GPU, excluding answer generation. The app found an accepted early source for Trostenwald and the Nestled Nook Inn. On the episode 46 ship question, none of the final five contained the annotated episode 46 turn. [Per-question rankings](retrieval.csv) · [Timing table](retrieval_comparison.csv)

**Abstention trades coverage for fewer guesses.** The pinned reranker emits raw scores. The initial threshold is zero. This would stop generation for 12 of 34 held-out questions, including all six expected abstentions and six questions that had answers. The threshold does not measure factual accuracy. A related but misleading scene can score highly. A second check verifies each source ID and verbatim quote against the supplied passages.

**Answer generation.** The project uses Google’s `gemini-3.8-flash`. Google returned HTTP 404 for `gemini-2.5-flash` with this account and recommended the replacement. The table above measures source retrieval. Historical Qwen answer scores are archived separately. [Historical review](qwen-baseline/findings.md)
