# Sources and attribution

Revanth Rameshkumar and Peter Bailey created CRD3 from the Critical Role show, community episode transcripts and Fandom wiki summaries. Credit belongs to the original performers, Dungeon Master, Critical Role transcript contributors, wiki editors and dataset authors. CRD3 is distributed under Creative Commons Attribution-ShareAlike 4.0. The processed transcripts, derived editorial facts, evidence labels and associated passage data in this project are offered under the same licence. [Authors’ repository](https://github.com/RevanthRameshkumar/CRD3) · [ACL paper](https://aclanthology.org/2020.acl-main.459/) · [Licence text](data/CRD3-LICENSE)

## Frozen input

The frozen CRD3 ZIP has SHA256 `c77a937394f265735ba54b32a7a051f77a97d264c74b0535dee77ef9791815b5`. Both campaigns come from the single `baseline/data/cleaned data` tree. Campaign 1 has 113 files through episode 115, with episodes 12 and 51 absent. Campaign 2 has all 46 files from episode 1 through 46. Together they contain 398,682 original turns, 47,574 transcript passages and four Campaign 1 editorial facts. [Authors' archive](https://github.com/RevanthRameshkumar/CRD3/archive/refs/heads/master.zip)

The two-campaign corpus fingerprint is `9bbd2cd2e108f958f4dffff097728ff0a35b2fe6f545ee451af9e12133e662e5`. The packaged manifest records the SHA256 of each input and processed table. Every passage has a numeric campaign and episode, and the original speaker labels and coordinates.

## Changes and evidence

We joined each turn's utterances with spaces, retained the original zero-based number and all speaker labels, and grouped adjacent dialogue into passages of up to approximately 248 embedding tokens. Up to two turns overlap where they fit. Long turns are split using tokenizer offsets and retain their original turn number. A source panel opens the entire original exchange, including the full turn when a passage contains a subspan. The index never combines episodes. Aligned summaries from the frozen c=2 tree helped identify scenes to inspect. They and the evaluation CSVs are excluded from both searchable indexes.

Four editorial facts summarize specific exchanges. Each cites the episode where that evidence appears. For example, the episode 7 recap of Kima's rescue is eligible from episode 7; the story event occurred earlier. Narrator knowledge is attributed to the narrator unless the dialogue shows what a character actually learned.

## Model provenance

- The E5 snapshot used for indexing and query encoding is `614241f622f53c4eeff9890bdc4f31cfecc418b3`. [Model card](https://huggingface.co/intfloat/multilingual-e5-small)
- The cross-encoder snapshot is `1427fd652930e4ba29e8149678df786c240d8825`. [Model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- Answer generation runs through Google’s OpenAI-compatible endpoint. It uses the model recorded with each benchmark run. The earlier Qwen results are preserved in the historical archive. [Google Gemini API](https://ai.google.dev/gemini-api/docs/openai)
