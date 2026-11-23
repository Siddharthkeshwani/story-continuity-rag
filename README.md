---
title: Story Continuity Assistant
emoji: 📚
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
---

# Story Continuity Assistant

A writer selects a campaign and the last episode they can use, asks a story question and inspects the original English dialogue. The dataset covers **159 episodes**, with 113 in Campaign 1 and 46 in Campaign 2. Selecting episode 1 of Campaign 2 excludes every Campaign 1 passage, including its editorial facts.

## Assignment summary

| Field | Description |
|---|---|
| Project title | Story Continuity Assistant for Episodic Writers |
| Problem | A recap can assign a line to the wrong character or reveal something the writer hasn't reached. |
| Solution | Filter by campaign and cutoff, rank passages and return a verdict with supporting text for each claim. |
| Users | Writers, script editors and adaptation teams. |
| Sources | Frozen English CRD3 dialogue and four reviewed Campaign 1 editorial facts. |
| Output | A claim or abstention with episode, original turn range and expandable dialogue. |

## Run

The prepared Windows workspace runs with `.\scripts\start.ps1`. The launcher creates the Python 3.11 environment on a fresh checkout. Put the Google Gemini free-plan credential in the root `.env` as `GEMINI_API_KEY`. All versions, the UI and the evaluator use `CHAT_MODEL=gemini-3.8-flash`. Search-only mode works without generation credentials. LangSmith uses `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`. The key names are documented in [.env.example](.env.example).

The app opens on [localhost](http://127.0.0.1:8501). In Campaign 1, ask who requested help finding Lady Kima. In Campaign 2, ask which town borders the Ustaloch in the opening narration. Expand the evidence for the original speaker labels. Campaign and cutoff changes clear displayed results.

## Prepare or restore the data

The code repository excludes raw dialogue, processed files, matrices and index caches. A fresh deployment downloads the matching archive from the published Dataset. Set `HF_DATASET_REPO` to override the repository or `HF_DATASET_REVISION` to pin its release. A local checkout can instead place `crd3_index_v1.zip` in the project root. Startup verifies the release checksum before extracting it. [Pinned release metadata](deployment/release.json) · [Dataset attribution](DATA_SOURCES.md)

To build from the frozen source on Python 3.11, run the following from the root. The preparation script accepts a local archive through `--archive`. The embedding model automatically selects CUDA when the Python environment has a CUDA-enabled PyTorch installation. Chroma and the exact FAISS shards use the shared GPU-generated float32 embeddings on CPU.

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python -m src.models
python -m src.ingest
python -m src.indexes --backend faiss
python -m src.indexes --backend chroma
python -m src.artifacts
```

## Retrieval and answering

| Version | Source selection |
|---|---|
| V1 | Manually selected C1E001 turn 51. |
| V2 | Preloaded FAISS shards keyed by campaign and episode, merged into an eligible top five. |
| V3 | Persistent Chroma cosine search with both campaign and cutoff in the database filter. |
| V4 | Up to 20 Chroma candidates, reranked to five. A top relevance logit below zero returns insufficient evidence before any chat request. |

The generator must cite supplied IDs and produce a matching verbatim extract for each cited passage. Invalid citations or quotes trigger one correction attempt and then an unresolved verdict for the affected part. The checks establish that the words were present. A reader still needs to verify who said them and whether they support the claim.

```powershell
python -m src.v1
python -m src.retrieve --version v4 --campaign 2 --cutoff 1 --evidence-only --question "Which town borders the Ustaloch?"
python -m unittest evaluation.test_integrity
python -m evaluation.run --cases evaluation/test.csv --versions v2 v3 v4
```

The evaluator pauses seven seconds between hosted questions and records bounded retries for HTTP 429. After repeated provider failures, the evaluator keeps checking the sources and periodically probes the provider. Rows marked `ProviderCooldown` were skipped during that pause. Use `--resume` to retry unfinished answers with the same corpus, cases and model. The separate evidence-only result measures retrieval with 34 held-out questions. Historical answers using Qwen are under `results/qwen-baseline/`. [Current findings](results/findings.md) · [Search comparison](results/retrieval_comparison.csv) · [Presentation](presentation/Story_Continuity.pptx)

## Cloud deployment

The public Hugging Face Dataset holds the 378 MB archive. The runtime can use Streamlit Community Cloud or a Hugging Face Docker Space. Hugging Face returned HTTP 402 when this account attempted to create a CPU Basic Docker Space. [Published Dataset](https://huggingface.co/datasets/Siddharth156/story-continuity-index) The release package includes both backends, the shared vectors, ordered IDs, processed turns and chunks, release metadata and the original data licence. The same package serves both the app and CLI. The release checksum is verified before the app opens either index.

`scripts/publish_space.py --dataset OWNER/DATASET --space OWNER/SPACE`, executed with Python, publishes the verified archive, sets the Dataset revision and the model, adds provider credentials through Space Secrets and uploads an allowlist of app files. Publishing requires a Hugging Face write login or `HF_TOKEN`. Deployment uses the project's Dockerfile. [Hugging Face Streamlit hosting](https://huggingface.co/docs/hub/spaces-sdks-streamlit)

Streamlit Community Cloud uses the GitHub repository’s `main` branch, `app.py` and Python 3.11. Store the Gemini and LangSmith settings in the app’s Secrets panel. The public archive downloads at the revision recorded in the release file. [Community Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
