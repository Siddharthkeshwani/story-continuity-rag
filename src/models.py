"""Pinned public model files for reproducible offline CPU inference."""
from pathlib import Path
import argparse
import shutil
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
E5 = "intfloat/multilingual-e5-small"
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISIONS = {E5: "614241f622f53c4eeff9890bdc4f31cfecc418b3",
             RERANKER: "1427fd652930e4ba29e8149678df786c240d8825"}


def model_dir(name):
    return ROOT / ".models" / name.split("/")[-1] / REVISIONS[name]


def download(name):
    folder = model_dir(name)
    files = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "sentencepiece.bpe.model", "model.safetensors"]
    if name == E5:
        files = ["modules.json", "1_Pooling/config.json", "sentence_bert_config.json"] + files
    for filename in files:
        dest = folder / filename
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".download")
        url = f"https://huggingface.co/{name}/resolve/{REVISIONS[name]}/{filename}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
                    length = response.headers.get("Content-Length")
                if length and tmp.stat().st_size != int(length):
                    raise OSError("Truncated model file")
                tmp.replace(dest)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        print(name, filename, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(REVISIONS), nargs="+", default=list(REVISIONS))
    for model in parser.parse_args().model:
        download(model)

