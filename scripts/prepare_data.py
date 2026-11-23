"""Import both frozen campaigns from the verified upstream ZIP."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "c77a937394f265735ba54b32a7a051f77a97d264c74b0535dee77ef9791815b5"
URL = "https://github.com/RevanthRameshkumar/CRD3/archive/refs/heads/master.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    folder = ROOT / "data" / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    archive = args.archive or folder / "CRD3-master.zip"
    if not archive.is_file():
        with urlopen(URL, timeout=120) as src, archive.open("wb") as out:
            shutil.copyfileobj(src, out)
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != DIGEST:
            raise ValueError("Archive differs from the documented frozen snapshot")
    with ZipFile(archive) as z:
        for name in z.namelist():
            match = re.fullmatch(r"[^/]+/(baseline/data/(?:cleaned data/C\d+E\d+|aligned data/c=2/C\d+E[\d_]+)\.json)", name)
            if match:
                dest = folder / match[1]
                dest.parent.mkdir(exist_ok=True, parents=True)
                with z.open(name) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)


if __name__ == "__main__":
    main()

