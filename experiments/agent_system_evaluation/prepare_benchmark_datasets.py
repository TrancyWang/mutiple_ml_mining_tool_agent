"""Download and normalize public benchmark datasets for the agent comparison.

The script intentionally keeps the source metadata next to the downloaded
files.  It does not call Gemini and is safe to rerun; existing files are
reused unless --force is supplied.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "benchmark_data"

SOURCES = {
    "bank_marketing": {
        "task": "tabular_classification",
        "url": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "citation": "Moro, Rita, and Cortez (2014), UCI Machine Learning Repository",
        "license": "CC BY 4.0",
        "description": "Predict whether a bank customer subscribes to a term deposit.",
    },
    "wine_quality": {
        "task": "tabular_regression_and_ordinal_classification",
        "url": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "citation": "Cortez et al. (2009), UCI Machine Learning Repository",
        "license": "CC BY 4.0",
        "description": "Predict wine quality from physicochemical measurements.",
    },
    "chnsenticorp": {
        "task": "chinese_sentiment_classification",
        "url": "https://huggingface.co/datasets/seamew/ChnSentiCorp",
        "citation": "ChnSentiCorp; source card reports 9,600/1,200/1,200 examples",
        "license": "Not specified on the source card; verify before redistribution",
        "description": "Chinese binary sentiment classification.",
    },
    "clue_tnews": {
        "task": "chinese_topic_classification",
        "url": "https://github.com/CLUEbenchmark/CLUE",
        "citation": "CLUE benchmark; TNEWS short-text news classification",
        "license": "Unknown on the Hugging Face dataset card; use source terms",
        "description": "Chinese short-text news classification with 15 categories.",
    },
}


def download(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "academic-agent-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def prepare_uci(force: bool) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="agent_benchmark_") as temp_dir:
        temp_zip = Path(temp_dir) / "uci.zip"
        download(SOURCES["bank_marketing"]["url"], temp_zip, force=force)
        with zipfile.ZipFile(temp_zip) as archive:
            names = archive.namelist()
            bank_name = next((name for name in names if name.endswith("bank-full.csv")), None)
            if bank_name is not None:
                with archive.open(bank_name) as handle:
                    bank = pd.read_csv(handle, sep=";")
            else:
                nested_name = next(name for name in names if name.endswith("bank.zip"))
                nested_bytes = archive.read(nested_name)
                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                    bank_name = next(name for name in nested.namelist() if name.endswith("bank-full.csv"))
                    with nested.open(bank_name) as handle:
                        bank = pd.read_csv(handle, sep=";")
            bank.to_csv(DATA_DIR / "bank_marketing.csv", index=False)

        temp_zip.unlink(missing_ok=True)
        download(SOURCES["wine_quality"]["url"], temp_zip, force=force)
        with zipfile.ZipFile(temp_zip) as archive:
            frames = []
            for name in archive.namelist():
                if name.endswith("winequality-red.csv"):
                    with archive.open(name) as handle:
                        red = pd.read_csv(handle, sep=";")
                    red["wine_type"] = "red"
                    frames.append(red)
                elif name.endswith("winequality-white.csv"):
                    with archive.open(name) as handle:
                        white = pd.read_csv(handle, sep=";")
                    white["wine_type"] = "white"
                    frames.append(white)
            wine = pd.concat(frames, ignore_index=True)
            wine.to_csv(DATA_DIR / "wine_quality.csv", index=False)
    return {"bank_marketing": len(bank), "wine_quality": len(wine)}


def read_arrow(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    source = pa.BufferReader(raw)
    try:
        table = ipc.open_stream(source).read_all()
    except pa.ArrowInvalid:
        source = pa.BufferReader(raw)
        table = ipc.open_file(source).read_all()
    return table.to_pandas()


def prepare_chnsenticorp(force: bool) -> dict[str, int]:
    split_urls = {
        "train": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-train.arrow",
        "validation": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-validation.arrow",
        "test": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/chn_senti_corp-test.arrow",
    }
    sizes = {}
    for split, url in split_urls.items():
        raw_path = DATA_DIR / "raw" / f"chnsenticorp-{split}.arrow"
        frame = read_arrow(download(url, raw_path, force=force))
        frame = frame[["text", "label"]].copy()
        frame["label"] = frame["label"].astype(int)
        output = DATA_DIR / f"chnsenticorp_{split}.parquet"
        write_frame(frame, output)
        sizes[split] = len(frame)
    return sizes


def prepare_tnews(force: bool) -> dict[str, int]:
    sizes = {}
    for split in ("train", "validation", "test"):
        url = f"https://huggingface.co/datasets/clue/clue/resolve/main/tnews/{split}-00000-of-00001.parquet"
        raw_path = DATA_DIR / "raw" / f"tnews-{split}.parquet"
        frame = pd.read_parquet(download(url, raw_path, force=force))
        columns = [column for column in ("sentence", "label", "label_des") if column in frame.columns]
        frame = frame[columns].copy()
        frame["label"] = frame["label"].astype(str)
        write_frame(frame, DATA_DIR / f"clue_tnews_{split}.parquet")
        sizes[split] = len(frame)
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public benchmark datasets")
    parser.add_argument("--force", action="store_true", help="redownload raw files")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts = {
        "tabular": prepare_uci(args.force),
        "chnsenticorp": prepare_chnsenticorp(args.force),
        "clue_tnews": prepare_tnews(args.force),
    }
    (DATA_DIR / "metadata.json").write_text(
        json.dumps({"sources": SOURCES, "counts": counts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
