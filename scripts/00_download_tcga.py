#!/usr/bin/env python3
"""Download TCGA pan-cancer expression, mutation, and phenotype data from UCSC Xena."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

XENA_BASE = "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download"

FILES = {
    "expression": (
        f"{XENA_BASE}/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz",
        "tcga_expression.tsv.gz",
    ),
    "mutations": (
        f"{XENA_BASE}/mc3.v0.2.8.PUBLIC.xena.gz",
        "tcga_mutations.tsv.gz",
    ),
    "phenotype": (
        f"{XENA_BASE}/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
        "tcga_phenotype.tsv.gz",
    ),
}


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r  {mb:.1f} / {total_mb:.1f} MB  ({pct:.0f}%)", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TCGA data from UCSC Xena.")
    parser.add_argument("--out-dir", default="data/raw", help="Directory to save files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (url, filename) in FILES.items():
        dest = out_dir / filename
        if dest.exists():
            print(f"[skip] {filename} already exists")
            continue
        print(f"Downloading {name}: {filename}")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            print(f"\n  -> saved to {dest}")
        except Exception as exc:
            print(f"\n  ERROR downloading {name}: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\nAll TCGA files downloaded.")


if __name__ == "__main__":
    main()
