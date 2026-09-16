"""
Download the range3/cc100-ja dataset (Japanese subset of CC-100, re-shared as
shuffled parquet shards) and repackage it into nanochat's expected shard format:

- each output shard is ~100MB (zstd compressed), containing ~250M characters
- parquet, single "text" column, row_group_size=1024
- shuffled across source files
- written directly into nanochat's base_data_climbmix dir so the existing
  nanochat/dataset.py loader picks it up with zero code changes

Usage:
    python -m dev.prepare_cc100_ja --num-source-files 3
"""
import os
import gc
import random
import argparse
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

import requests
import pyarrow.parquet as pq
import pyarrow as pa

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Prepare range3/cc100-ja for nanochat pretraining")
parser.add_argument("--num-source-files", type=int, default=3, help="Number of train_N.parquet source files to download (23 total available, ~2GB each compressed)")
parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel download workers")
parser.add_argument("--chars-per-shard", type=int, default=250_000_000, help="Target characters per output shard (matches nanochat convention)")
parser.add_argument("--doc-cap", type=int, default=10_000, help="Max characters per document (cc100-ja rows are already short, this is just a safety cap)")
args = parser.parse_args()

REPO = "range3/cc100-ja"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"

base_dir = get_base_dir()
output_dir = os.path.join(base_dir, "base_data_climbmix")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Download the chosen number of source shard files to a temp dir

download_dir = tempfile.mkdtemp(prefix="cc100ja_")
print(f"Downloading {args.num_source_files} source files from {REPO} into {download_dir}")

def download_file(filename):
    filepath = os.path.join(download_dir, filename)
    url = f"{BASE_URL}/{filename}"
    print(f"  downloading {filename}...")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  done: {filename} ({size_mb:.1f} MB)")
    return filepath

filenames = [f"train_{i}.parquet" for i in range(args.num_source_files)]
with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
    source_paths = list(pool.map(download_file, filenames))

# -----------------------------------------------------------------------------
# 2) Read the "text" column out of each source file, truncate long docs, shuffle

all_docs = []
for path in source_paths:
    pf = pq.ParquetFile(path)
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        texts = rg.column("text").to_pylist()
        for t in texts:
            if t:
                all_docs.append(t[:args.doc_cap])
    print(f"  read {os.path.basename(path)}: running total {len(all_docs):,} docs")

total_chars = sum(len(d) for d in all_docs)
print(f"Total documents: {len(all_docs):,} | Total characters: {total_chars:,}")

print("Shuffling...")
random.Random(42).shuffle(all_docs)

# Free the downloaded source parquet files now that we've extracted the text
shutil.rmtree(download_dir, ignore_errors=True)

# -----------------------------------------------------------------------------
# 3) Write out into nanochat's shard format

row_group_size = 1024
shard_docs = []
shard_index = 0
shard_characters = 0

def flush_shard():
    global shard_docs, shard_index, shard_characters
    if not shard_docs:
        return
    shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
    shard_table = pa.Table.from_pydict({"text": shard_docs})
    pq.write_table(
        shard_table,
        shard_path,
        row_group_size=row_group_size,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    size_mb = os.path.getsize(shard_path) / 1024 / 1024
    print(f"Wrote {shard_path} | docs: {len(shard_docs):,} | chars: {shard_characters:,} | size: {size_mb:.1f}MB")
    shard_docs = []
    shard_characters = 0
    shard_index += 1

for doc in all_docs:
    shard_docs.append(doc)
    shard_characters += len(doc)
    collected_enough_chars = shard_characters >= args.chars_per_shard
    docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0
    if collected_enough_chars and docs_multiple_of_row_group_size:
        flush_shard()
# flush whatever remains as a final (possibly smaller) shard
flush_shard()

del all_docs
gc.collect()

print(f"\nDone. Wrote {shard_index} shards to {output_dir}")
print(f"Total characters packaged: {total_chars:,}")
if shard_index < 2:
    print("WARNING: fewer than 2 shards were written. nanochat requires >=2 shards (last one is used as val split).")
