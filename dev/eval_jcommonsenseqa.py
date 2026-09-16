"""
Zero-shot / few-shot evaluation of the base model on JCommonsenseQA
(5-choice Japanese commonsense QA), following the same average-token-loss
scoring approach as nanochat/core_eval.py's multiple_choice task type.

Data source: leemeng/jcommonsenseqa-v1.1 (validation split), downloaded on
first run to ~/.cache/nanochat/jcommonsenseqa/validation.parquet
"""
import os
import argparse
import random

import requests
import pyarrow.parquet as pq

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import forward_model, stack_sequences

DATA_URL = "https://huggingface.co/datasets/leemeng/jcommonsenseqa-v1.1/resolve/main/data/validation-00000-of-00001-40f65e99de401641.parquet"
NUM_FEWSHOT = 2

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=-1, help="limit number of eval examples (-1 = all)")
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval")
print(f"Loaded checkpoint step: {meta.get('step')}, val_bpb: {meta.get('val_bpb')}")

base_dir = get_base_dir()
data_dir = os.path.join(base_dir, "jcommonsenseqa")
data_path = os.path.join(data_dir, "validation.parquet")
if not os.path.exists(data_path):
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading JCommonsenseQA validation set to {data_path}...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    with open(data_path, "wb") as f:
        f.write(response.content)

table = pq.read_table(data_path)
data = table.to_pylist()
print(f"Loaded {len(data)} JCommonsenseQA validation examples")

def render_item(item, with_answer_idx=None):
    choices = [item[f"choice{i}"] for i in range(5)]
    q = f"質問: {item['question']}\n選択肢: " + " / ".join(choices) + "\n答え: "
    if with_answer_idx is not None:
        q += choices[with_answer_idx]
    return q, choices

rng = random.Random(1234)
pad_token_id = tokenizer.get_bos_token_id()
correct = 0
n_eval = len(data) if args.limit < 0 else min(args.limit, len(data))

for idx in range(n_eval):
    item = data[idx]
    # few-shot prefix
    other_idxs = [i for i in range(len(data)) if i != idx]
    fewshot_idxs = rng.sample(other_idxs, NUM_FEWSHOT)
    prefix = ""
    for fi in fewshot_idxs:
        fitem = data[fi]
        ftext, _ = render_item(fitem, with_answer_idx=fitem["label"])
        prefix += ftext + "\n\n"
    qtext, choices = render_item(item)
    prompts = [prefix + qtext + c for c in choices]
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # common prefix length = start of continuation for each choice
    min_len = min(len(t) for t in tokens)
    start_idx = min_len
    for i in range(min_len):
        tok0 = tokens[0][i]
        if not all(t[i] == tok0 for t in tokens):
            start_idx = i
            break
    end_idxs = [len(t) for t in tokens]
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, _ = forward_model(model, input_ids)
    mean_losses = [losses[i, start_idx-1:end_idxs[i]-1].mean().item() for i in range(len(choices))]
    pred = mean_losses.index(min(mean_losses))
    if pred == item["label"]:
        correct += 1
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{n_eval} | running accuracy: {correct/(idx+1):.4f}")

acc = correct / n_eval
print(f"\nJCommonsenseQA ({NUM_FEWSHOT}-shot) accuracy: {acc:.4f} ({correct}/{n_eval})")
print(f"Random baseline (5-choice): 0.2000")
