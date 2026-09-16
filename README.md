# nanochat-ja

A from-scratch **Japanese-native** LLM (tokenizer → pretraining → SFT → chat), built on top of
[karpathy/nanochat](https://github.com/karpathy/nanochat), trained end-to-end on a **single
RTX 3090 (24GB)**.

> This is a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT License, see
> [`LICENSE`](LICENSE)). The core training/inference framework (tokenizer, GPT model, optimizer,
> data loader, engine, SFT loop) is upstream's; this fork adds a Japanese data pipeline, a
> from-scratch Japanese tokenizer, a Japanese SFT dataset, evaluation on a Japanese benchmark, and
> a couple of chat front-ends, all described below.

## Results

| Stage | Metric | Result |
|---|---|---|
| Tokenizer (65,536 vocab, trained on 1B chars of CC-100 ja) | Compression on Japanese text | 7.64 bytes/token, vs. 2.06 (GPT-2) / 2.85 (GPT-4) — **63–73% more efficient** |
| Base pretraining (depth=12, ~488M params) | Validation bits-per-byte | 0.818 → **0.801** after extending training (12 → 26 tokens/param) |
| Base model | [JCommonsenseQA](https://github.com/yahoo-japan/JGLUE) (2-shot) | **24.1%** (270/1119) vs. 20.0% random baseline |
| SFT (`llm-jp/magpie-sft-v1.0`, native Japanese) | Validation bits-per-byte | 1.205 → **0.451** |

Sample conversation after SFT:

```
User: 日本の首都はどこですか?
Assistant: 日本の首都は東京です。東京は2010年に天皇陛下が即位したことを契機に「首都令」という
新たな法律により東京都が定められました。...

User: おすすめの日本食を教えてください。
Assistant: もちろんです、以下にいくつかのおすすめの日本食を紹介します：
1. 寿司（すし）: 新鮮な魚介類を使用し、酢飯と新鮮な魚介類で巻いたものです。...
2. うどん: うどんは小麦粉と水で作った小麦粉の麺で、その独特の食感と軽快な味わいが特徴です。...
```

(A ~488M parameter model trained for ~17 GPU-hours on one consumer GPU will still hallucinate
dates and struggle with arithmetic — see [Design notes](#design-notes--what-i-learned) for an
honest account of what worked and what didn't.)

## Quickstart

No API keys or credentials are required anywhere in this pipeline — every dataset download below
is a public, anonymous request to Hugging Face.

```bash
uv sync --extra gpu   # or --extra cpu for CPU/MPS (will be very slow)
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"  # where data/checkpoints are cached
```

### 1. Pretraining data (Japanese CC-100)

```bash
python -m dev.prepare_cc100_ja --num-source-files 8   # ~16GB download, ~9.6B characters
```

Downloads shards from [`range3/cc100-ja`](https://huggingface.co/datasets/range3/cc100-ja) and
repackages them into the parquet shard format `nanochat/dataset.py` expects.

### 2. Tokenizer

```bash
python -m scripts.tok_train --vocab-size 65536 --max-chars 1000000000
python -m scripts.tok_eval
```

Trained from scratch on the Japanese corpus only. 1B characters was chosen after a 3B-character
run OOM'd during BPE pair-counting (see [Design notes](#design-notes--what-i-learned)).

### 3. Base pretraining

```bash
python -m scripts.base_train --depth=12 --device-batch-size=16
```

`--depth=12` picks a ~488M parameter model sized for a 24GB card; nanochat derives every other
hyperparameter (width, batch size, LR, weight decay, training horizon) from it. `--device-batch-size`
may need to go lower on smaller GPUs — see upstream's README for the OOM-tuning guidance this fork
inherited. To extend training with more of the downloaded corpus later:

```bash
python -m scripts.base_train --depth=12 --device-batch-size=16 \
  --resume-from-step=<last step> --target-param-data-ratio=26
```

### 4. Supervised fine-tuning (chat)

```bash
python -m scripts.chat_sft --total-batch-size=32768 --magpie-epochs=1
```

Uses [`llm-jp/magpie-sft-v1.0`](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0), a native
(not translated) Japanese instruction dataset. `--total-batch-size` is set explicitly because the
pretraining default (524,288 tokens) is sized for billions of pretraining tokens, not a ~130K-row
SFT set — see [Design notes](#design-notes--what-i-learned).

### 5. Talk to it

```bash
python -m scripts.chat_cli                 # interactive CLI (upstream)
python -m dev.chat_relay "こんにちは"        # single-shot, persists conversation state to disk
python -m dev.webapp                        # local web UI at http://127.0.0.1:8000
```

### Evaluation

```bash
python -m dev.eval_jcommonsenseqa           # JCommonsenseQA (Japanese commonsense QA), auto-downloads
python -m scripts.base_eval                 # upstream's CORE metric (English benchmarks; not very
                                             # informative for a Japanese-only model, included for completeness)
```

## Design notes / what I learned

**Why depth=12, not d20/d26 (upstream's GPT-2-grade target)?** Upstream's speedrun targets an
8×H100 node. On one RTX 3090, a compute budget calculation (peak BF16 FLOPs of the 3090 × available
time, divided by FLOPs/token at various depths) put the compute-optimal size around depth 9–12 for
a several-hour run — depth=12 was chosen because it's also nanochat's own internal reference depth
(the muP-style scaling formulas are calibrated against d12), keeping the run inside a well-tested
hyperparameter regime.

**Why train the tokenizer from scratch instead of reusing GPT-2/GPT-4's?** Those are optimized for
English; a from-scratch BPE tokenizer on the Japanese corpus alone compresses Japanese text far
better (see the Results table). Vocab size 65,536 (vs. upstream's default 32,768) was chosen
because Japanese's much larger character inventory benefits from a bigger vocabulary, and it's
still a clean power of two for nanochat's vocab-padding logic. The tradeoff: `value_embeds` and
`lm_head` scale with vocab size, so this roughly doubled their parameter count relative to the
default — worth it for the compression gain, but the resulting VRAM headroom is tighter as a
result (`--device-batch-size` had to drop from 32 to 16 on a 24GB card).

**Why CC-100 ja for pretraining?** There is no dataset actually named "Japanese SlimPajama" — that
name doesn't correspond to a real, independent Japanese corpus (SlimPajama itself is English-only,
occasionally used as one *ingredient* alongside separate Japanese corpora in other projects' data
mixes). [`range3/cc100-ja`](https://huggingface.co/datasets/range3/cc100-ja) — a parquet re-shard
of the Japanese portion of [CC-100](https://data.statmt.org/cc-100/) — was chosen instead: it's
large, in a format nanochat's loader accepts almost unmodified, and has real precedent in Japanese
LLM projects.

**Extending pretraining safely.** `scripts/base_train.py --resume-from-step` is designed for
resuming an *interrupted* run of the same schedule — its checkpoint carries the optimizer's
learning-rate state, which is correct to restore in that case. Resuming with a *larger*
`--target-param-data-ratio` to train further is a different situation: the checkpointed LR reflects
the old (already-decayed) schedule, not a fresh warm restart for the new one. Left as-is, extending
training this way would silently train at a near-zero learning rate for the entire extension. This
fork fixes it by snapshotting the freshly-computed LR before `optimizer.load_state_dict()` and
restoring it after (momentum buffers still load normally) — see the diff in `scripts/base_train.py`.

**SFT dataset: three iterations.**
1. First attempt used [`kunishou/databricks-dolly-15k-ja`](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja)
   (English Dolly, machine-translated; CC-BY-SA-3.0) with the pretraining batch size inherited
   unchanged (524,288 tokens) — the entire 14K-row dataset fit in ~5 optimizer steps, effectively
   no learning happened.
2. Fixed the batch size (32,768) and ran 4 epochs over the same 14K rows — training loss dropped
   sharply but validation bpb *rose* (1.04 → 1.26): classic overfitting on a small, repeated,
   narrow, translated dataset. Cutting back to 1 epoch fixed the overfitting (bpb 1.04 → 0.79) but
   generation quality was still inconsistent (e.g. answering the capital-of-Japan question
   incorrectly on one run).
3. Switched entirely to [`llm-jp/magpie-sft-v1.0`](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)
   (Apache-2.0): ~9x more rows, and *natively generated* Japanese (via the
   [Magpie](https://arxiv.org/abs/2406.08464) method — question and answer both generated by
   strong Japanese-capable models) rather than translated. One epoch (958 steps, ~9.5 minutes) took
   validation bpb from 1.20 to 0.45 and produced consistently correct, on-topic answers with no
   repetition-loop degeneration. `tasks/japanese_dolly.py` (the now-unused loader from step 1) was
   removed; `tasks/magpie_ja.py` is the one actually in use.

**Ampere-specific findings.** Flash Attention 3 (via the `kernels-community/flash-attn3` package)
works on this Ampere (SM 86) card, not just Hopper — worth checking on any card, since falling back
to SDPA meaningfully hurts throughput with nanochat's sliding-window attention pattern. Observed
steady-state throughput: ~58K tok/sec, ~74–75% MFU.

## Repository structure

Everything under `nanochat/`, `scripts/` (except the two noted below), `runs/`, and `tests/` is
upstream, unmodified. This fork's additions:

```
dev/
├── prepare_cc100_ja.py     # download + reshard range3/cc100-ja into nanochat's shard format
├── eval_jcommonsenseqa.py  # Japanese commonsense QA benchmark (auto-downloads its data)
├── gen_test.py             # quick multi-prompt generation smoke test (base model)
├── gen_test_sft.py         # same, for the chat/SFT model
├── chat_relay.py           # single-shot CLI chat with persisted conversation state
└── webapp.py               # minimal local Flask chat UI (binds to 127.0.0.1 only)
tasks/
└── magpie_ja.py            # Task wrapper for llm-jp/magpie-sft-v1.0
scripts/
├── base_train.py           # + fix: preserve fresh LR across --resume-from-step when the
│                            #   training horizon changes (see Design notes)
└── chat_sft.py             # + swapped the default English SFT mixture (SmolTalk/MMLU/GSM8K)
                             #   for tasks.magpie_ja.MagpieJa
```

## Datasets used (not included in this repo)

No corpora, checkpoints, or generated samples are committed here — everything above downloads its
own data on first run, cached under `$NANOCHAT_BASE_DIR`. Each dataset keeps its own upstream
license; check the source before any use beyond personal experimentation.

| Dataset | Used for | License |
|---|---|---|
| [`range3/cc100-ja`](https://huggingface.co/datasets/range3/cc100-ja) (derived from [CC-100](https://data.statmt.org/cc-100/)) | Pretraining | No additional restriction claimed by the preparer; subject to the [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use) |
| [`llm-jp/magpie-sft-v1.0`](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0) | SFT | Apache-2.0 |
| [`leemeng/jcommonsenseqa-v1.1`](https://huggingface.co/datasets/leemeng/jcommonsenseqa-v1.1) (from [JGLUE](https://github.com/yahoo-japan/JGLUE)) | Evaluation only | CC-BY-4.0 |
| [`kunishou/databricks-dolly-15k-ja`](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) | Early SFT experiment (superseded, see Design notes) | CC-BY-SA-3.0 |

## Upstream nanochat

The rest of this section is inherited context from upstream and still applies to the underlying
framework.

nanochat is Andrej Karpathy's minimal, single-GPU-node LLM training harness — tokenization,
pretraining, finetuning, evaluation, and inference in one hackable codebase, built around a single
complexity dial (`--depth`). See the [original repository](https://github.com/karpathy/nanochat)
for the full upstream README, the GPT-2-speedrun leaderboard, guides, and discussions.

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT, inherited from upstream — see [`LICENSE`](LICENSE). Training data and dataset licenses are
separate and listed above; they are not affected by this repository's own license.
