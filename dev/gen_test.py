"""Quick generation test for the freshly pretrained base model (depth=12, Japanese)."""
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval")
print(f"Loaded checkpoint step: {meta.get('step')}, val_bpb: {meta.get('val_bpb')}")

engine = Engine(model, tokenizer)

prompts = [
    "日本の首都は",
    "今日の天気は",
    "人工知能とは",
    "むかしむかしあるところに、",
    "私の趣味は",
    "美味しい料理を作るには、まず",
    "日本で一番高い山は",
    "こんにちは、元気ですか?",
]

for p in prompts:
    tokens = tokenizer(p, prepend="<|bos|>")
    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=60, temperature=0.7, top_k=50, seed=42)
    text = tokenizer.decode(sample[0])
    print("=" * 60)
    print(f"PROMPT: {p}")
    print(f"OUTPUT: {text}")
