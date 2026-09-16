"""Quick conversational generation test for the SFT'd model (depth=12, Japanese)."""
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("sft", device, phase="eval")
print(f"Loaded SFT checkpoint step: {meta.get('step')}, val_bpb: {meta.get('val_bpb')}")

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

engine = Engine(model, tokenizer)

prompts = [
    "日本の首都はどこですか?",
    "自己紹介をしてください。",
    "おすすめの日本食を教えてください。",
    "1+1は何ですか?",
    "富士山について教えてください。",
]

for p in prompts:
    conversation_tokens = [bos, user_start] + tokenizer.encode(p) + [user_end, assistant_start]
    sample, _ = engine.generate_batch(conversation_tokens, num_samples=1, max_tokens=80, temperature=0.7, top_k=50, seed=42)
    text = tokenizer.decode(sample[0])
    print("=" * 60)
    print(f"USER: {p}")
    print(f"ASSISTANT: {text}")
