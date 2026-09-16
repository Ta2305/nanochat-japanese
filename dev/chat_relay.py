"""
Single-shot chat relay: send one user message to the SFT model and get the
assistant's reply, persisting conversation state (token history) to a file
across invocations so a multi-turn conversation can be driven by repeated
calls to this script.

Usage:
    python -m dev.chat_relay "こんにちは" [--reset] [--temperature 0.6] [--top-k 50]
"""
import argparse
import pickle
import os

from nanochat.common import compute_init, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

parser = argparse.ArgumentParser(description="Single-shot chat relay with persisted conversation state")
parser.add_argument("message", type=str, help="the user message to send")
parser.add_argument("--reset", action="store_true", help="start a fresh conversation instead of continuing")
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--max-tokens", type=int, default=300)
args = parser.parse_args()

state_path = os.path.join(get_base_dir(), "chat_relay_state.pkl")

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("sft", device, phase="eval")

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

if args.reset or not os.path.exists(state_path):
    conversation_tokens = [bos]
else:
    with open(state_path, "rb") as f:
        conversation_tokens = pickle.load(f)

conversation_tokens.append(user_start)
conversation_tokens.extend(tokenizer.encode(args.message))
conversation_tokens.append(user_end)
conversation_tokens.append(assistant_start)

engine = Engine(model, tokenizer)
response_tokens = []
for token_column, _ in engine.generate(conversation_tokens, num_samples=1, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k):
    response_tokens.append(token_column[0])

if not response_tokens or response_tokens[-1] != assistant_end:
    response_tokens.append(assistant_end)
conversation_tokens.extend(response_tokens)

with open(state_path, "wb") as f:
    pickle.dump(conversation_tokens, f)

reply_tokens = response_tokens[:-1] if response_tokens[-1] == assistant_end else response_tokens
print(tokenizer.decode(reply_tokens))
