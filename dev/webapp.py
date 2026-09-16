"""
Minimal local web chat UI for the SFT model.

Loads the model once at startup and keeps a single global conversation
(this is a personal single-user tool, not a multi-tenant service).

Usage:
    python -m dev.webapp
Then access via an SSH tunnel:
    ssh -L 8000:localhost:8000 <user>@<this-server>
and open http://localhost:8000 in your browser.
"""
import os
from flask import Flask, request, jsonify, Response

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("sft", device, phase="eval")
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Single global conversation (personal tool, one user at a time)
conversation_tokens = [bos]

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanochat-ja</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, "Hiragino Sans", sans-serif; max-width: 760px; margin: 0 auto; padding: 1rem; background: Canvas; color: CanvasText; }
  h1 { font-size: 1.1rem; opacity: 0.7; font-weight: 500; }
  #chat { display: flex; flex-direction: column; gap: 0.6rem; margin-bottom: 5.5rem; }
  .msg { padding: 0.6rem 0.9rem; border-radius: 0.8rem; max-width: 80%; white-space: pre-wrap; line-height: 1.5; }
  .user { align-self: flex-end; background: #2563eb; color: white; }
  .assistant { align-self: flex-start; background: rgba(127,127,127,0.15); }
  .pending { opacity: 0.5; }
  #bar { position: fixed; bottom: 0; left: 0; right: 0; display: flex; gap: 0.5rem; padding: 0.8rem; background: Canvas; border-top: 1px solid rgba(127,127,127,0.3); }
  #inputRow { max-width: 760px; margin: 0 auto; display: flex; gap: 0.5rem; width: 100%; }
  #msgInput { flex: 1; padding: 0.6rem 0.8rem; border-radius: 0.6rem; border: 1px solid rgba(127,127,127,0.4); background: Field; color: FieldText; font-size: 1rem; }
  button { padding: 0.6rem 1rem; border-radius: 0.6rem; border: none; background: #2563eb; color: white; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  #resetBtn { background: transparent; color: inherit; opacity: 0.6; border: 1px solid rgba(127,127,127,0.4); }
</style>
</head>
<body>
<h1>nanochat-ja (depth=12, MagpieJa SFT)</h1>
<div id="chat"></div>
<div id="bar"><div id="inputRow">
  <input id="msgInput" placeholder="メッセージを入力..." autocomplete="off">
  <button id="sendBtn" onclick="send()">送信</button>
  <button id="resetBtn" onclick="doReset()">リセット</button>
</div></div>
<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('msgInput');
const sendBtn = document.getElementById('sendBtn');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chatEl.appendChild(div);
  window.scrollTo(0, document.body.scrollHeight);
  return div;
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMsg('user', text);
  const pending = addMsg('assistant', '...');
  pending.classList.add('pending');
  sendBtn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    pending.textContent = data.reply;
    pending.classList.remove('pending');
  } catch (e) {
    pending.textContent = 'エラーが発生しました: ' + e;
  }
  sendBtn.disabled = false;
  inputEl.focus();
}

async function doReset() {
  await fetch('/api/reset', {method: 'POST'});
  chatEl.innerHTML = '';
}

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

@app.route("/api/chat", methods=["POST"])
def chat():
    global conversation_tokens
    message = request.json.get("message", "")
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(message))
    conversation_tokens.append(user_end)
    conversation_tokens.append(assistant_start)

    response_tokens = []
    for token_column, _ in engine.generate(conversation_tokens, num_samples=1, max_tokens=300, temperature=0.6, top_k=50):
        response_tokens.append(token_column[0])

    if not response_tokens or response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    reply_tokens = response_tokens[:-1] if response_tokens[-1] == assistant_end else response_tokens
    reply = tokenizer.decode(reply_tokens)
    return jsonify({"reply": reply})

@app.route("/api/reset", methods=["POST"])
def reset():
    global conversation_tokens
    conversation_tokens = [bos]
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Listening on http://127.0.0.1:{port} (localhost only)")
    app.run(host="127.0.0.1", port=port, debug=False)
