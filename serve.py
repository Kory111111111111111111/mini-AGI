#!/usr/bin/env python3
"""
A chat window for talking to the model.

    python3 serve.py                 then open http://127.0.0.1:8080

Nothing but the conversation. Generation streams a character at a time, so
what you watch is the model writing rather than a spinner followed by a wall
of text - at this size that difference is most of the information.

Decoding is greedy and deterministic: the same conversation gives the same
reply twice. The loops greedy decoding falls into are held off by the
adaptation trace in minagi/decode.py, at the strength config.yaml sets.

IT LEARNS WHILE YOU TALK TO IT, and this CHANGES THE WEIGHTS ON DISK. The
exchange joins one continuous stream and the model takes an optimiser step
every `training.chunk` characters of it - the same path minagi/stream.py
takes through a corpus, which is why there is no second mechanism to reason
about. Read minagi/live.py for what it costs; it is not free, and on short
chat it is a far denser diet than documents. `--no-learn` serves read-only.

Binds to localhost.
"""

import argparse
import json
import os
import re
import sys
import threading
import time

# see train.py: the allocator reads this once at CUDA init, so it has to be
# set before torch loads
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from minagi.recur import load_any
from minagi.tokenizer import ByteTokenizer

app = Flask(__name__)

# One model, one conversation at a time. The lock is what makes that true:
# two overlapping requests would interleave their forward passes through the
# same KV caches and produce two corrupted replies instead of one good one.
LOCK = threading.Lock()
STATE = {"model": None, "tok": None, "weights": None, "learner": None}

U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"

# A passage of the corpus the conversation opens with. Empty when priming is
# off or no corpus is on disk.
PRIME = ""


def load_prime(chars, root="data/train/self-knowledge"):
    """
    Corpus text to open the conversation with, or "" if there is none.

    TWO THINGS IT BUYS, and they are separate. The reply has context to stand
    on: "hi" alone is two characters, and every character the model writes
    attends over what came before it and routes among the experts that whole
    context has voted for. And the learning stream starts part-way to its
    first step rather than twenty exchanges short of one.

    Taken from the self-knowledge lane because it is already in the register
    the chat is in - <user>/<bot> turns about what the model is - so this is
    text the model has read, not a preamble invented here and trained on.
    The `self-` files are the short question-and-answer ones; `auto-src-` are
    whole source files quoted back, and priming with one of those biases the
    router toward code and teaches the model to open by dumping a module.

    CUT WHERE THE MARKERS BALANCE, not at the last closing tag. This corpus
    documents its own format, so a file contains a code block reading
    "<bot>   </bot>" as prose - and those are real marker tokens, not
    spellings. Trimming to the last </bot> landed inside that block and
    returned a passage with an unclosed turn in it, which hands the model an
    open <bot> to answer inside before the conversation has started.

    Deterministic: the first file that yields a usable passage, every time. A
    passage picked at random would make the same conversation give different
    answers across restarts, and repeating a reply is worth more than variety.
    """
    if chars <= 0 or not os.path.isdir(root):
        return ""
    names = sorted(f for f in os.listdir(root) if f.endswith(".txt"))
    # conversational turns first, source dumps only if there are none
    names = ([n for n in names if n.startswith("self-")]
             + [n for n in names if not n.startswith("self-")])
    for name in names[:20]:
        with open(os.path.join(root, name), errors="replace") as f:
            text = f.read(chars * 3)
        best = ""
        at = text.find(f"{B1}\n")
        while 0 <= at < chars:
            end = at + len(B1) + 1
            head = text[:end]
            if (head.count(U0) == head.count(U1) == head.count(B0)
                    == head.count(B1)):
                best = head
            at = text.find(f"{B1}\n", end)
        if len(best) > chars // 3:
            return best
    return ""


def _manifest(path):
    """As loaded, so a save does not overwrite it with a blank one."""
    try:
        with open(os.path.join(path, "manifest.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load(weights, device=None, learn=True, lr=3e-4, save_every=8,
         chunk=None, resident=None):
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # read_only marks nothing dirty, so an expert paged in is never written
    # back. That is right for serving and wrong for learning - what the model
    # learned would live in VRAM until the slot was reused and then be gone.
    ro = not learn
    try:
        model, _ = load_any(weights, dev, read_only=ro, resident=resident)
    except torch.cuda.OutOfMemoryError:
        # training may be holding most of the card; serve slowly rather than
        # not at all
        torch.cuda.empty_cache()
        dev = torch.device("cpu")
        model, _ = load_any(weights, dev, read_only=ro)
        print("[warn] CUDA is full, serving on CPU", file=sys.stderr)
    model.eval()
    STATE.update(model=model, tok=ByteTokenizer(), weights=weights)
    print(f"[loaded] {weights} on {dev}", file=sys.stderr)

    # A paged model loads with an EMPTY card - every slot -1. A reply is
    # admitted by its own text's vote, so this does not change a reply; what
    # it changes is that the page can show experts on the card before the
    # first question instead of nothing. It is one forward over the priming
    # text, which admits the experts that text asks for, as any forward does.
    if PRIME:
        from minagi.precision import amp
        ids = STATE["tok"].encode(PRIME).ids[-model.cfg.block:]
        with torch.no_grad(), amp(dev):
            model(torch.tensor([ids], device=dev))

    if learn:
        from minagi.config import get as _g, load as _lc
        from minagi.live import LiveLearner
        c = _lc()
        STATE["learner"] = LiveLearner(
            model, lr=lr,
            trunk_lr_mult=_g(c, "training.trunk_lr_mult", 0.1),
            wd=_g(c, "training.weight_decay", 0.1),
            clip=_g(c, "training.clip", 1.0),
            # Defaults to the granularity the corpus is read at, so talking
            # to it and reading a book are one mechanism. That is the right
            # default and a poor fit for a short session: at 2048 an exchange
            # of ninety characters needs twenty-odd turns to reach one step.
            # --learn-chunk trades that against the denser diet live.py warns
            # about.
            chunk=chunk or _g(c, "training.chunk", 2048),
            save_every=save_every,
            weights_dir=weights,
            manifest=_manifest(weights))
        # The prime goes into the stream too, so the first step arrives
        # sooner. Kept SHORTER than a chunk on purpose: long enough to
        # advance the counter, short enough that it never takes a step by
        # itself, so restarting cannot train on the same passage repeatedly.
        if PRIME:
            STATE["learner"].feed(PRIME[-(STATE["learner"].chunk - 1):],
                                  STATE["tok"], note="prime")
        ch = STATE["learner"].chunk
        left = max(0, ch - STATE["learner"].pending)
        print(f"[learning] on - one step per {ch} characters, "
              f"{left} to the first (about {max(1, round(left / 93))} "
              f"exchanges), saved to {weights}/ every {save_every} steps",
              file=sys.stderr)
    return model


def resident_experts():
    """
    Which experts are on the card right now, by stable id - plus which ones
    the most recent forward actually routed through (`used`).

    `slots` is slot -> expert index, and the index is not an identity:
    pruning renumbers the pool, so the same index means different experts at
    different times. `uid` is what survives that, which is why it is what the
    page shows.

    `resident` is residency: every expert paged into VRAM, shaded by global
    gate weight. `used` is usage: the slots the last forward picked, read
    from each pool site's `last_route` (during generation each forward is
    one character, so that is what the last character chose; during prefill
    it is a union over the chunk). Pure telemetry - read-only, and any
    failure here falls back to residency alone rather than breaking serving.
    """
    model = STATE.get("model")
    pool = getattr(model, "pool", None)
    if pool is None or not hasattr(pool, "slots"):
        return None
    gate = pool.gate.detach().abs()
    top = float(gate.max().clamp_min(1e-9))
    slot_uid = {}
    out = []
    for j, e in enumerate(pool.slots):
        e = int(e)
        if e < 0:
            continue
        uid = int(pool.uid[e])
        slot_uid[j] = uid
        out.append({"slot": j, "uid": uid,
                    "gate": round(float(gate[e]) / top, 3)})
    used = []
    try:
        acc, n = None, 0
        for m in (model.modules() if model is not None else []):
            lr = getattr(m, "last_route", None)
            if lr is None:
                continue
            v = lr.detach().float().cpu()
            acc = v if acc is None else acc + v
            n += 1
        if n:
            share = (acc / n).tolist()
            ranked = sorted(((s, j) for j, s in enumerate(share)
                             if s > 1e-9), reverse=True)[:16]
            used = [slot_uid[j] for _, j in ranked if j in slot_uid]
    except Exception:                                   # noqa: BLE001
        used = []
    return {"resident": out, "n_experts": int(pool.n_experts()),
            "n_slots": len(pool.slots),
            "admitted": len(getattr(pool, "_admitted", ()) or ()),
            "used": used}


def learn_state():
    """How far the stream is from its next optimiser step."""
    lr_ = STATE.get("learner")
    if lr_ is None:
        return None
    return {"pending": int(lr_.pending), "chunk": int(lr_.chunk),
            "steps": int(lr_.steps)}


def build_prompt(messages, budget, prime="", system_prompt=""):
    """
    Build the conversation prompt with Streaming Attention Sink & clean boundary eviction.

    1. Anchor/Sink: Preserves priming and system persona instructions.
    2. Attention Sink eviction: If the conversation exceeds budget, drops older middle turns
       cleanly between message blocks rather than blind character chopping.
    3. Recency: Retains newest turns up to the remaining character budget.
    """
    turn_blocks = []

    # 1. System instructions if provided
    if system_prompt and system_prompt.strip():
        sp = system_prompt.strip()
        turn_blocks.append(f"{U0}\n[System Instructions]\n{sp}\n{U1}\n{B0}\nUnderstood.\n{B1}\n")

    # 2. Convert messages to clean turn blocks
    for m in messages:
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if m.get("role") == "user":
            turn_blocks.append(f"{U0}\n{text}\n{U1}\n")
        else:
            turn_blocks.append(f"{B0}\n{text}\n{B1}\n")

    if not turn_blocks:
        return prime + f"{B0}\n"

    suffix = f"{B0}\n"
    avail_budget = budget - len(prime) - len(suffix)

    # If all turns fit within budget, join everything directly
    total_len = sum(len(b) for b in turn_blocks)
    if total_len <= avail_budget:
        return prime + "".join(turn_blocks) + suffix

    # Otherwise apply Attention Sink Eviction:
    # Always keep the anchor block (first 1-2 blocks: system prompt or initial prompt)
    sink_blocks = []
    sink_len = 0
    max_sink_share = int(avail_budget * 0.35)
    for b in turn_blocks[:2]:
        if sink_len + len(b) <= max_sink_share:
            sink_blocks.append(b)
            sink_len += len(b)
        else:
            break

    remaining_budget = avail_budget - sink_len
    # Select most recent turns that fit in remaining budget from the end
    recent_blocks = []
    recent_len = 0
    candidates = turn_blocks[len(sink_blocks):]
    for b in reversed(candidates):
        if recent_len + len(b) <= remaining_budget:
            recent_blocks.append(b)
            recent_len += len(b)
        else:
            break
    recent_blocks.reverse()

    # Reassemble: sink + (clean omitted tag if turns were dropped) + recent + suffix
    dropped_count = len(candidates) - len(recent_blocks)
    omitted = f"{U0}\n[... {dropped_count} earlier turns omitted ...]\n{U1}\n{B0}\n...\n{B1}\n" if dropped_count > 0 else ""
    if omitted and sink_blocks and (sink_len + recent_len + len(omitted) <= avail_budget):
        prompt = prime + "".join(sink_blocks) + omitted + "".join(recent_blocks) + suffix
    else:
        prompt = prime + "".join(sink_blocks) + "".join(recent_blocks) + suffix

    # Fallback safety guard: if still slightly over budget, ensure clean slice at tag boundaries
    if len(prompt) > budget:
        prompt = prompt[-budget:]
        for tag in (f"{U0}\n", f"{B0}\n", f"{U1}\n", f"{B1}\n", "\n"):
            idx = prompt.find(tag)
            if 0 < idx < 35:
                prompt = prompt[idx:]
                break

    return prompt


@torch.no_grad()
def stream(prompt, max_new):
    from minagi.config import get as _g, load as _lc
    from minagi.decode import pick_next
    from minagi.precision import amp

    c = _lc()
    strength = _g(c, "decoding.adapt_strength", 2.5)
    decay = _g(c, "decoding.adapt_decay", 0.88)

    model, tok = STATE["model"], STATE["tok"]
    device = next(model.parameters()).device
    block = model.cfg.block
    ids = tok.encode(prompt).ids[-block:]
    out = torch.tensor([ids or [10]], device=device)

    # THE TEXT CHOOSES. The prompt is read in chunks, and each chunk adds its
    # characters' requests to the text's vote and is admitted the most-voted
    # experts; then every character of the reply is a forward of its own that
    # adds its requests and routes among what the whole text has voted for,
    # loading whatever is not already on the card. Nothing here chooses
    # experts - the forward does.
    pool = getattr(model, "pool", None)

    # Prefill in chunks, the way training reads a corpus. Feeding a long
    # prompt in one pass materialises activations for every position across
    # every block application at once, which is what puts a long context out
    # of reach; the cache carries the reach instead. Every forward computes
    # in the process-wide precision, as reading and model.generate do.
    CHUNK = 512
    caches = None

    def read(text):
        """`text` into fresh caches from position 0, a chunk at a time; the
        last chunk's logits. Position 0 begins a new text, so the vote that
        admits experts is cast by what is read here and nothing before it."""
        nonlocal caches
        caches = model.empty_caches()
        lg = None
        for i in range(0, text.shape[1], CHUNK):
            with amp(device):
                lg = model(text[:, i:i + CHUNK], caches=caches, pos_offset=i)[0]
        return lg

    logits = read(out)
    pos = out.shape[1]                       # where the next character sits

    cur = out[:, -1:]
    produced = []
    # The prompt's forwards have admitted its experts. Report them, so the
    # page starts the reply showing what is actually answering it.
    yield {"swap": {"at": 0, "moved": None, "pool": resident_experts()}}
    for i in range(max_new):
        loads = getattr(pool, "loads", 0)
        if logits is None:
            if pos + cur.shape[1] > block:
                # THE WINDOW IS FULL. A cached key keeps the rotary position
                # it was read at, so dropping the oldest key and placing the
                # next character at the end again - block-1, every time -
                # gives the newest characters one position between them, and
                # attention loses their order. Re-read the newest half of the
                # text from position 0 instead: what model.generate does, and
                # what every window of training is, a text from position 0.
                ctx = out[:, -(block // 2):]
                logits = read(ctx)
                pos = ctx.shape[1]
            else:
                with amp(device):
                    logits = model(cur, caches=caches, pos_offset=pos)[0]
                pos += cur.shape[1]
            moved = getattr(pool, "loads", 0) - loads
            if moved:
                # this character asked for experts that were not on the card;
                # `moved` counts the ones copied onto it
                yield {"swap": {"at": i, "moved": int(moved),
                                "pool": resident_experts()}}
        nxt = pick_next(logits[:, -1, :].float(), out, temperature=0.0,
                        adapt_strength=strength, adapt_decay=decay)
        out = torch.cat([out, nxt], dim=1)
        cur = nxt
        logits = None                       # spent; the next pass recomputes
        produced.append(int(nxt[0, 0]))
        text = tok.decode(produced)
        if text.endswith(B1):
            break
        yield {"t": tok.decode([produced[-1]])}


def remember(user_text, bot_text):
    """
    Read the finished exchange, the way the corpus reader reads a file.

    ONLY THIS TURN goes in. The prompt above carries the whole conversation,
    so re-feeding it every time would put the earliest turns through the
    stream once per exchange - re-reading rather than reading on, which is
    the dense regime minagi/live.py warns about.

    The caller already holds LOCK.
    """
    learner = STATE.get("learner")
    if learner is None:
        return None
    from minagi.live import exchange_text
    text = exchange_text(user_text, bot_text.replace(B1, ""))
    if not text:
        return None
    try:
        recs = learner.feed(text, STATE["tok"], note="chat")
        # Reported even when no step was taken. A step needs `chunk`
        # characters and an exchange is about ninety, so the first twenty or
        # so take none at all - and silence there is indistinguishable from
        # learning being broken, which is exactly the wrong thing to leave
        # ambiguous.
        out = {"steps": len(recs),
               "pending": learner.pending,
               "chunk": learner.chunk,
               "total": learner.steps}
        if recs:
            out["loss"] = round(float(recs[-1].get("loss", 0.0)), 4)
            if learner.due_to_save():
                learner.save()
                out["saved"] = True
        return out
    except Exception as e:                                    # noqa: BLE001
        # A failed step must not cost the reply that was already written.
        print(f"[learn] failed: {e}", file=sys.stderr)
        return {"error": str(e)}


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    model = STATE["model"]
    msgs = body.get("messages", [])
    system_prompt = body.get("system_prompt", "")
    prompt = build_prompt(msgs, int(model.cfg.block * 0.9), PRIME, system_prompt=system_prompt)
    max_new = int(body.get("max_new", 400))
    # the half of the exchange the model did not predict, which is where the
    # signal in a conversation is
    last_user = next((m.get("content") for m in reversed(msgs)
                      if m.get("role") == "user"), "")

    def events():
        t0, n = time.time(), 0
        reply = []
        learned = None
        try:
            with LOCK:
                for ev in stream(prompt, max_new):
                    if "t" in ev:
                        n += 1
                        reply.append(ev["t"])
                    yield f"data: {json.dumps(ev)}\n\n"
                learned = remember(last_user, "".join(reply))
        except torch.cuda.OutOfMemoryError:
            # Almost always a training run holding the card. Say so and stay
            # up: dying here closes the socket, and all the browser can tell
            # you then is that the fetch failed.
            torch.cuda.empty_cache()
            yield "data: " + json.dumps({"error":
                "the GPU is out of memory - something else is probably using "
                "it. Restart with --device cpu, or stop the other process."
                }) + "\n\n"
        except GeneratorExit:
            # Client closed stream or clicked Stop
            return
        except Exception as e:                            # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        dt = max(time.time() - t0, 1e-6)
        done = {"done": True, "n": n, "cps": round(n / dt, 1),
                "learn": learn_state(), "pool": resident_experts()}
        if learned:
            done["learned"] = learned
        yield "data: " + json.dumps(done) + "\n\n"

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/state")
def api_state():
    return jsonify({"learn": learn_state(),
                    "pool": resident_experts()})


@app.route("/api/prime")
def api_prime():
    return jsonify({"chars": len(PRIME),
                    "turns": PRIME.count(U0),
                    "text": PRIME})


CONV_DIR = os.path.join("runs", "conversations")
os.makedirs(CONV_DIR, exist_ok=True)


def parse_metrics():
    manifest_step = None
    manifest_experts = None
    manifest_path = os.path.join(STATE.get("weights") or "weights", "manifest.json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                mf = json.load(f)
                manifest_step = mf.get("step")
                manifest_experts = mf.get("cfg", {}).get("pool_max") or len(mf.get("experts", {}))
        except Exception:
            pass

    samples_path = os.path.join("runs", "samples.txt")
    entries = []
    samples = []
    if os.path.isfile(samples_path):
        try:
            with open(samples_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            blocks = text.split("==============================================================================")
            for i, raw_blk in enumerate(blocks):
                blk = raw_blk.strip()
                if not blk:
                    continue
                s_m = re.search(r"step\s+(\d+)\s+([\d\.]+)M\s+of\s+.*?(\d+)\s+min\s+(\d+)\s+experts", blk)
                if s_m:
                    t_m = re.search(r"train\s+loss\s+([\d\.]+)", blk)
                    h_m = re.search(r"held-out\s+loss\s+([\d\.]+)", blk)
                    b_m = re.search(r"([\d\.]+)\s+bits/char\s+perplexity\s+([\d\.]+)", blk)
                    c_m = re.search(r"reading\s+([\d\,]+)\s+char/s", blk)
                    entry = {
                        "step": int(s_m.group(1)),
                        "chars_m": float(s_m.group(2)),
                        "min": int(s_m.group(3)),
                        "experts": int(s_m.group(4)),
                        "train_loss": float(t_m.group(1)) if t_m else None,
                        "val_loss": float(h_m.group(1)) if h_m else None,
                        "bits_char": float(b_m.group(1)) if b_m else None,
                        "perplexity": float(b_m.group(2)) if b_m else None,
                        "cps": c_m.group(1).replace(",", "") if c_m else None
                    }
                    entries.append(entry)
                    if i + 1 < len(blocks):
                        nxt = blocks[i + 1].strip()
                        if nxt.startswith("---") or "prompt:" in nxt:
                            samples.append({"step": entry["step"], "sample": nxt})
        except Exception as e:
            print(f"[metrics] error parsing samples.txt: {e}", file=sys.stderr)

    return {
        "live_step": manifest_step,
        "live_experts": manifest_experts,
        "history": entries,
        "samples": samples
    }


@app.route("/api/metrics")
def api_metrics():
    return jsonify(parse_metrics())


@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    convs = []
    if os.path.isdir(CONV_DIR):
        for name in os.listdir(CONV_DIR):
            if name.endswith(".json"):
                fpath = os.path.join(CONV_DIR, name)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        convs.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Conversation"),
                            "timestamp": data.get("timestamp", ""),
                            "step": data.get("step"),
                            "experts": data.get("experts"),
                            "turns": len(data.get("messages", [])) // 2
                        })
                except Exception:
                    pass
    convs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(convs)


@app.route("/api/conversations/<cid>", methods=["GET"])
def api_get_conversation(cid):
    fpath = os.path.join(CONV_DIR, f"{cid}.json")
    if os.path.isfile(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "not found"}), 404


@app.route("/api/conversations", methods=["POST"])
def api_save_conversation():
    data = request.get_json(force=True)
    cid = data.get("id")
    if not cid:
        cid = f"conv_{int(time.time()*1000)}"
        data["id"] = cid
    if not data.get("timestamp"):
        data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fpath = os.path.join(CONV_DIR, f"{cid}.json")
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return jsonify({"success": True, "id": cid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversations/<cid>", methods=["DELETE"])
def api_delete_conversation(cid):
    fpath = os.path.join(CONV_DIR, f"{cid}.json")
    if os.path.isfile(fpath):
        try:
            os.remove(fpath)
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "not found"}), 404


@app.route("/runs/<path:filename>")
def serve_runs(filename):
    return send_from_directory("runs", filename)


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-AGI</title>
<style>
  :root{
    --bg:#ffffff; --fg:#1f2328; --dim:#6b7280; --line:#e5e7eb;
    --user:#f3f4f6; --accent:#2563eb; --accent-hover:#1d4ed8;
    --err:#dc2626; --ok:#16a34a; --glow:rgba(37,99,235,.28);
    --card-bg:#f9fafb; --sidebar-bg:#f8fafc;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0d1117; --fg:#e6edf3; --dim:#8b949e; --line:#21262d;
           --user:#161b22; --accent:#58a6ff; --accent-hover:#79b8ff;
           --err:#f85149; --ok:#3fb950; --glow:rgba(88,166,255,.30);
           --card-bg:#161b22; --sidebar-bg:#0d1117; }
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:row;overflow:hidden}

  /* SIDEBAR: CONVERSATION ARCHIVE */
  #sidebar{width:270px;background:var(--sidebar-bg);border-right:1px solid var(--line);
           display:flex;flex-direction:column;flex-shrink:0;transition:transform .2s ease, width .2s ease;
           z-index:20}
  #sidebar.collapsed{width:0;overflow:hidden;border-right:none}
  .sb-head{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
  .sb-head b{font-size:13px;font-weight:600;letter-spacing:.3px}
  .sb-btn{background:none;border:1px solid var(--line);border-radius:6px;padding:4px 9px;
          font-size:12px;color:var(--fg);cursor:pointer}
  .sb-btn:hover{border-color:var(--accent);color:var(--accent)}
  #conv-list{flex:1;overflow-y:auto;padding:8px}
  .conv-item{padding:9px 10px;border-radius:6px;cursor:pointer;display:flex;
             flex-direction:column;gap:3px;margin-bottom:4px;border:1px solid transparent;
             transition:background .15s}
  .conv-item:hover{background:var(--user)}
  .conv-item.active{background:var(--user);border-color:var(--line)}
  .conv-title{font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .conv-meta{font-size:11px;color:var(--dim);display:flex;justify-content:space-between;align-items:center}
  .conv-badge{font-size:10px;padding:1px 5px;border-radius:4px;background:var(--line);color:var(--dim)}
  .conv-del{background:none;border:none;color:var(--dim);font-size:12px;cursor:pointer;padding:0 2px;opacity:0}
  .conv-item:hover .conv-del{opacity:1}
  .conv-del:hover{color:var(--err)}

  /* MAIN CONTENT AREA */
  #main{flex:1;display:flex;flex-direction:column;height:100%;overflow:hidden;position:relative}

  /* HEADER */
  header{border-bottom:1px solid var(--line);padding:10px 16px;
         display:flex;align-items:center;gap:12px;flex:0 0 auto;background:var(--bg)}
  .brand-area{display:flex;align-items:center;gap:8px}
  header b{font-weight:600;font-size:15px}
  .pulse-dot{width:8px;height:8px;border-radius:50%;background:var(--ok);display:inline-block;box-shadow:0 0 6px var(--ok)}
  .nav-tabs{display:flex;gap:4px;margin-left:14px;background:var(--user);padding:3px;border-radius:8px}
  .tab{font-size:12px;padding:4px 10px;border:none;border-radius:5px;cursor:pointer;background:none;color:var(--dim);font-weight:500;display:inline-flex;align-items:center;gap:5px}
  .ic{width:14px;height:14px;flex:none}
  .tab.active{background:var(--bg);color:var(--fg);box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .sp{flex:1}
  .hdr-btn{font-size:12px;color:var(--dim);text-decoration:none;padding:5px 9px;border-radius:6px;border:1px solid var(--line);cursor:pointer;background:none;display:inline-flex;align-items:center;gap:5px}
  .hdr-btn:hover{color:var(--fg);border-color:var(--accent)}

  /* MECHANISMS PANEL */
  #mech{border-bottom:1px solid var(--line);padding:8px 16px;flex:0 0 auto;background:var(--bg)}
  .mrow{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:6px}
  .roster-top{font-size:11px;color:var(--dim);display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
  .roster-top b{font-weight:600;color:var(--fg);font-variant-numeric:tabular-nums}
  #roster{display:flex;gap:2px;height:9px}
  #roster:empty{display:none}
  .seg{flex:1 1 0;min-width:0;border-radius:2px;background:var(--accent);transition:opacity .15s, transform .15s}
  .seg.use{box-shadow:0 0 0 1px var(--accent), 0 0 8px var(--accent);transform:scaleY(1.35);z-index:2}
  .seg.new{animation:segflash 1.4s ease-out}
  .seg.off-card{background:rgba(120,130,150,0.18);border:1px solid rgba(255,255,255,0.08);opacity:0.25}
  .seg.off-card:hover{opacity:0.6}
  @keyframes segflash{0%{filter:brightness(2.2);transform:scaleY(1.6);background:#fff}100%{filter:none;transform:none}}
  .mem-line{height:2px;border-radius:1px;background:var(--line);overflow:hidden}
  #mem-f{height:100%;width:0;border-radius:1px;background:var(--ok);transition:width .18s linear}
  .note{font-size:11px;color:var(--ok);opacity:0;transition:opacity .2s}
  .note.on{opacity:1}
  .note.swap{color:var(--accent)}


  /* VIEW 1: CHAT VIEW */
  #chat-view{flex:1;display:flex;flex-direction:column;overflow:hidden}
  #log{flex:1 1 auto;overflow-y:auto;padding:20px 16px}
  .wrap{max-width:740px;margin:0 auto}
  .msg{margin:0 0 18px}
  .who{font-size:12px;color:var(--dim);margin-bottom:4px;display:flex;justify-content:space-between}
  .body{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;line-height:1.6}
  .user .body{background:var(--user);padding:10px 14px;border-radius:10px;display:inline-block;max-width:92%}
  .bot .body{background:transparent;padding:4px 0}
  .err{color:var(--err)}
  .caret::after{content:"\258B";color:var(--accent);animation:blink 1s step-end infinite}
  @keyframes blink{50%{opacity:0}}
  .empty{color:var(--dim);text-align:center;margin-top:16vh;line-height:1.9}
  .empty code{background:var(--user);padding:2px 6px;border-radius:4px;font-size:12px}

  /* CHAT FOOTER */
  footer{border-top:1px solid var(--line);padding:14px 16px;flex:0 0 auto;background:var(--bg)}
  form{max-width:740px;margin:0 auto;display:flex;flex-direction:column;gap:6px}
  .input-wrap{display:flex;align-items:flex-end;gap:12px;background:var(--user);border:1px solid var(--line);
              border-radius:14px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.04);
              transition:border-color .15s,box-shadow .15s,background .15s}
  .input-wrap:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px var(--glow);background:var(--bg)}
  textarea{flex:1;resize:none;font:inherit;font-size:14px;color:var(--fg);background:transparent;border:none;
           outline:none;padding:2px 0;margin:0;line-height:1.5;min-height:56px;max-height:180px;
           overflow-y:hidden;scrollbar-width:thin;scrollbar-color:var(--dim) transparent}
  textarea::-webkit-scrollbar{width:4px}
  textarea::-webkit-scrollbar-track{background:transparent}
  textarea::-webkit-scrollbar-thumb{background:var(--line);border-radius:10px}
  textarea::-webkit-scrollbar-button{display:none;width:0;height:0}
  .input-actions{display:flex;align-items:center;gap:6px;flex-shrink:0}
  .btn-send{height:38px;padding:0 16px;background:var(--accent);color:#fff;border:none;border-radius:10px;
            font-weight:600;font-size:13px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;
            transition:background .15s,transform .1s;box-shadow:0 1px 2px rgba(0,0,0,.08)}
  .btn-send:hover{background:var(--accent-hover)}
  .btn-send:active{transform:scale(.97)}
  .btn-send:disabled{opacity:.4;cursor:not-allowed}
  .btn-stop{height:38px;padding:0 14px;background:var(--err);color:#fff;border:none;border-radius:10px;
            font-weight:600;font-size:13px;cursor:pointer;display:none;align-items:center;gap:6px;
            transition:opacity .15s,transform .1s;box-shadow:0 1px 2px rgba(0,0,0,.08)}
  .btn-stop:hover{opacity:.9}
  .btn-stop:active{transform:scale(.97)}
  .stat{max-width:740px;margin:4px auto 0;font-size:11px;color:var(--dim);display:flex;justify-content:space-between}

  /* VIEW 2: LIVE METRICS DASHBOARD */
  #metrics-view{flex:1;overflow-y:auto;padding:20px 24px;display:none}
  .dash-wrap{max-width:980px;margin:0 auto}
  .dash-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}
  .dash-top h2{margin:0;font-size:18px;font-weight:600}
  .dash-ctrls{display:flex;gap:10px;align-items:center;font-size:12px;color:var(--dim)}

  /* KPI CARDS GRID */
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:24px}
  .kpi-card{background:var(--card-bg);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .kpi-label{font-size:11px;text-transform:uppercase;color:var(--dim);letter-spacing:.5px;font-weight:600;margin-bottom:6px}
  .kpi-val{font-size:24px;font-weight:700;letter-spacing:-.5px}
  .kpi-sub{font-size:12px;color:var(--dim);margin-top:4px}

  /* CHARTS SECTION */
  .chart-box{background:var(--card-bg);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin-bottom:20px}
  .chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
  .chart-head h3{margin:0;font-size:14px;font-weight:600}
  .chart-legend{display:flex;gap:14px;font-size:11px;color:var(--dim)}
  .legend-dot{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:4px}
  .svg-chart{width:100%;height:auto;display:block}

  /* CHECKPOINTS LOG / SAMPLES */
  .sample-box{background:var(--card-bg);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin-bottom:20px}
  .sample-card{border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-top:10px;background:var(--bg)}
  .sample-step{font-size:12px;font-weight:600;color:var(--accent);margin-bottom:6px}
  .sample-text{font-size:12px;font-family:monospace;white-space:pre-wrap;color:var(--dim);line-height:1.5}
</style></head>
<body>

<!-- SIDEBAR -->
<div id="sidebar">
  <div class="sb-head">
    <b>Past Chats</b>
    <button class="sb-btn" id="new-chat-btn">+ New</button>
  </div>
  <div id="conv-list"></div>
</div>

<!-- MAIN CONTAINER -->
<div id="main">
  <header>
    <div class="brand-area">
      <button class="hdr-btn" id="toggle-sidebar" title="Toggle history sidebar"><svg class="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><rect x="1.8" y="2.5" width="12.4" height="11" rx="2"/><line x1="6" y1="2.5" x2="6" y2="13.5"/><circle cx="3.9" cy="5.2" r="1.15" fill="currentColor" stroke="none"/></svg></button>
      <b>mini-AGI</b>
      <span class="pulse-dot" id="live-dot" title="Training engine active"></span>
    </div>

    <div class="nav-tabs">
      <button class="tab active" id="tab-chat"><svg class="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 3.2c0-.7.5-1.2 1.2-1.2h8.6c.7 0 1.2.5 1.2 1.2v6c0 .7-.5 1.2-1.2 1.2H8.2L4.9 13V10.4H3.7c-.7 0-1.2-.5-1.2-1.2z"/><circle cx="6.4" cy="6.2" r="1.05" fill="currentColor" stroke="none"/><circle cx="9.9" cy="6.2" r="1.05" fill="currentColor" stroke="none"/></svg>Chat</button>
      <button class="tab" id="tab-metrics"><svg class="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><line x1="2" y1="13.5" x2="14" y2="13.5"/><line x1="4.5" y1="9.5" x2="4.5" y2="13.5"/><line x1="8" y1="7" x2="8" y2="13.5"/><line x1="11.5" y1="5.6" x2="11.5" y2="13.5"/><circle cx="11.5" cy="3.4" r="1.2" fill="currentColor" stroke="none"/></svg>Live Metrics</button>
    </div>

    <span class="sp"></span>
    <a href="/runs/dashboard.png" target="_blank" class="hdr-btn">Export PNG</a>
    <button class="hdr-btn" id="reset">Reset View</button>
  </header>


  <!-- MECHANISMS BAR -->
  <div id="mech">
    <div class="mrow">
      <div class="roster-top"><span>consulting <b id="exp-n">-</b></span><span id="mem-n">-</span></div>
      <div id="roster"></div>
      <div class="mem-line"><div id="mem-f"></div></div>
      <span class="note" id="note"></span>
    </div>
  </div>

  <!-- CHAT VIEW -->
  <div id="chat-view">
    <div id="log"><div class="wrap"><div id="primebox"></div><div id="thread"></div></div></div>
    <footer>
      <form id="f">
        <div class="input-wrap">
          <textarea id="q" placeholder="Send a message..."></textarea>
          <div class="input-actions">
            <button id="send" class="btn-send" type="submit" title="Send message">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
              <span>Send</span>
            </button>
            <button id="stop" class="btn-stop" type="button" title="Stop generating">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
              <span>Stop</span>
            </button>
          </div>
        </div>
      </form>
      <div class="stat"><span id="stat"></span></div>
    </footer>
  </div>

  <!-- LIVE METRICS DASHBOARD VIEW -->
  <div id="metrics-view">
    <div class="dash-wrap">
      <div class="dash-top">
        <h2>Live Training Metrics & Dashboard</h2>
        <div class="dash-ctrls">
          <label><input type="checkbox" id="auto-refresh" checked> Auto-refresh (5s)</label>
          <button class="hdr-btn" id="btn-refresh"><svg class="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13.6 8a5.6 5.6 0 1 1-1.7-4"/><polyline points="13.8,1.6 13.6,4.9 10.3,4.7"/></svg>Refresh</button>
        </div>
      </div>

      <!-- KPI CARDS -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Current Step</div>
          <div class="kpi-val" id="kpi-step">-</div>
          <div class="kpi-sub" id="kpi-step-sub">Continual learning</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Expert Pool Capacity</div>
          <div class="kpi-val" id="kpi-exp">-</div>
          <div class="kpi-sub" id="kpi-exp-sub">Dynamic allocation</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Latest Held-Out Loss</div>
          <div class="kpi-val" id="kpi-val-loss" style="color:var(--ok)">-</div>
          <div class="kpi-sub" id="kpi-val-sub">Generalization benchmark</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Train Loss & Perplexity</div>
          <div class="kpi-val" id="kpi-trn-loss" style="color:var(--accent)">-</div>
          <div class="kpi-sub" id="kpi-perp-sub">Active batch error</div>
        </div>
      </div>

      <!-- INTERACTIVE CHARTS -->
      <div class="chart-box">
        <div class="chart-head">
          <h3>Training vs Held-Out Loss Trajectory</h3>
          <div class="chart-legend">
            <span><span class="legend-dot" style="background:#3b82f6"></span> Train Loss</span>
            <span><span class="legend-dot" style="background:#10b981"></span> Held-out Loss</span>
          </div>
        </div>
        <div id="loss-chart-box"></div>
      </div>

      <div class="chart-box">
        <div class="chart-head">
          <h3>Expert Pool Growth Over Time</h3>
          <div class="chart-legend">
            <span><span class="legend-dot" style="background:#8b5cf6"></span> Active Pool Experts</span>
          </div>
        </div>
        <div id="experts-chart-box"></div>
      </div>

      <!-- SAMPLES PROGRESSION -->
      <div class="sample-box">
        <div class="chart-head">
          <h3>Model Output Quality by Checkpoint</h3>
          <span style="font-size:11px;color:var(--dim)">Greedy generation on held-out prompt</span>
        </div>
        <div id="samples-container"></div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const thread = $('thread'), log = $('log'), q = $('q'), stat = $('stat');
const send = $('send'), stopBtn = $('stop');
let messages = [];
let busy = false;
let currentReader = null;
let abortCtrl = null;
let activeConvId = 'conv_' + Date.now();
let latestMetricsData = null;

let MECH = {pending:0, chunk:0, steps:0, admitted:0, slots:0, uids:[], loaded:0};

// SIDEBAR TOGGLE
$('toggle-sidebar').onclick = () => $('sidebar').classList.toggle('collapsed');
$('new-chat-btn').onclick = () => startNewChat();

// TAB SWITCHING
$('tab-chat').onclick = () => switchTab('chat');
$('tab-metrics').onclick = () => switchTab('metrics');

function switchTab(t){
  if (t === 'chat'){
    $('tab-chat').classList.add('active');
    $('tab-metrics').classList.remove('active');
    $('chat-view').style.display = 'flex';
    $('metrics-view').style.display = 'none';
  } else {
    $('tab-metrics').classList.add('active');
    $('tab-chat').classList.remove('active');
    $('chat-view').style.display = 'none';
    $('metrics-view').style.display = 'block';
    fetchMetrics();
  }
}

// MECHANISM VISUALIZATIONS
function note(text, kind){
  const n = $('note'); n.textContent = text;
  n.className = 'note on' + (kind ? ' ' + kind : '');
  clearTimeout(note._t);
  note._t = setTimeout(() => { n.className = 'note'; }, 2600);
}
function flare(el){ el.classList.remove('fired'); void el.offsetWidth; el.classList.add('fired'); }

function drawMem(){
  const {pending, chunk} = MECH;
  if (!chunk){ $('mem-n').textContent = 'off'; $('mem-f').style.width = '0'; return; }
  const left = Math.max(0, chunk - pending);
  $('mem-n').textContent = left ? 'remembering in ' + left + ' chars'
                                : (busy ? 'updates when reply finishes' : 'ready to update');
  $('mem-f').style.width = (100 * Math.min(pending, chunk) / chunk) + '%';
}
function drawExp(){
  const {admitted, slots, total} = MECH;
  if (!slots){ $('exp-n').textContent = '-'; return; }
  const poolTotal = total || slots;
  $('exp-n').innerHTML = `<b>${admitted}</b> resident of <b>${poolTotal}</b> pool experts <span style="font-weight:normal;color:var(--dim)">(${slots} VRAM slots)</span>`;
}
function drawChips(pool, changed){
  if (!pool) return;
  const used = new Set(pool.used || []);
  const box = $('roster'); box.innerHTML = '';
  const total = pool.n_experts || pool.n_slots || 0;
  const resByUid = new Map();
  for (const e of (pool.resident || [])){
    resByUid.set(e.uid, e);
  }
  for (let uid = 0; uid < total; uid++){
    const s = document.createElement('div');
    const isResident = resByUid.has(uid);
    const isUsed = used.has(uid);
    const isSwapped = changed && changed.has(uid);
    const e = resByUid.get(uid);

    if (isResident){
      s.className = 'seg' + (isSwapped ? ' new' : '') + (isUsed ? ' use' : '');
      s.style.opacity = isUsed ? 1 : (0.28 + 0.72 * (e.gate || 0)).toFixed(2);
      s.title = `Expert #${uid} (Slot ${e.slot}) · Gate: ${e.gate}${isUsed ? ' · ACTIVE (routed by last token)' : ' · Resident in VRAM'}${isSwapped ? ' · JUST SWAPPED IN' : ''}`;
    } else {
      s.className = 'seg off-card';
      s.title = `Expert #${uid} · Paged on disk (swaps into VRAM when routed)`;
    }
    box.appendChild(s);
  }
  if (changed && changed.size) setTimeout(() => {
    box.querySelectorAll('.seg.new').forEach(s => s.classList.remove('new'));
  }, 1400);
}

function applyState(d){
  if (d.learn){ MECH.pending = d.learn.pending; MECH.chunk = d.learn.chunk; MECH.steps = d.learn.steps; }
  if (d.pool){
    MECH.admitted = d.pool.admitted || 0; MECH.slots = d.pool.n_slots || 0;
    MECH.total = d.pool.n_experts || d.pool.n_slots || 0;
    const seen = new Set(MECH.uids);
    const now = (d.pool.resident || []).map(e => e.uid);
    const changed = MECH.uids.length ? new Set(now.filter(u => !seen.has(u))) : null;
    MECH.uids = now;
    drawChips(d.pool, changed);
  }
  drawMem(); drawExp();
}

fetch('/api/state').then(r => r.json()).then(applyState).catch(() => {});
fetch('/api/prime').then(r => r.json()).then(p => {
  if (!p.chars) return;
  const d = document.createElement('details'); d.className = 'prime';
  d.innerHTML = '<summary>primed with ' + p.chars + ' chars</summary><pre>' + p.text + '</pre>';
  $('primebox').appendChild(d);
}).catch(() => {});

// CONVERSATION ARCHIVING
async function loadConversationsList(){
  try {
    const res = await fetch('/api/conversations');
    const list = await res.json();
    const box = $('conv-list'); box.innerHTML = '';
    if (!list.length){ box.innerHTML = '<div style="font-size:11px;color:var(--dim);padding:8px">No archived chats yet.</div>'; return; }
    list.forEach(c => {
      const it = document.createElement('div');
      it.className = 'conv-item' + (c.id === activeConvId ? ' active' : '');
      it.innerHTML = `
        <div class="conv-title">${escapeHtml(c.title || 'Conversation')}</div>
        <div class="conv-meta">
          <span>${c.timestamp ? c.timestamp.split(' ')[0] : ''}</span>
          <span class="conv-badge">Step ${c.step || '-'}</span>
          <button class="conv-del" title="Delete">✕</button>
        </div>
      `;
      it.onclick = (e) => {
        if (e.target.classList.contains('conv-del')) {
          e.stopPropagation();
          deleteConversation(c.id);
        } else {
          loadConversation(c.id);
        }
      };
      box.appendChild(it);
    });
  } catch (e) {}
}

async function loadConversation(id){
  if (busy) return;
  try {
    const res = await fetch('/api/conversations/' + id);
    const data = await res.json();
    activeConvId = data.id;
    messages = data.messages || [];
    renderThread();
    loadConversationsList();
  } catch (e) {}
}

async function saveCurrentConversation(){
  if (!messages.length) return;
  const firstUser = messages.find(m => m.role === 'user');
  const title = firstUser ? firstUser.content.slice(0, 36) : 'Chat';
  const payload = {
    id: activeConvId,
    title: title,
    messages: messages,
    step: latestMetricsData ? latestMetricsData.live_step : null,
    experts: latestMetricsData ? latestMetricsData.live_experts : null
  };
  try {
    await fetch('/api/conversations', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    loadConversationsList();
  } catch (e) {}
}

async function deleteConversation(id){
  try {
    await fetch('/api/conversations/' + id, {method: 'DELETE'});
    if (activeConvId === id) startNewChat();
    else loadConversationsList();
  } catch (e) {}
}

function startNewChat(){
  if (busy) return;
  activeConvId = 'conv_' + Date.now();
  messages = [];
  renderThread();
  loadConversationsList();
  q.focus();
}

function renderThread(){
  thread.innerHTML = '';
  if (!messages.length){ empty(); return; }
  messages.forEach(m => add(m.role, m.content));
}

function escapeHtml(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function empty(){
  thread.innerHTML = '<div class="empty">Ask it something.<br>mini-AGI is continuously learning from the corpus.</div>';
}
function add(role, text){
  if (thread.querySelector('.empty')) thread.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = `<div class="who">${role === 'user' ? 'You' : 'mini-AGI'}</div><div class="body"></div>`;
  d.querySelector('.body').textContent = text || '';
  thread.appendChild(d);
  return d.querySelector('.body');
}

function grow(){
  q.style.height = '0px';
  const targetH = Math.max(54, Math.min(q.scrollHeight, 180));
  q.style.height = targetH + 'px';
  q.style.overflowY = q.scrollHeight > 180 ? 'auto' : 'hidden';
}
q.addEventListener('input', grow);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); $('f').requestSubmit(); }
});

$('reset').onclick = () => startNewChat();

// STOP GENERATION
stopBtn.onclick = () => {
  if (!busy) return;
  if (currentReader) { try { currentReader.cancel(); } catch(e){} }
  if (abortCtrl) { try { abortCtrl.abort(); } catch(e){} }
  stat.textContent = 'Generation stopped';
};

// FORM SUBMISSION
$('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true; send.style.display = 'none'; stopBtn.style.display = 'inline-block';
  q.value = ''; grow();

  add('user', text);
  messages.push({role:'user', content:text});
  const body = add('bot', '');
  body.classList.add('caret');
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
  abortCtrl = new AbortController();
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages, max_new: 400}),
      signal: abortCtrl.signal
    });
    currentReader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for(;;){
      const {value, done} = await currentReader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const frames = buf.split('\n\n');
      buf = frames.pop();
      for (const f of frames){
        if (!f.startsWith('data: ')) continue;
        const d = JSON.parse(f.slice(6));
        if (d.t){
          reply += d.t;
          body.textContent = reply;
          log.scrollTop = log.scrollHeight;
          MECH.pending++;
          drawMem();
        } else if (d.swap){
          applyState(d);
          if (d.swap.moved){
            flare($('mech'));
            note(d.swap.moved + (d.swap.moved === 1 ? ' expert swapped from disk' : ' experts swapped from disk'), 'swap');
          }
        } else if (d.error){
          body.classList.add('err');
          body.textContent = reply + '\n\n[' + d.error + ']';
        } else if (d.done){
          applyState(d);
          if (d.learned && d.learned.steps){
            note(d.learned.steps + (d.learned.steps === 1 ? ' weight update' : ' weight updates'), 'swap');
          }
          stat.textContent = `${d.n} chars, ${d.cps}/s`;
        }
      }
    }
  } catch (err) {
    if (String(err).includes('AbortError')) {
      reply += ' [stopped]';
      body.textContent = reply;
    } else {
      body.classList.add('err');
      body.textContent = reply + '\n\n[' + err + ']';
    }
  }
  body.classList.remove('caret');
  messages.push({role:'bot', content:reply});
  busy = false; send.style.display = 'inline-block'; stopBtn.style.display = 'none';
  q.focus();
  saveCurrentConversation();
};

// LIVE METRICS & CHARTS RENDERING
async function fetchMetrics(){
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    latestMetricsData = data;
    renderMetrics(data);
  } catch (e) {}
}

function renderMetrics(data){
  const hist = data.history || [];
  const latest = hist.length ? hist[hist.length - 1] : null;

  $('kpi-step').textContent = data.live_step != null ? data.live_step.toLocaleString() : (latest ? latest.step.toLocaleString() : '-');
  $('kpi-exp').textContent = data.live_experts != null ? data.live_experts + ' experts' : (latest ? latest.experts + ' experts' : '-');
  if (latest){
    $('kpi-val-loss').textContent = latest.val_loss != null ? latest.val_loss.toFixed(4) : '-';
    $('kpi-val-sub').textContent = `${latest.bits_char ? latest.bits_char.toFixed(2) + ' bits/char' : ''}`;
    $('kpi-trn-loss').textContent = latest.train_loss != null ? latest.train_loss.toFixed(4) : '-';
    $('kpi-perp-sub').textContent = `Perplexity: ${latest.perplexity || '-'}`;
  }

  renderLossChart(hist);
  renderExpertsChart(hist, data.live_step, data.live_experts);
  renderSamples(data.samples || []);
}

function renderLossChart(hist){
  const box = $('loss-chart-box');
  if (!hist.length){ box.innerHTML = '<div style="color:var(--dim);font-size:12px;padding:20px 0">Waiting for training checkpoints...</div>'; return; }

  const W = 700, H = 220;
  const padL = 45, padR = 25, padT = 20, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const minStep = hist[0].step;
  const maxStep = Math.max(hist[hist.length - 1].step, minStep + 1);

  let allLoss = [];
  hist.forEach(h => {
    if (h.train_loss) allLoss.push(h.train_loss);
    if (h.val_loss) allLoss.push(h.val_loss);
  });
  const minLoss = Math.max(0, Math.min(...allLoss) - 0.2);
  const maxLoss = Math.max(...allLoss) + 0.2;

  const x = s => padL + ((s - minStep) / (maxStep - minStep)) * plotW;
  const y = l => padT + (1 - (l - minLoss) / (maxLoss - minLoss)) * plotH;

  let trainPts = [], valPts = [];
  hist.forEach(h => {
    if (h.train_loss != null) trainPts.push(`${x(h.step).toFixed(1)},${y(h.train_loss).toFixed(1)}`);
    if (h.val_loss != null) valPts.push(`${x(h.step).toFixed(1)},${y(h.val_loss).toFixed(1)}`);
  });

  let grid = '';
  for (let i = 0; i <= 4; i++){
    const val = minLoss + (i / 4) * (maxLoss - minLoss);
    const yPos = y(val);
    grid += `<line x1="${padL}" y1="${yPos}" x2="${W - padR}" y2="${yPos}" stroke="var(--line)" stroke-dasharray="3,3" />`;
    grid += `<text x="${padL - 8}" y="${yPos + 4}" font-size="10" fill="var(--dim)" text-anchor="end">${val.toFixed(2)}</text>`;
  }

  let dots = '';
  hist.forEach(h => {
    if (h.val_loss != null){
      dots += `<circle cx="${x(h.step)}" cy="${y(h.val_loss)}" r="4" fill="#10b981"><title>Step ${h.step}: Val ${h.val_loss.toFixed(4)}</title></circle>`;
    }
    if (h.train_loss != null){
      dots += `<circle cx="${x(h.step)}" cy="${y(h.train_loss)}" r="3" fill="#3b82f6"><title>Step ${h.step}: Train ${h.train_loss.toFixed(4)}</title></circle>`;
    }
  });

  box.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="svg-chart">
      ${grid}
      <polyline fill="none" stroke="#3b82f6" stroke-width="2.5" points="${trainPts.join(' ')}" />
      <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="${valPts.join(' ')}" />
      ${dots}
      <text x="${padL}" y="${H - 10}" font-size="10" fill="var(--dim)">Step ${minStep}</text>
      <text x="${W - padR}" y="${H - 10}" font-size="10" fill="var(--dim)" text-anchor="end">Step ${maxStep}</text>
    </svg>
  `;
}

function renderExpertsChart(hist, liveStep, liveExp){
  const box = $('experts-chart-box');
  if (!hist.length && !liveExp){ box.innerHTML = ''; return; }

  let pts = hist.map(h => ({step: h.step, exp: h.experts}));
  if (liveStep != null && liveExp != null && (!pts.length || liveStep > pts[pts.length - 1].step)){
    pts.push({step: liveStep, exp: liveExp});
  }

  const W = 700, H = 160;
  const padL = 45, padR = 25, padT = 20, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const minStep = pts[0].step;
  const maxStep = Math.max(pts[pts.length - 1].step, minStep + 1);
  const minExp = Math.min(...pts.map(p => p.exp)) - 1;
  const maxExp = Math.max(...pts.map(p => p.exp)) + 2;

  const x = s => padL + ((s - minStep) / (maxStep - minStep)) * plotW;
  const y = e => padT + (1 - (e - minExp) / (maxExp - minExp)) * plotH;

  let pointsStr = pts.map(p => `${x(p.step).toFixed(1)},${y(p.exp).toFixed(1)}`).join(' ');
  let dots = pts.map(p => `<circle cx="${x(p.step)}" cy="${y(p.exp)}" r="4" fill="#8b5cf6"><title>Step ${p.step}: ${p.exp} experts</title></circle>`).join('');

  box.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" class="svg-chart">
      <polyline fill="none" stroke="#8b5cf6" stroke-width="2" points="${pointsStr}" />
      ${dots}
      <text x="${padL}" y="${H - 10}" font-size="10" fill="var(--dim)">Step ${minStep}</text>
      <text x="${W - padR}" y="${H - 10}" font-size="10" fill="var(--dim)" text-anchor="end">Step ${maxStep} (${pts[pts.length-1].exp} experts)</text>
      <text x="${padL - 8}" y="${y(pts[pts.length-1].exp) + 4}" font-size="10" fill="var(--dim)" text-anchor="end">${pts[pts.length-1].exp}</text>
    </svg>
  `;
}

function renderSamples(samples){
  const box = $('samples-container');
  if (!samples.length){ box.innerHTML = '<div style="color:var(--dim);font-size:12px;padding:8px">No checkpoint samples logged yet.</div>'; return; }
  box.innerHTML = '';
  samples.slice(-4).reverse().forEach(s => {
    const card = document.createElement('div');
    card.className = 'sample-card';
    card.innerHTML = `<div class="sample-step">Checkpoint at Step ${s.step.toLocaleString()}</div><div class="sample-text">${escapeHtml(s.sample)}</div>`;
    box.appendChild(card);
  });
}

$('btn-refresh').onclick = fetchMetrics;
setInterval(() => {
  if ($('auto-refresh').checked && $('metrics-view').style.display === 'block'){
    fetchMetrics();
  }
}, 5000);

// INITIALIZATION
empty();
loadConversationsList();
fetchMetrics();
requestAnimationFrame(grow);
</script>
</body></html>'''



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights",
                    help="the weights directory to talk to")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; default auto, falling back to CPU if "
                         "the card is full")
    ap.add_argument("--no-learn", dest="learn", action="store_false",
                    help="serve without learning. The weights are then opened "
                         "read-only and nothing is written back")
    ap.add_argument("--learn-lr", type=float, default=3e-4,
                    help="learning rate for the live stream")
    ap.add_argument("--prime-chars", type=int, default=1024,
                    help="characters of corpus to open the conversation "
                         "with, so the router has something to choose "
                         "experts from and the learning stream starts part "
                         "of the way to its first step. 0 disables it")
    ap.add_argument("--learn-chunk", type=int, default=None,
                    help="characters of conversation per optimiser step. "
                         "Defaults to training.chunk, which is tuned for "
                         "reading a corpus; try 256 for a short session, at "
                         "the cost of a denser diet than documents give")
    ap.add_argument("--save-every", type=int, default=8,
                    help="optimiser steps between writing the weights out")
    ap.add_argument("--precision", default=None,
                    choices=["bf16", "fp32"],
                    help="what the forward computes in; defaults to whatever "
                         "config.yaml trains with. No fp16: the recurrent "
                         "state reaches ~10,000 and RMSNorm squares it, past "
                         "fp16's 65,504")
    ap.add_argument("--resident", type=int, default=None,
                    help="number of experts to hold in VRAM slots (e.g. 146 to hold all)")
    ap.add_argument("--context", type=int, default=None,
                    help="override maximum context window in characters (e.g. 8192, 16384)")
    args = ap.parse_args()

    from minagi.config import get as _g, load as _lc
    from minagi.precision import set_compute_dtype
    set_compute_dtype(args.precision
                      or _g(_lc(), "training.precision", "bf16"))

    if not os.path.exists(os.path.join(args.weights, "core.npz")):
        raise SystemExit(
            f"no model in {args.weights}. Train one first:\n"
            f"    python3 train.py read data/train --save")

    global PRIME
    PRIME = load_prime(args.prime_chars)
    if args.prime_chars and not PRIME:
        print("  no corpus to prime from - the router will choose experts "
              "from the prompt alone", file=sys.stderr)

    if args.learn:
        print(f"\n  LEARNING IS ON. Talking to this model CHANGES "
              f"{args.weights}/ on disk.\n  Use --no-learn to serve it "
              f"read-only.\n")
    model = load(args.weights, args.device, learn=args.learn,
                 lr=args.learn_lr, save_every=args.save_every,
                 chunk=args.learn_chunk, resident=args.resident)
    if args.context and args.context != model.cfg.block:
        from minagi.recur import build_rope
        dev = next(model.parameters()).device
        cos, sin = build_rope(args.context, model.cfg.d_model // model.cfg.n_head,
                              model.cfg.rope_theta, dev)
        model.cfg.block = args.context
        model.register_buffer("rope_cos", cos, persistent=False)
        model.register_buffer("rope_sin", sin, persistent=False)
        print(f"  context set to {args.context:,} characters")
    if PRIME:
        print(f"  primed with {len(PRIME)} characters of self-knowledge")
    pool = getattr(model, "pool", None)
    if pool is not None:
        # Both numbers are the POOL. Comparing model.n_params(), which counts
        # the trunk plus whatever is resident, against the pool total reads as
        # the card holding more than the model has.
        print(f"  {pool.n_experts()} experts, "
              f"{pool.vram_params() / 1e6:.1f}M of {pool.n_params() / 1e6:.1f}M "
              f"on the card, context {model.cfg.block:,}")
    else:
        print(f"  {model.n_params() / 1e6:.1f}M parameters, "
              f"context {model.cfg.block:,}")
    print(f"\n  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
