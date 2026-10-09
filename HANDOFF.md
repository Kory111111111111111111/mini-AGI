# mini-AGI Project Handoff & State Summary

> **Date:** October 7–8, 2026  
> **Repository:** `c:\Users\koryi\Desktop\APPS\mini-AGI`  
> **Active Environment:** Windows Local (RTX 4060 Ti 16GB) + RunPod Cloud (NVIDIA A40 46GB VRAM)  
> **Cloud Pod Status:** Pod `4umq0z7gnj506m` (`mini-agi-a40`) is **RUNNING** and actively training. Pod `4n030bulintcs3` remains **EXITED**.  
> **Local Server Status:** Local `serve.py` stopped (0 MB VRAM used, port 8080 free). VoiceChat server ready on port 7860.

---

## 1. Executive Summary & Current Operational State

1. **Active A40 GPU Cloud Training (`mini-agi-a40`):**
   * Currently executing a high-capacity joint conversational training run over `data/train/chat` (~2,197 multi-turn dialogs) + `data/train/Custom` on an NVIDIA A40 (46GB VRAM).
   * **Context Window Scaled:** Automatically expanded up to **9,216 tokens** (`block: 16384`, `context_now: 9216`).
   * **MoE Pool Dynamically Expanded:** Grew from 87 experts to **146 experts** (`pool_max: 146`, with 87 resident in GPU VRAM).
   * **Progress:** Over **116.1M characters read** in the current session (>201.3M lifetime characters in manifest). The run is ~26.6% through its 1-pass traversal of 435.88M characters.
   * **VRAM Utilization:** 45.4 GB / 46.0 GB (98.5% allocated, bf16 mixed precision).

2. **New Subsystem: Sesame Voice Arena (`VoiceChat/`):**
   * A full-duplex, real-time voice-to-voice conversational studio and multi-agent debate arena inspired by **Sesame AI**.
   * Completely self-contained and zero-dependency frontend (`index.html`, `app.js`, `audio.js`, `orb.js`, `debate.js`, `server.py`).
   * Features a fluid procedural audio-reactive Canvas orb, dual agent turn-taking with natural conversational gap delays, director microphone/keyboard barge-in interjections, live transcripts, and configurable endpoints for local `serve.py` or remote neural TTS / speech models (CSM-1B, Moshi, Kokoro).

3. **Local Machine State (Windows RTX 4060 Ti):**
   * Local weights directory (`weights/`) holds the pre-A40 87-expert checkpoint (Step 36,699, `read_chars: 44,641,280`, `context_now: 2252`).
   * Once cloud training finishes (or at any intermediate checkpoint), the 146-expert weights on the pod will be synced locally via `tools/sync_to_runpod.py` / `scp`.

---

## 2. Cloud Infrastructure & Pod Status

### Active Pod: `mini-agi-a40` (GPU Training Pod)
* **Pod ID:** `4umq0z7gnj506m`
* **Status:** `RUNNING`
* **Data Center:** `EU-SE-1` (Sweden)
* **Hardware:** 1x NVIDIA A40 (46,068 MiB VRAM), 9 vCPUs, 50 GB RAM, 75 GB NVMe disk.
* **Cost:** ~$0.49 / hr.
* **Direct SSH:**
  ```bash
  ssh -p 22028 root@194.68.245.26
  ```
  *(Configured with local public key `~/.ssh/id_ed25519.pub`)*
* **JupyterLab URL:** `https://4umq0z7gnj506m-8888.proxy.runpod.net/lab`
* **Volume Path:** `/workspace/mini-AGI` (overlay persistent disk).

### Paused Pod: `deliberate_blue_falcon` (RTX A5000)
* **Pod ID:** `4n030bulintcs3`
* **Status:** `EXITED` ($0.00/hr billing).
* **Policy Reminder:** **Do NOT start or spin up `4n030bulintcs3`** unless the user explicitly requests it (the host machine lacked available A5000 GPUs on last attempt).

---

## 3. Live Training Run Details

### Execution Pipeline
The training job runs detached inside tmux session `train` on pod `4umq0z7gnj506m`:
```bash
tmux new-session -d -s train 'python3 -u train.py read \
  data/train/chat \
  --held-out data/val \
  --chunk 4096 \
  --context 9216 \
  --resident 87 \
  --ram-capacity 96 \
  --passes 1 \
  --save \
  --no-pool-checkpoint \
  --no-plots \
  --precision bf16 2>&1 | tee -a train.log'
```

### Live Metrics & Checkpoint State (A40 Pod)
* **Active PID:** 9373
* **Current Step:** ~74,710+
* **Total Chars Read in Run:** ~116.1M / 435.88M (~26.6% through 1 pass)
* **Lifetime Characters in Manifest:** 201,333,760 (`read_chars`)
* **Total Parameter Bytes on Disk:** ~4.83 GB (`total_bytes: 4,835,468,052`)
* **MoE Pool Size:** **146 experts** (AutoGrow dynamically added +59 experts during chat dataset ingestion)
* **Resident Pool in VRAM:** 87 experts (~273.7M parameters active in VRAM)
* **Context Budget:** 9,216 tokens (`context_now: 9216`, max block 16,384)
* **Throughput:** ~12.5M – 13.0M characters/hour

### Inspecting Live Pod Status
To check live training from the local terminal:
```powershell
# View recent training logs & checkpoints
ssh -p 22028 root@194.68.245.26 "tail -n 25 /workspace/mini-AGI/train.log"

# View live GPU utilization
ssh -p 22028 root@194.68.245.26 "nvidia-smi"

# Attach directly to tmux session (press Ctrl+B then D to detach)
ssh -t -p 22028 root@194.68.245.26 "tmux attach -t train"
```

---

## 4. Sesame Voice Arena Subsystem (`VoiceChat/`)

Located at [VoiceChat/](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/VoiceChat).

### Core Components
* [index.html](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/VoiceChat/index.html): Responsive HUD containing dual agent cards (Agent A vs Agent B), reactive canvas stage, live transcript feed, director prompt input bar, push-to-talk microphone trigger, and settings modal.
* [orb.js](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/VoiceChat/orb.js): Procedural 3D-feel audio-reactive orb on HTML5 Canvas using simplex harmonic deformation, responding dynamically to real-time audio FFT frequencies. Dynamically morphs between speaker themes:
  * Cyan / Ocean (`#00f5d4`): Agent A (Maya)
  * Violet / Magenta (`#7928ca`): Agent B (Miles)
  * Rose / Amber (`#f72585`): Director / User speech
* [audio.js](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/VoiceChat/audio.js): Audio manager covering Web Audio API analysers, microphone stream capture, TTS audio streaming playback, and barge-in / speech interruption detection.
* [debate.js](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/VoiceChat/debate.js): Turn-taking orchestrator managing autonomous back-and-forth debates with configurable pause intervals, director topic steering, and transcript generation.
* [server.py](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/VoiceChat/server.py): Zero-dependency local server hosting `VoiceChat/` with proper CORS headers.

### How to Launch Voice Arena Locally
```powershell
cd VoiceChat
python server.py
```
Then navigate to `http://127.0.0.1:7860`.

---

## 5. Local Enhancements to `serve.py`

* **Attention Sink & Boundary Eviction:**
  * Preserves initial ~35% of token budget for priming instructions and character cards (attention sink).
  * Safely evicts older turns strictly along `</bot>` / `<user>` delimiters to prevent token clipping or broken partial sentences.
* **Context Override:** Supports `--context 8192` to initialize large rotary embedding tables on startup.
* **Persona Drawer Removed (Oct 2026):** The model ignored system instructions, so the persona button/drawer/`localStorage` plumbing was deleted from the `serve.py` web UI. Header icons are custom inline SVGs (node-dot motif), not emoji.
* **Expert Roster Strip (Oct 2026):** The `#mech` twin meters + UID chip boxes became one segment strip: one segment per resident expert shaded by gate weight (UID/gate in tooltip) over a 2px memory-countdown underline. `resident_experts()` also returns `used` — slots the last forward routed through, read from pool sites' `last_route` (per-character during generation, chunk-union during prefill) — highlighted with an accent ring. Gate shading is approximate (gates are anti-predictive of busyness per README); the ring is ground truth.
* **Hardware Footprint:** Consumes **< 2 GB VRAM** on local NVIDIA RTX 4060 Ti.

---

## 6. Helper Tools & Diagnostics (`tools/`)

* [tools/sample_now.py](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/tools/sample_now.py): Loads the current model weights on CPU to test generation without consuming GPU VRAM or interrupting active training runs.
* [tools/start_a40_train.sh](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/tools/start_a40_train.sh): Standard launch command for RunPod A40 instance.
* [tools/sync_to_runpod.py](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/tools/sync_to_runpod.py) & [tools/upload_chat_data.py](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/tools/upload_chat_data.py): Asset and dataset synchronization scripts.
* [tools/serve_ui_only.py](file:///c:/Users/koryi/Desktop/APPS/mini-AGI/tools/serve_ui_only.py): UI-only preview server — serves the real `serve.py` page template with mocked `/api/*` endpoints, no model load, no GPU, no disk writes. For frontend work while training holds the GPU.

---

## 7. Action Plan & Next Steps for the Next Agent

### When the Training Run Completes:
1. **Check Completion:** Confirm process `9373` in `tmux attach -t train` has completed and written final checkpoint to `/workspace/mini-AGI/weights/`.
2. **Download Weights to Local Machine:**
   ```powershell
   # Pull updated weights from A40 pod to local Windows repo
   scp -P 22028 -r root@194.68.245.26:/workspace/mini-AGI/weights/* .\weights\
   ```
3. **Stop Cloud Pod to Halt Billing:**
   Use the RunPod MCP tool `pod-action`:
   ```json
   {
     "ServerName": "runpod",
     "ToolName": "pod-action",
     "Arguments": {
       "id": "4umq0z7gnj506m",
       "body": { "action": "stop" }
     }
   }
   ```
4. **Test Model Locally:**
   Run inference with local GPU:
   ```powershell
   .\.venv\Scripts\python serve.py --device cuda --context 8192 --port 8080 --no-learn
   ```
   Open `http://127.0.0.1:8080` to verify conversational coherence after the 146-expert chat run.
5. **Connect with VoiceChat:**
   Start `VoiceChat/server.py` on port 7860, open Settings, and point the reasoning core to `http://localhost:8080`.
