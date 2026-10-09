# Sesame Voice Arena 🎙️

A real-time voice-to-voice conversational studio and multi-agent debate arena inspired by **Sesame AI**.

Features a fluid, organic audio-reactive visualizer orb, native turn-taking, barge-in / interruption controls, and real-time audio playback.

---

## Quick Start (Zero Dependencies)

Run the server script from this directory:

```powershell
python server.py
```

Or open `index.html` directly in any modern web browser (Chrome, Edge, Firefox).

---

## Core Features

### 1. ⚔️ Multi-Agent Debate Arena
* **Agent A vs. Agent B:** Configure distinct personas, debate stances, reasoning cores, and voice engines.
* **Autonomous Turn-Taking:** Agents debate back-and-forth in real-time with natural conversational gap delays.
* **Director Interjections:** Interject via keyboard prompt or Hold-to-Talk microphone to moderate and steer the debate without breaking the loop.

### 2. 🎙️ Direct Voice Chat Mode
* Switch to 1-on-1 voice conversation with continuous microphone streaming and immediate speech interruptions.

### 3. 🌊 Sesame Fluid Audio-Reactive Orb
* Procedurally generated HTML5 Canvas mesh with multi-frequency harmonic deformations.
* Dances dynamically to audio energy (reactive to both incoming AI speech synthesis and your microphone input).
* Dynamic speaker colorways:
  * **Cyan / Ocean Blue:** Agent A (Maya)
  * **Violet / Magenta:** Agent B (Miles)
  * **Rose / Amber:** Director / Your Voice

---

## Supported Voice & Brain Models

### Voice-to-Voice Engines
* **Sesame CSM-1B:** Real-time conversational speech model (1B params, Llama 3.2 backbone + Mimi audio codec).
* **Kyutai Moshi (7B):** Full-duplex conversational model.
* **Kokoro-82M:** Ultra-fast sub-30ms neural TTS.
* **Mini-Omni 2 / Fish Speech:** End-to-end audio streaming.
* **Browser Native:** Low-latency Web Speech API fallback.

### Reasoning Cores
* **mini-AGI:** Continual learning byte-level MoE model.
* **DeepSeek-R1 / V3**
* **Gemini 2.0 Flash**
* **Llama 3.2**

---

## Connecting Live Remote Voice Servers (RunPod / GPU)

Click the **Gear icon (Settings)** in the top right:
1. **Remote CSM Endpoint:** Point to your hosted FastAPI or `csm.rs` server (e.g. `https://<pod-id>-8765.proxy.runpod.net` or `http://localhost:8765`).
2. **mini-AGI Endpoint:** Point to `http://localhost:8080` (or your RunPod pod IP running `serve.py`).
