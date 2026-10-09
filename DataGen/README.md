# mini-AGI DataGen Studio

A dedicated local synthetic dataset generation studio built directly into `mini-AGI/DataGen`. It connects to multiple frontier AI providers using your local API keys in `.env.local` to generate, curate, inspect, and save high-signal training documents formatted specifically for mini-AGI.

---

## Features

- **Direct Integration with `.env.local`**: Connects immediately to:
  - **DeepSeek** (`deepseek-reasoner` / R1 reasoning traces, `deepseek-chat` / V3)
  - **Google Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-pro`, `gemini-1.5-flash`)
  - **OpenRouter** (Claude 3.5 Sonnet, DeepSeek R1, Llama 3.3 70B, Qwen 2.5 72B)
  - **Z.AI GLM Coding** (`glm-4.5`, `glm-4-plus`, `glm-4-flash`)
  - **Google Serper** (Real-time live web search augmentation)
- **Deep Thinking & Reasoning Extraction**: Streams and isolates `<think>` reasoning traces into collapsible inspection accordions alongside clean final dialogue turns.
- **Strict mini-AGI Format Enforcement**: Outputs raw `<user>...</user>` and `<bot>...</bot>` XML blocks with no markdown wrappers or random Python function dumps.
- **One-Click Multi-Destination Export**:
  - `data/train/Custom/` — Immediately available for local or cloud continual learning.
  - `data/train/conversation/` — Direct input to conversational training runs.
  - `DataGen/exports/` — Local archive with export viewer and download support.
- **Obsidian Glassmorphic UI**: Ultra-clean dark theme matching the IDE and mini-AGI design system.

---

## Quick Start

### 1. Launch via Batch File
Double-click `start.bat` in this folder, or run:
```cmd
start.bat
```

### 2. Manual Command Line
From the repository root or this directory:
```bash
# From mini-AGI root
.venv\Scripts\python DataGen\server.py

# Or from inside DataGen
..\.venv\Scripts\python server.py
```
Then navigate to: **`http://127.0.0.1:5050`** in your browser.

---

## Configuration (`.env.local`)

`DataGen/.env.local` is gitignored and read locally on each request. It supports:
```env
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-...
GLM_CODING_API_KEY=...
SERPER_API_KEY=...
```

---

## Recommended Prompting Presets

The UI comes equipped with built-in presets tuned for mini-AGI's byte-level architecture:
1. **Fluid Conversation & Persona**: Multi-turn dialogue focusing on friendliness, humor, and self-awareness without unprompted code syntax.
2. **Deep Reasoning (<think>)**: Problem-solving with step-by-step internal reasoning enclosed in `<think>...</think>`.
3. **Architecture & Self-Knowledge**: Questions probing expert pool capacity, resident VRAM paging, byte-level tokenization, and dynamic weights.
4. **Instruction Following**: Structured tasks, summaries, and roleplay.
