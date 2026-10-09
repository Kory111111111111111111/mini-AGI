# mini-AGI Training & Commands Cheatsheet

This cheatsheet covers all the essential commands for training, serving, and monitoring **mini-AGI** on your machine.

---

## 1. Quick Overview: GUI vs. Terminal

- **Web GUI (`serve.py`)**: A chat window where you can talk to the model in real time and watch its experts route dynamically. It also learns from your conversation if live learning is enabled.
- **Terminal Training (`train.py`)**: Where you feed the model files, books, codebases, or corpora to do serious end-to-end training.

---

## 2. Training Commands (`train.py`)

Always make sure you use the virtual environment executable (`.\.venv\Scripts\python`).

### A. Train on Your Custom Files / Folders
Point the reader at any file or folder containing text, markdown, or code:

```powershell
# Read all files in data/, run 3 passes over them, and save the learned weights:
.\.venv\Scripts\python train.py read data --passes 3 --save

# Train on a specific subfolder (e.g., data/weird):
.\.venv\Scripts\python train.py read data/weird --passes 5 --save

# Dry run (test loss without modifying saved weights on disk):
.\.venv\Scripts\python train.py read data/weird --passes 1
```

#### Useful Flags for `train.py read`:
| Flag | Description | Default |
|---|---|---|
| `--passes N` | How many times to read through the files | `1` |
| `--save` | **Crucial**: writes the updated weights to disk. Without this, weights are not kept! | Off |
| `--weights-dir <path>` | Where weights live | `weights` |
| `--held-out <path>` | Validate against a held-out evaluation set | None |
| `--lr <float>` | Learning rate for the pool | `5e-5` |
| `--trunk-lr-mult <float>` | Trunk learning rate multiplier (slows trunk to prevent catastrophic forgetting) | `0.1` |
| `--sample-every N` | Generate sample generations to `runs/samples.txt` every N evals | `10` |

---

### B. Train on the Full Standard Corpus
If you want to train on the official 8-domain dataset (TinyStories, Code, Hermes Chat, Chess, Math, Wikipedia, Reasoning):

1. **Download and generate the datasets:**
   ```powershell
   # Build a small slice (few thousand examples) to test:
   .\.venv\Scripts\python -m corpora all --limit 5000

   # Or download and build the full dataset:
   .\.venv\Scripts\python -m corpora all
   ```

2. **Start the long training run:**
   ```powershell
   .\.venv\Scripts\python train.py read data/train --save --weights-dir weights --held-out data/val --sample-every 10
   ```

---

## 3. Starting the Web Chat GUI (`serve.py`)

Run the local web server:

```powershell
# Start with live conversational learning enabled (default chunk = 512 chars):
.\.venv\Scripts\python serve.py --port 8080 --learn-chunk 512

# Or start in read-only mode (talking will NOT modify saved weights):
.\.venv\Scripts\python serve.py --port 8080 --no-learn
```

Once running, open your browser to:
👉 **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

The header shows an expert roster strip: one segment per resident expert (shade = gate weight, ring = routed by the last character), with the memory countdown beneath it. While the GPU is busy training, do frontend work against the mock server instead — no model, no GPU, no disk writes:
```powershell
.\.venv\Scripts\python tools/serve_ui_only.py --port 8080
```

---

## 4. Monitoring Progress & Viewing Charts

mini-AGI generates live telemetry and sample logs during training:

### A. View Generated Samples
Open [`runs/samples.txt`](runs/samples.txt) in your editor to see what the model is generating at each evaluation checkpoint.

### B. Generate & Update Progress Graphs
```powershell
# Redraw the loss and perplexity curves:
.\.venv\Scripts\python tools/plot_progress.py

# Redraw the full visual dashboard (loss, experts, context):
.\.venv\Scripts\python tools/plot_dashboard.py

# Plot which experts are active/pruned across history:
.\.venv\Scripts\python tools/plot_experts.py
```
Generated chart images are saved directly to:
- `runs/training_progress.png`
- `runs/dashboard.png`

---

## 5. Inspecting Model Stats

To see how many experts the model currently has, how many parameters are allocated, and disk vs. VRAM usage:

```powershell
.\.venv\Scripts\python -m minagi.store weights
```

---

## 6. Hardware Verification

If you ever want to check CUDA performance, attention speeds, and VRAM memory:

```powershell
.\.venv\Scripts\python tools/device_check.py
```
