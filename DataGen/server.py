#!/usr/bin/env python3
"""
mini-AGI DataGen Studio - Local Multi-Provider Synthetic Training Data Generator
Connects directly to DeepSeek, Gemini, OpenRouter, and GLM Coding using .env.local.
Formats and saves multi-turn conversational docs with <user>, <bot>, and <think> tags.
"""

import os
import sys
import json
import time
import re
import glob
import threading
import requests
from flask import Flask, request, jsonify, Response, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ROOT_DIR)
ENV_FILE = os.path.join(ROOT_DIR, ".env.local")
EXPORTS_DIR = os.path.join(ROOT_DIR, "exports")
CUSTOM_TRAIN_DIR = os.path.join(REPO_DIR, "data", "train", "Custom")
CONV_TRAIN_DIR = os.path.join(REPO_DIR, "data", "train", "conversation")

os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(CUSTOM_TRAIN_DIR, exist_ok=True)

def load_env():
    """Load keys from .env.local safely."""
    keys = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip()
    return keys

# Predefined models per provider with researched latest flagships
PROVIDER_CATALOG = {
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "models": [
            {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash (Flagship 552B MoE, 1M Context)", "supports_thinking": True},
            {"id": "deepseek-v4-pro", "name": "DeepSeek-V4-Pro (Flagship Deep Reasoning, 1M Context)", "supports_thinking": True},
            {"id": "deepseek-chat", "name": "DeepSeek-V3 / Chat (Legacy)", "supports_thinking": False},
            {"id": "deepseek-reasoner", "name": "DeepSeek-R1 / Reasoner (Legacy)", "supports_thinking": True}
        ]
    },
    "gemini": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "models": [
            {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Latest Flagship 2026)", "supports_thinking": True},
            {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash", "supports_thinking": True},
            {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite (Ultra Fast)", "supports_thinking": False},
            {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview (Deep Reasoning)", "supports_thinking": True},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "supports_thinking": True},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "supports_thinking": True},
            {"id": "gemini-flash-latest", "name": "Gemini Flash (Latest Stable Alias)", "supports_thinking": True},
            {"id": "gemini-pro-latest", "name": "Gemini Pro (Latest Stable Alias)", "supports_thinking": True}
        ]
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "type_in": True,
        "default_model": "anthropic/claude-3.7-sonnet",
        "popular_models": [
            {"id": "anthropic/claude-3.7-sonnet", "name": "Claude 3.7 Sonnet"},
            {"id": "anthropic/claude-3.7-sonnet:thinking", "name": "Claude 3.7 Sonnet (Thinking)"},
            {"id": "deepseek/deepseek-r1-0528", "name": "DeepSeek R1 (0528)"},
            {"id": "deepseek/deepseek-chat-v3.1", "name": "DeepSeek V3.1"},
            {"id": "openai/o3-mini", "name": "OpenAI o3-mini"},
            {"id": "openai/o3-mini-high", "name": "OpenAI o3-mini (High Effort)"},
            {"id": "openai/o1", "name": "OpenAI o1"},
            {"id": "openai/gpt-4o", "name": "GPT-4o"},
            {"id": "qwen/qwq-32b", "name": "Qwen QwQ 32B"},
            {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"}
        ],
        "models": [
            {"id": "anthropic/claude-3.7-sonnet", "name": "Claude 3.7 Sonnet", "supports_thinking": True},
            {"id": "anthropic/claude-3.7-sonnet:thinking", "name": "Claude 3.7 Sonnet (Thinking)", "supports_thinking": True},
            {"id": "deepseek/deepseek-r1-0528", "name": "DeepSeek R1 (0528)", "supports_thinking": True},
            {"id": "deepseek/deepseek-chat-v3.1", "name": "DeepSeek V3.1", "supports_thinking": False},
            {"id": "openai/o3-mini", "name": "OpenAI o3-mini", "supports_thinking": True},
            {"id": "openai/o1", "name": "OpenAI o1", "supports_thinking": True},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "supports_thinking": False},
            {"id": "qwen/qwq-32b", "name": "Qwen QwQ 32B", "supports_thinking": True},
            {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "supports_thinking": False}
        ]
    },
    "glm": {
        "name": "GLM Coding (Z.AI)",
        "env_key": "GLM_CODING_API_KEY",
        "models": [
            {"id": "glm-5.3", "name": "GLM-5.3 (Latest Flagship 2026 - Reasoning & Agentic)", "supports_thinking": True},
            {"id": "glm-5.3-flash", "name": "GLM-5.3 Flash (Ultra Fast)", "supports_thinking": False},
            {"id": "glm-5.3-flashx", "name": "GLM-5.3 FlashX", "supports_thinking": False},
            {"id": "glm-5.2", "name": "GLM-5.2 (1M Context)", "supports_thinking": False},
            {"id": "glm-5.1", "name": "GLM-5.1 (Long-Horizon Coding)", "supports_thinking": False},
            {"id": "glm-5", "name": "GLM-5 (744B MoE)", "supports_thinking": False},
            {"id": "glm-4.7", "name": "GLM-4.7", "supports_thinking": False},
            {"id": "glm-4.6", "name": "GLM-4.6", "supports_thinking": False},
            {"id": "glm-4.5", "name": "GLM-4.5 (Legacy)", "supports_thinking": False}
        ]
    }
}

DEFAULT_SYSTEM_PROMPT = """You are a synthetic dataset generator producing conversational training data for a byte-level language model called "mini-AGI".
Your goal is to generate natural, fluid, multi-turn conversations that teach the model how to be an articulate, friendly, and cohesive conversational partner.

### Reasoning & Formatting Rules:
1. Every conversation must use XML boundary tags on their own lines:
   <user>
   [User question, greeting, or comment]
   </user>
   <bot>
   [Assistant response. For every conversation topic, formulate internal thinking, tone calibration, and step-by-step reasoning inside <think> ... </think> before answering.]
   </bot>
2. Provide multi-turn conversations (2 to 4 back-and-forth turns per dialogue).
3. Do NOT include markdown code fences (no ```xml). Output raw text only.
4. Separate conversations with an empty line.
5. NO unprompted Python functions or code dumps unless the user explicitly requests code."""

_openrouter_cache = {"timestamp": 0, "models": []}

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/config", methods=["GET"])
def get_config():
    env = load_env()
    providers_status = {}
    for pid, pdata in PROVIDER_CATALOG.items():
        key_val = env.get(pdata["env_key"], "")
        providers_status[pid] = {
            "name": pdata["name"],
            "configured": bool(key_val and len(key_val) > 4),
            "models": pdata["models"],
            "type_in": pdata.get("type_in", False),
            "popular_models": pdata.get("popular_models", []),
            "default_model": pdata.get("default_model", pdata["models"][0]["id"] if pdata["models"] else "")
        }
    has_serper = bool(env.get("SERPER_API_KEY"))
    return jsonify({
        "providers": providers_status,
        "has_search": has_serper,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT
    })

@app.route("/api/openrouter/models", methods=["GET"])
def get_openrouter_models():
    now = time.time()
    if now - _openrouter_cache["timestamp"] < 3600 and _openrouter_cache["models"]:
        return jsonify({"models": _openrouter_cache["models"]})
    
    env = load_env()
    api_key = env.get("OPENROUTER_API_KEY", "")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            model_ids = [m["id"] for m in data if "id" in m]
            _openrouter_cache["models"] = model_ids
            _openrouter_cache["timestamp"] = now
            return jsonify({"models": model_ids})
    except Exception as e:
        print(f"[openrouter models error] {e}")
    
    return jsonify({"models": _openrouter_cache["models"]})

def call_serper(query, api_key):
    """Run real-time Google search via Serper."""
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 4},
            timeout=8
        )
        if r.status_code == 200:
            data = r.json()
            snippets = []
            for item in data.get("organic", [])[:4]:
                snippets.append(f"Title: {item.get('title')}\nSnippet: {item.get('snippet')}")
            return "\n\n".join(snippets)
    except Exception as e:
        print(f"[serper error] {e}")
    return ""

def stream_deepseek(api_key, model, messages, temperature=0.7, max_tokens=3000, thinking_level="high"):
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    if thinking_level != "off":
        # DeepSeek V4-Pro and Flash support effort levels (low, high, max)
        effort_map = {"low": "low", "medium": "high", "high": "high", "max": "max"}
        payload["effort"] = effort_map.get(thinking_level, "high")

    r = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
    if r.status_code != 200:
        yield f"data: {json.dumps({'error': f'DeepSeek HTTP {r.status_code}: {r.text}'})}\n\n"
        return

    thinking_opened = False
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]
            
            # DeepSeek Reasoner & V4 stream
            reasoning = delta.get("reasoning_content")
            if reasoning:
                if not thinking_opened:
                    thinking_opened = True
                    yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                yield f"data: {json.dumps({'type': 'thinking', 'delta': reasoning})}\n\n"

            content = delta.get("content")
            if content:
                if thinking_opened:
                    thinking_opened = False
                    yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
                yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
        except Exception:
            continue
    if thinking_opened:
        yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

def stream_openrouter(api_key, model, messages, temperature=0.7, max_tokens=3000, thinking_level="high"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "mini-AGI DataGen",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    if thinking_level != "off":
        payload["include_reasoning"] = True
        
        # OpenAI models (o1, o3-mini) reasoning effort
        if any(k in model.lower() for k in ["o1", "o3"]):
            effort_val = "low" if thinking_level == "low" else ("high" if thinking_level in ["high", "max"] else "medium")
            payload["reasoning"] = {"effort": effort_val}
        
        # Claude 3.7 Sonnet extended thinking budget
        elif "claude-3.7" in model.lower() or "claude-3-7" in model.lower():
            budget = 1024 if thinking_level == "low" else (4096 if thinking_level == "medium" else 12000)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            if max_tokens < budget + 2000:
                payload["max_tokens"] = budget + 4000
        else:
            payload["reasoning"] = {"effort": thinking_level}

    r = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
    if r.status_code != 200:
        yield f"data: {json.dumps({'error': f'OpenRouter HTTP {r.status_code}: {r.text}'})}\n\n"
        return

    thinking_opened = False
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]
            
            reasoning = delta.get("reasoning")
            if reasoning:
                if not thinking_opened:
                    thinking_opened = True
                    yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                yield f"data: {json.dumps({'type': 'thinking', 'delta': reasoning})}\n\n"

            content = delta.get("content")
            if content:
                if thinking_opened:
                    thinking_opened = False
                    yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
                yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
        except Exception:
            continue
    if thinking_opened:
        yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

def stream_glm(api_key, model, messages, temperature=0.7, max_tokens=3000, thinking_level="high"):
    url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    if thinking_level != "off":
        payload["thinking"] = {"type": "enabled"}

    r = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
    if r.status_code != 200:
        yield f"data: {json.dumps({'error': f'GLM HTTP {r.status_code}: {r.text}'})}\n\n"
        return

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content")
            if content:
                yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
        except Exception:
            continue
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

def stream_gemini(api_key, model, messages, temperature=0.7, max_tokens=3000, thinking_level="high"):
    url = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    if thinking_level != "off":
        budget = 1024 if thinking_level == "low" else (4096 if thinking_level == "medium" else 8192)
        payload["thinking_config"] = {"thinking_budget": budget}

    r = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
    if r.status_code != 200:
        yield f"data: {json.dumps({'error': f'Gemini HTTP {r.status_code}: {r.text}'})}\n\n"
        return

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content")
            if content:
                yield f"data: {json.dumps({'type': 'content', 'delta': content})}\n\n"
        except Exception:
            continue
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.route("/api/generate", methods=["POST"])
def generate():
    env = load_env()
    body = request.get_json(force=True)
    provider = body.get("provider", "deepseek")
    model = body.get("model", "deepseek-flash")
    system_prompt = body.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    user_prompt = body.get("prompt", "")
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 3500))
    use_search = bool(body.get("use_search", False))
    thinking_level = body.get("thinking_level", "high")  # "off", "low", "medium", "high", "max"

    # Automatically expand max_tokens headroom when reasoning is active
    # This prevents the reasoning trace from consuming the entire token budget and starving the actual response
    effective_max_tokens = max_tokens
    if thinking_level != "off":
        effective_max_tokens = max(max_tokens, 6500)

    # Optional Serper web search augmentation
    if use_search and env.get("SERPER_API_KEY"):
        search_res = call_serper(user_prompt, env["SERPER_API_KEY"])
        if search_res:
            user_prompt = f"Background research info:\n{search_res}\n\nTask:\n{user_prompt}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return Response(
        dispatch_stream(provider, model, messages, temperature, effective_max_tokens, thinking_level, env),
        mimetype="text/event-stream"
    )

def dispatch_stream(provider, model, messages, temperature, max_tokens, thinking_level, env):
    """Unified dispatcher for streaming across DeepSeek, Gemini, OpenRouter, and GLM."""
    if provider == "deepseek":
        key = env.get("DEEPSEEK_API_KEY")
        if not key:
            yield f"data: {json.dumps({'error': 'DEEPSEEK_API_KEY missing in .env.local'})}\n\n"
            return
        yield from stream_deepseek(key, model, messages, temperature, max_tokens, thinking_level)
    elif provider == "gemini":
        key = env.get("GEMINI_API_KEY")
        if not key:
            yield f"data: {json.dumps({'error': 'GEMINI_API_KEY missing in .env.local'})}\n\n"
            return
        yield from stream_gemini(key, model, messages, temperature, max_tokens, thinking_level)
    elif provider == "openrouter":
        key = env.get("OPENROUTER_API_KEY")
        if not key:
            yield f"data: {json.dumps({'error': 'OPENROUTER_API_KEY missing in .env.local'})}\n\n"
            return
        yield from stream_openrouter(key, model, messages, temperature, max_tokens, thinking_level)
    elif provider == "glm":
        key = env.get("GLM_CODING_API_KEY")
        if not key:
            yield f"data: {json.dumps({'error': 'GLM_CODING_API_KEY missing in .env.local'})}\n\n"
            return
        yield from stream_glm(key, model, messages, temperature, max_tokens, thinking_level)
    else:
        yield f"data: {json.dumps({'error': f'Unknown provider {provider}'})}\n\n"

# =========================================================================
# MINI-AGI STUDENT & ARENA SPARRING ENDPOINTS
# =========================================================================
MINIAGI_URL = "http://127.0.0.1:8080"

@app.route("/api/student/state", methods=["GET"])
def student_state():
    """Query local mini-AGI live serving/learning telemetry."""
    try:
        r = requests.get(f"{MINIAGI_URL}/api/state", timeout=3)
        if r.status_code == 200:
            data = r.json()
            return jsonify({
                "online": True,
                "url": MINIAGI_URL,
                "learn": data.get("learn"),
                "pool": data.get("pool")
            })
        return jsonify({"online": False, "url": MINIAGI_URL, "status_code": r.status_code})
    except Exception as e:
        return jsonify({"online": False, "url": MINIAGI_URL, "error": str(e)})

@app.route("/api/student/chat", methods=["POST"])
def student_chat():
    """Proxy chat generation to local mini-AGI serve.py, streaming back SSE tokens & learning status."""
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    max_new = int(body.get("max_new", 300))
    system_prompt = body.get("system_prompt", "")

    def event_stream():
        try:
            r = requests.post(
                f"{MINIAGI_URL}/api/chat",
                json={"messages": messages, "max_new": max_new, "system_prompt": system_prompt},
                stream=True,
                timeout=120
            )
            if r.status_code != 200:
                yield f"data: {json.dumps({'error': f'mini-AGI returned HTTP {r.status_code}: {r.text}'})}\n\n"
                return
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    yield f"{line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Failed to communicate with mini-AGI: {str(e)}'})}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/api/arena/teacher_turn", methods=["POST"])
def arena_teacher_turn():
    """Generate the next Teacher pedagogical challenge or critique for mini-AGI."""
    env = load_env()
    body = request.get_json(force=True)
    provider = body.get("provider", "deepseek")
    model = body.get("model", "deepseek-flash")
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 800))
    thinking_level = body.get("thinking_level", "high")
    curriculum = body.get("curriculum", "architecture")
    custom_goal = body.get("custom_goal", "")
    history = body.get("history", [])

    effective_max_tokens = max(max_tokens, 4500) if thinking_level != "off" else max_tokens

    if curriculum == "architecture":
        role_desc = (
            "You are an expert AI professor tutoring mini-AGI on its internal architecture: "
            "(1) Continual online backprop loop without separate training/inference stages. "
            "(2) 150 total expert pool with 32-87 resident VRAM slots dynamically paged by a gating router. "
            "(3) Byte-level modeling (vocab 265 bytes: 256 raw bytes + special marker tokens) without any BPE tokenizer. "
            "(4) New experts spawned by cloning and inheriting parent weights with small perturbation, not random noise.\n"
            "Your objective: Pose rigorous questions, evaluate mini-AGI's replies, praise accurate insights, gently correct hallucinations, and ask the next progressive question."
        )
    elif curriculum == "logic":
        role_desc = (
            "You are a rigorous Socratic logic and deduction coach training mini-AGI. "
            "Pose multi-step logic puzzles, syllogisms, or algorithmic deduction challenges. "
            "Evaluate its reasoning steps, expose logical flaws, and guide it to structured analytical thinking."
        )
    elif curriculum == "conversation":
        role_desc = (
            "You are a conversational mentor helping mini-AGI develop natural, coherent, high-register conversational fluency. "
            "Engage in intelligent discussion, critique repetitive loops or stiffness, and model crisp, engaging dialogue."
        )
    elif curriculum == "adversarial":
        role_desc = (
            "You are an AI robustness evaluator testing mini-AGI for edge cases, prompt injection resistance, "
            "and factual stability. Push its boundaries while diagnosing any failure modes."
        )
    else:
        role_desc = f"You are a specialized AI Teacher guiding mini-AGI with the following objective:\n{custom_goal or 'Conduct an educational multi-turn dialogue with mini-AGI.'}"

    system_prompt = (
        f"{role_desc}\n\n"
        "GUIDELINES FOR THE TEACHER:\n"
        "- When starting, pose a clear, engaging initial challenge or question.\n"
        "- When following up on mini-AGI's response, explicitly critique what mini-AGI got right or wrong, provide concise corrective explanation, and ask the next challenge question.\n"
        "- Keep your spoken output under 150 words per turn so the student stays focused.\n"
        "- Do NOT output <user> or <bot> tags in your response; speak directly as the Teacher."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "assistant", "content": turn.get("teacher", "")})
        messages.append({"role": "user", "content": f"[mini-AGI Student Response]:\n{turn.get('student', '')}"})

    if not history:
        messages.append({"role": "user", "content": "Begin the training session. Introduce the topic and pose your first question or challenge to mini-AGI."})
    else:
        messages.append({"role": "user", "content": "Evaluate mini-AGI's latest response above, give brief constructive feedback, and pose the next challenge or follow-up question."})

    return Response(
        dispatch_stream(provider, model, messages, temperature, effective_max_tokens, thinking_level, env),
        mimetype="text/event-stream"
    )

@app.route("/api/save", methods=["POST"])
def save_export():
    body = request.get_json(force=True)
    content = body.get("content", "").strip()
    title = body.get("title", "").strip() or f"synth_{int(time.time())}"
    destinations = body.get("destinations", ["exports"])

    # Sanitize title
    safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)
    filename = f"{safe_title}.txt"

    saved_paths = []
    if "exports" in destinations:
        p = os.path.join(EXPORTS_DIR, filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        saved_paths.append(p)

    if "custom" in destinations:
        p = os.path.join(CUSTOM_TRAIN_DIR, filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        saved_paths.append(p)

    if "conversation" in destinations:
        p = os.path.join(CONV_TRAIN_DIR, filename)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        saved_paths.append(p)

    return jsonify({
        "success": True,
        "filename": filename,
        "chars": len(content),
        "paths": saved_paths
    })

@app.route("/api/exports", methods=["GET"])
def list_exports():
    files = []
    for p in glob.glob(os.path.join(EXPORTS_DIR, "*.txt")):
        stat = os.stat(p)
        files.append({
            "name": os.path.basename(p),
            "size": stat.st_size,
            "modified": stat.st_mtime
        })
    files.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify({"files": files})

@app.route("/api/exports/<name>", methods=["GET"])
def get_export(name):
    return send_from_directory(EXPORTS_DIR, name)

@app.route("/api/exports/<name>", methods=["DELETE"])
def delete_export(name):
    p = os.path.join(EXPORTS_DIR, name)
    if os.path.exists(p):
        os.remove(p)
        return jsonify({"success": True})
    return jsonify({"error": "File not found"}), 404

def run():
    port = 5050
    print(f"\n=======================================================")
    print(f"  mini-AGI DataGen Studio live at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

if __name__ == "__main__":
    run()
