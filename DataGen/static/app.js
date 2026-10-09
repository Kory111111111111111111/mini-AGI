/**
 * mini-AGI DataGen Studio · Frontend Engine
 * Multi-Provider Synthetic Dataset Generator with Per-Model Reasoning & Thinking Effort
 */

const state = {
  providers: {},
  activeProvider: 'deepseek',
  activeModel: 'deepseek-flash',
  thinkingLevel: 'high',
  customModelOverride: false,
  isGenerating: false,
  rawContent: '',
  thinkingContent: '',
  conversations: [],
  activeTab: 'formatted',
  activePreset: 'chat',
  openrouterModelsLoaded: false,
  currentMode: 'studio'
};

const arenaState = {
  active: false,
  paused: false,
  currentRound: 0,
  maxRounds: 3,
  studentMaxChars: 250,
  autoPlay: true,
  history: [],
  teacherProvider: 'deepseek',
  teacherModel: 'deepseek-flash',
  teacherThinking: 'high',
  curriculum: 'architecture',
  customCurriculum: '',
  studentOnline: false,
  activeAbortController: null
};

let arenaPollInterval = null;

const PRESET_PROMPTS = {
  chat: `Generate 3 distinct, high-signal multi-turn conversations between a human and mini-AGI.
Tone: Warm, thoughtful, witty, articulate, and deeply cohesive.
Guidelines: Focus on natural human dialogue, clear explanations, and friendly conversation. Avoid unprompted Python code snippets or synthetic function declarations.`,

  audio: `Generate 2 deep technical conversations exploring audio DSP, C++ JUCE plugin development, comb filtering algorithms, and synthetic reverb topologies.
Format with natural curiosity from the user and precise mathematical and architectural explanations from the bot.`,

  creative: `Generate 2 immersive, atmospheric conversations discussing deep philosophy, artificial consciousness, and memory preservation.
The dialogue should feel articulate, reflective, and poetic.`,

  self: `Generate 2 deep conversations where the user asks mini-AGI about its inner architecture.
Topics: The continual learning loop, expert pool expansion (87 resident VRAM slots paged from 150 pool capacity), 265-byte vocabulary with no BPE tokenizer, and why new experts inherit parent weights instead of random noise.`,

  freeform: ``
};

const THINKING_DESCRIPTIONS = {
  off: { label: 'Off', hint: 'Direct answers with no internal reasoning trace' },
  low: { label: 'Low Effort', hint: 'Light reasoning (~1k tokens) for quick validation' },
  medium: { label: 'Medium Effort', hint: 'Balanced thinking (~4k tokens) across all turns' },
  high: { label: 'High Effort', hint: 'Deep step-by-step reasoning (~8k tokens) before answering' },
  max: { label: 'Max Effort', hint: 'Exhaustive thinking trace allowed by provider' }
};

const $ = (id) => document.getElementById(id);

// 1. INITIALIZATION
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadConfig();
  selectPreset('chat');
  setThinkingLevel('high');
});

async function loadConfig() {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    state.providers = data.providers;

    if (data.default_system_prompt) {
      $('system-prompt-text').value = data.default_system_prompt;
    }

    if (!data.has_search) {
      const serperRow = $('row-serper');
      if (serperRow) serperRow.style.opacity = '0.5';
    }

    updateProviderHeader();
    selectProvider('deepseek');
  } catch (err) {
    showToast('Failed to load API configuration', 'error');
  }
}

function updateProviderHeader() {
  let count = 0;
  for (const pid in state.providers) {
    if (state.providers[pid].configured) count++;
  }
  const text = $('providers-status-text');
  if (text) {
    text.textContent = `${count} / 4 Providers Active (.env.local)`;
  }
}

// 2. PROVIDER & MODEL SELECTION
async function selectProvider(providerId) {
  state.activeProvider = providerId;
  state.customModelOverride = false;
  $('box-custom-override').style.display = 'none';

  // Update Segmented Tabs
  document.querySelectorAll('.prov-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.provider === providerId);
  });

  const pdata = state.providers[providerId];
  if (!pdata) return;

  const boxOpenRouter = $('box-openrouter-input');
  const boxDropdown = $('box-standard-dropdown');

  if (providerId === 'openrouter') {
    // OpenRouter: Show Direct Type-in Input
    boxDropdown.style.display = 'none';
    boxOpenRouter.style.display = 'block';

    const input = $('input-openrouter-model');
    if (!input.value) {
      input.value = pdata.default_model || 'anthropic/claude-3.7-sonnet';
    }
    state.activeModel = input.value.trim();
    adjustThinkingForModel(state.activeModel);

    // Lazy load OpenRouter model catalog for autocomplete
    loadOpenRouterModels();
  } else {
    // DeepSeek, Gemini, GLM-5: Show Dropdown
    boxOpenRouter.style.display = 'none';
    boxDropdown.style.display = 'block';

    const select = $('select-model');
    select.innerHTML = '';

    if (pdata.models && pdata.models.length > 0) {
      pdata.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name + (m.supports_thinking ? ' (⚡ Thinking)' : '');
        opt.dataset.thinking = m.supports_thinking ? 'true' : 'false';
        select.appendChild(opt);
      });
      state.activeModel = pdata.models[0].id;
      adjustThinkingForModel(state.activeModel, pdata.models[0].supports_thinking);
    }
  }
}

async function loadOpenRouterModels() {
  if (state.openrouterModelsLoaded) return;
  try {
    const res = await fetch('/api/openrouter/models');
    const data = await res.json();
    if (data.models && data.models.length > 0) {
      const datalist = $('datalist-openrouter-models');
      datalist.innerHTML = '';
      data.models.forEach(mid => {
        const opt = document.createElement('option');
        opt.value = mid;
        datalist.appendChild(opt);
      });
      state.openrouterModelsLoaded = true;
    }
  } catch (e) {
    console.error('Failed to load OpenRouter models list', e);
  }
}

function adjustThinkingForModel(modelName, supportsThinking = null) {
  const m = modelName.toLowerCase();
  const capTag = $('model-cap-tag');

  const isThinkingModel = supportsThinking ?? (
    m.includes('reason') || m.includes('think') || m.includes('r1') ||
    m.includes('o1') || m.includes('o3') || m.includes('flash') || m.includes('pro')
  );

  if (capTag) {
    capTag.textContent = isThinkingModel ? 'Supports Reasoning' : 'Standard Model';
    capTag.style.color = isThinkingModel ? 'var(--accent-amber)' : 'var(--accent-cyan)';
  }

  // Adjust default thinking level if transitioning
  if (isThinkingModel && state.thinkingLevel === 'off') {
    setThinkingLevel('high');
  } else if (!isThinkingModel && state.thinkingLevel === 'max') {
    setThinkingLevel('medium');
  }
}

function setThinkingLevel(level) {
  state.thinkingLevel = level;
  document.querySelectorAll('.level-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.level === level);
  });

  const info = THINKING_DESCRIPTIONS[level] || THINKING_DESCRIPTIONS.high;
  const labelEl = $('thinking-mode-label');
  const hintEl = $('thinking-hint-text');

  if (labelEl) {
    labelEl.textContent = info.label;
    labelEl.style.color = level === 'off' ? 'var(--text-dim)' : 'var(--accent-amber)';
  }
  if (hintEl) {
    hintEl.textContent = info.hint;
  }
}

// 3. EVENT LISTENERS
function setupEventListeners() {
  // Provider Tabs
  document.querySelectorAll('.prov-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      selectProvider(tab.dataset.provider);
    });
  });

  // Standard Dropdown Change
  $('select-model').addEventListener('change', (e) => {
    state.activeModel = e.target.value;
    const selectedOpt = e.target.selectedOptions[0];
    adjustThinkingForModel(state.activeModel, selectedOpt?.dataset.thinking === 'true');
  });

  // OpenRouter Type-in Input
  $('input-openrouter-model').addEventListener('input', (e) => {
    state.activeModel = e.target.value.trim();
    adjustThinkingForModel(state.activeModel);
    updateChipActiveState(state.activeModel);
  });

  // OpenRouter Quick Chips
  document.querySelectorAll('#openrouter-quick-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const m = chip.dataset.model;
      $('input-openrouter-model').value = m;
      state.activeModel = m;
      adjustThinkingForModel(m);
      updateChipActiveState(m);
    });
  });

  // Thinking Level Tabs
  document.querySelectorAll('.level-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setThinkingLevel(btn.dataset.level);
    });
  });

  // Custom Model Override Toggle
  $('btn-toggle-custom-model').addEventListener('click', () => {
    state.customModelOverride = !state.customModelOverride;
    const box = $('box-custom-override');
    box.style.display = state.customModelOverride ? 'block' : 'none';
    if (state.customModelOverride) {
      $('input-custom-model-id').focus();
    } else {
      if (state.activeProvider !== 'openrouter') {
        state.activeModel = $('select-model').value;
      }
    }
  });

  $('input-custom-model-id').addEventListener('input', (e) => {
    if (state.customModelOverride && e.target.value.trim()) {
      state.activeModel = e.target.value.trim();
      adjustThinkingForModel(state.activeModel);
    }
  });

  // Presets
  document.querySelectorAll('.preset-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      selectPreset(pill.dataset.preset);
    });
  });

  // View Tabs
  $('tab-formatted').addEventListener('click', () => switchTab('formatted'));
  $('tab-raw').addEventListener('click', () => switchTab('raw'));
  $('tab-thinking').addEventListener('click', () => switchTab('thinking'));

  // Sliders
  $('range-temp').addEventListener('input', (e) => {
    $('val-temp').textContent = e.target.value;
  });
  $('range-tokens').addEventListener('input', (e) => {
    $('val-tokens').textContent = parseInt(e.target.value).toLocaleString();
  });
  $('range-batch').addEventListener('input', (e) => {
    $('val-batch').textContent = e.target.value;
  });

  // Generation Controls
  $('btn-generate').addEventListener('click', startGeneration);
  $('btn-stop').addEventListener('click', stopGeneration);

  // Keyboard Shortcut: Ctrl+Enter to Generate
  $('prompt-input').addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      startGeneration();
    }
  });

  // Raw XML Toolbar & Live Sync
  $('raw-output-text').addEventListener('input', (e) => {
    state.rawContent = e.target.value;
    $('chip-chars').textContent = `${state.rawContent.length.toLocaleString()} chars`;
    parseAndRenderConversations();
  });

  $('btn-format-xml').addEventListener('click', formatXmlContent);
  $('btn-validate-xml').addEventListener('click', validateXmlTags);
  $('btn-insert-user').addEventListener('click', () => insertTagPair('user'));
  $('btn-insert-bot').addEventListener('click', () => insertTagPair('bot'));
  $('btn-insert-think').addEventListener('click', () => insertTagPair('think'));

  // Copy & Clear
  $('btn-copy').addEventListener('click', copyContent);
  $('btn-clear').addEventListener('click', clearContent);

  // Sync / Save Buttons
  $('btn-save-doc').addEventListener('click', saveDocument);
  $('btn-quick-export').addEventListener('click', saveDocument);

  // Mode Switcher (Studio vs Arena)
  $('btn-mode-studio').addEventListener('click', () => switchMode('studio'));
  $('btn-mode-arena').addEventListener('click', () => switchMode('arena'));

  // Arena Provider Tabs
  document.querySelectorAll('#sidebar-arena-view .prov-tab').forEach(tab => {
    tab.addEventListener('click', () => selectArenaProvider(tab.dataset.provider));
  });

  // Arena Model Select
  $('select-arena-model').addEventListener('change', (e) => {
    arenaState.teacherModel = e.target.value;
    updateArenaTeacherBadge();
  });

  $('input-arena-openrouter-model').addEventListener('input', (e) => {
    if (e.target.value.trim()) {
      arenaState.teacherModel = e.target.value.trim();
      updateArenaTeacherBadge();
    }
  });

  // Arena Thinking Levels
  document.querySelectorAll('#arena-thinking-levels .t-level-btn').forEach(btn => {
    btn.addEventListener('click', () => setArenaThinkingLevel(btn.dataset.level));
  });

  // Arena Curriculum
  $('select-arena-curriculum').addEventListener('change', (e) => {
    arenaState.curriculum = e.target.value;
    $('box-custom-curriculum').style.display = e.target.value === 'custom' ? 'block' : 'none';
  });

  // Arena Sliders
  $('range-arena-rounds').addEventListener('input', (e) => {
    $('val-arena-rounds').textContent = e.target.value;
    arenaState.maxRounds = parseInt(e.target.value);
  });
  $('range-student-tokens').addEventListener('input', (e) => {
    $('val-student-tokens').textContent = e.target.value;
    arenaState.studentMaxChars = parseInt(e.target.value);
  });
  $('check-arena-autoplay').addEventListener('change', (e) => {
    arenaState.autoPlay = e.target.checked;
  });

  // Arena Action Buttons
  $('btn-arena-start').addEventListener('click', toggleArenaSession);
  $('btn-arena-step').addEventListener('click', stepArenaSession);
  $('btn-arena-reset').addEventListener('click', resetArenaSession);
  $('btn-arena-save').addEventListener('click', saveArenaToDatasets);
  $('btn-arena-clear').addEventListener('click', resetArenaSession);
  $('btn-arena-intervene').addEventListener('click', injectArenaTurn);
  $('arena-intervention-input').addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      injectArenaTurn();
    }
  });
}

function formatXmlContent() {
  let text = state.rawContent || $('raw-output-text').value;
  if (!text || !text.trim()) {
    showToast('Nothing to format', 'info');
    return;
  }

  // Ensure tags sit cleanly on their own lines
  text = text.replace(/<(user|bot|think)>/gi, '\n<$1>\n');
  text = text.replace(/<\/(user|bot|think)>/gi, '\n</$1>\n');
  text = text.split('\n').map(l => l.trim()).filter(l => l.length > 0 || true).join('\n');
  text = text.replace(/\n{3,}/g, '\n\n').trim() + '\n';

  state.rawContent = text;
  $('raw-output-text').value = text;
  $('chip-chars').textContent = `${text.length.toLocaleString()} chars`;
  parseAndRenderConversations();
  validateXmlTags();
  showToast('Formatted XML and tags', 'success');
}

function validateXmlTags() {
  const text = state.rawContent || $('raw-output-text').value;
  const statusEl = $('validate-tag-status');
  if (!statusEl) return;

  const uOpen = (text.match(/<user>/gi) || []).length;
  const uClose = (text.match(/<\/user>/gi) || []).length;
  const bOpen = (text.match(/<bot>/gi) || []).length;
  const bClose = (text.match(/<\/bot>/gi) || []).length;
  const tOpen = (text.match(/<think>/gi) || []).length;
  const tClose = (text.match(/<\/think>/gi) || []).length;

  const isBalanced = (uOpen === uClose) && (bOpen === bClose) && (tOpen === tClose);

  if (isBalanced && uOpen > 0) {
    statusEl.textContent = `✓ ${uOpen} Turns Balanced`;
    statusEl.style.color = 'var(--accent-emerald)';
  } else if (uOpen === 0 && bOpen === 0) {
    statusEl.textContent = `Validate Tags`;
    statusEl.style.color = '';
  } else {
    statusEl.textContent = `⚠ Tag Mismatch (${uOpen}/${uClose})`;
    statusEl.style.color = 'var(--accent-rose)';
  }
}

function insertTagPair(tagName) {
  const textarea = $('raw-output-text');
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const sel = textarea.value.substring(start, end);
  const replacement = `\n<${tagName}>\n${sel || '...'}\n</${tagName}>\n`;

  textarea.value = textarea.value.substring(0, start) + replacement + textarea.value.substring(end);
  state.rawContent = textarea.value;
  textarea.focus();
  parseAndRenderConversations();
  validateXmlTags();
}

function updateChipActiveState(modelId) {
  document.querySelectorAll('#openrouter-quick-chips .chip').forEach(chip => {
    chip.classList.toggle('active', chip.dataset.model === modelId);
  });
}

function selectPreset(presetKey) {
  state.activePreset = presetKey;
  document.querySelectorAll('.preset-pill').forEach(p => {
    p.classList.toggle('active', p.dataset.preset === presetKey);
  });

  const textarea = $('prompt-input');
  if (PRESET_PROMPTS[presetKey]) {
    textarea.value = PRESET_PROMPTS[presetKey];
  } else {
    textarea.value = '';
    textarea.focus();
  }
}

function switchTab(tabId) {
  state.activeTab = tabId;
  $('tab-formatted').classList.toggle('active', tabId === 'formatted');
  $('tab-raw').classList.toggle('active', tabId === 'raw');
  $('tab-thinking').classList.toggle('active', tabId === 'thinking');

  $('view-formatted').style.display = tabId === 'formatted' ? 'flex' : 'none';
  $('view-raw').style.display = tabId === 'raw' ? 'flex' : 'none';
  $('view-thinking').style.display = tabId === 'thinking' ? 'flex' : 'none';
}

// 4. STREAMING ENGINE
let activeAbortController = null;

async function startGeneration() {
  if (state.isGenerating) return;

  const promptText = $('prompt-input').value.trim();
  if (!promptText) {
    showToast('Please enter a generation prompt or choose a preset', 'error');
    $('prompt-input').focus();
    return;
  }

  // Determine active model
  let modelToUse = state.activeModel;
  if (state.customModelOverride && $('input-custom-model-id').value.trim()) {
    modelToUse = $('input-custom-model-id').value.trim();
  } else if (state.activeProvider === 'openrouter') {
    modelToUse = $('input-openrouter-model').value.trim() || 'anthropic/claude-3.7-sonnet';
  }

  const payload = {
    provider: state.activeProvider,
    model: modelToUse,
    prompt: promptText,
    system_prompt: $('system-prompt-text').value,
    temperature: parseFloat($('range-temp').value),
    max_tokens: parseInt($('range-tokens').value),
    thinking_level: state.thinkingLevel,
    use_search: $('toggle-search').checked
  };

  // Reset Canvas
  state.rawContent = '';
  state.thinkingContent = '';
  state.conversations = [];
  $('raw-output-text').value = '';
  $('think-code-content').textContent = '';
  $('conversation-container').innerHTML = '';
  $('empty-state').style.display = 'none';

  setGeneratingState(true);
  const startTime = Date.now();
  const timerInterval = setInterval(() => {
    if (!state.isGenerating) {
      clearInterval(timerInterval);
      return;
    }
    const sec = ((Date.now() - startTime) / 1000).toFixed(1);
    $('chip-time').textContent = `${sec}s`;
  }, 100);

  activeAbortController = new AbortController();

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: activeAbortController.signal
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep partial line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.substring(6).trim();
        if (!jsonStr) continue;

        try {
          const packet = JSON.parse(jsonStr);
          handleStreamPacket(packet);
        } catch (e) {
          // ignore malformed fragments
        }
      }
    }

  } catch (err) {
    if (err.name !== 'AbortError') {
      showToast(`Generation error: ${err.message}`, 'error');
    }
  } finally {
    setGeneratingState(false);
    activeAbortController = null;

    // Remove live thinking indicator
    const liveThink = $('live-thinking-indicator');
    if (liveThink) liveThink.remove();

    // If external thinking was streamed but not inside rawContent, inject it into the first <bot> turn
    if (state.thinkingContent.trim() && !state.rawContent.includes('<think>') && state.rawContent.includes('<bot>')) {
      state.rawContent = state.rawContent.replace(
        /<bot>(\s*)/i,
        `<bot>\n<think>\n${state.thinkingContent.trim()}\n</think>\n`
      );
      $('raw-output-text').value = state.rawContent;
    }

    parseAndRenderConversations();
    showToast('Generation complete', 'success');
  }
}

function handleStreamPacket(packet) {
  if (packet.error) {
    showToast(packet.error, 'error');
    return;
  }

  if (packet.type === 'thinking_start') {
    $('tab-thinking').style.display = 'inline-block';
  } else if (packet.type === 'thinking') {
    state.thinkingContent += packet.delta;
    $('think-code-content').textContent = state.thinkingContent;
    $('think-token-count').textContent = `${state.thinkingContent.length.toLocaleString()} chars`;

    // Render live thinking indicator card in Conversation Canvas so it is never empty
    let liveThink = $('live-thinking-indicator');
    if (!liveThink) {
      liveThink = document.createElement('div');
      liveThink.id = 'live-thinking-indicator';
      liveThink.className = 'conversation-card live-thinking-card';
      $('conversation-container').prepend(liveThink);
    }
    const snippet = state.thinkingContent.slice(-350);
    liveThink.innerHTML = `
      <div class="conv-index-header">
        <span style="color:var(--accent-amber); font-weight:600;">🧠 Model is reasoning... (${state.thinkingContent.length.toLocaleString()} chars)</span>
        <span style="color:var(--text-dim);">Live Trace</span>
      </div>
      <div class="turn-bubble bot" style="border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.04);">
        <pre style="font-family:var(--font-mono); font-size:11px; color:#cbd5e1; line-height:1.5; white-space:pre-wrap; word-break:break-word;">${escapeHtml(snippet)}</pre>
      </div>
    `;
  } else if (packet.type === 'content') {
    // Remove live indicator once content arrives
    const liveThink = $('live-thinking-indicator');
    if (liveThink) liveThink.remove();

    state.rawContent += packet.delta;
    $('raw-output-text').value = state.rawContent;
    $('chip-chars').textContent = `${state.rawContent.length.toLocaleString()} chars`;
    parseAndRenderConversations();
  }
}

function stopGeneration() {
  if (activeAbortController) {
    activeAbortController.abort();
    setGeneratingState(false);
    showToast('Generation halted', 'info');
  }
}

function setGeneratingState(isGen) {
  state.isGenerating = isGen;
  $('btn-generate').style.display = isGen ? 'none' : 'inline-flex';
  $('btn-stop').style.display = isGen ? 'inline-flex' : 'none';
  $('btn-generate').classList.toggle('loading', isGen);
}

// 5. ROBUST STREAMING PARSER & DOM RENDERER
function parseAndRenderConversations() {
  const container = $('conversation-container');
  const text = state.rawContent;
  if (!text || !text.trim()) {
    container.innerHTML = '';
    $('chip-turns').textContent = '0 turns';
    return;
  }

  // Regex that matches <user> or <bot> tags even if they have not closed yet (streaming)
  const markerRegex = /<(user|bot)>([\s\S]*?)(?:<\/\1>|(?=<(?:user|bot)>)|$)/gi;
  const turns = [];
  let match;

  while ((match = markerRegex.exec(text)) !== null) {
    const role = match[1].toLowerCase();
    const rawTurnText = match[2];
    if (rawTurnText.trim().length > 0 || role === 'bot') {
      turns.push({ role, rawText: rawTurnText });
    }
  }

  // Group into conversation dialogues
  const convs = [];
  let currentConv = [];

  for (const turn of turns) {
    if (turn.role === 'user' && currentConv.some(t => t.role === 'bot')) {
      convs.push(currentConv);
      currentConv = [turn];
    } else {
      currentConv.push(turn);
    }
  }
  if (currentConv.length > 0) {
    convs.push(currentConv);
  }

  let totalTurns = turns.length;
  container.innerHTML = '';

  convs.forEach((conv, cIdx) => {
    const card = document.createElement('div');
    card.className = 'conversation-card';

    const header = document.createElement('div');
    header.className = 'conv-index-header';
    header.innerHTML = `<span>Conversation #${cIdx + 1}</span><span>mini-AGI Format</span>`;
    card.appendChild(header);

    conv.forEach(turn => {
      const bubble = document.createElement('div');
      bubble.className = `turn-bubble ${turn.role}`;

      const headerRow = document.createElement('div');
      headerRow.className = 'turn-header-row';

      const roleTag = document.createElement('div');
      roleTag.className = 'turn-role-tag';
      roleTag.textContent = turn.role === 'user' ? 'User Turn' : 'mini-AGI Bot';
      headerRow.appendChild(roleTag);

      const turnActions = document.createElement('div');
      turnActions.className = 'turn-actions';
      turnActions.innerHTML = `
        <span class="sub-hint">${turn.rawText.length.toLocaleString()}c</span>
        <button class="turn-btn btn-turn-copy" title="Copy text">Copy</button>
        <button class="turn-btn danger btn-turn-del" title="Delete turn">✕</button>
      `;

      turnActions.querySelector('.btn-turn-copy').addEventListener('click', () => {
        navigator.clipboard.writeText(turn.rawText.trim());
        showToast('Turn copied to clipboard', 'success');
      });

      turnActions.querySelector('.btn-turn-del').addEventListener('click', () => {
        deleteTurn(turn);
      });

      headerRow.appendChild(turnActions);
      bubble.appendChild(headerRow);

      // Handle <think> tags (both closed and in-progress/open)
      let turnBody = turn.rawText.trim();
      let thinkContent = '';
      let actualResponse = turnBody;

      const thinkRegex = /<think>([\s\S]*?)(?:<\/think>|$)/i;
      const thinkMatch = thinkRegex.exec(turnBody);

      if (thinkMatch) {
        thinkContent = thinkMatch[1].trim();
        // The actual response is everything after </think> if closed, or empty if still inside <think>
        if (turnBody.includes('</think>')) {
          actualResponse = turnBody.split(/<\/think>/i).slice(1).join('</think>').trim();
        } else {
          actualResponse = ''; // Still thinking
        }
      }

      // 1. Render Reasoning Trace (if present)
      if (thinkContent.length > 0) {
        const thinkDetails = document.createElement('details');
        thinkDetails.className = 'think-accordion';
        thinkDetails.setAttribute('open', ''); // Keep open so user sees both reasoning and answer
        thinkDetails.innerHTML = `
          <summary class="think-summary">
            <span>🧠 Reasoning Trace (${thinkContent.length.toLocaleString()} chars)</span>
            <span>▾</span>
          </summary>
          <pre class="think-pre">${escapeHtml(thinkContent)}</pre>
        `;
        bubble.appendChild(thinkDetails);
      }

      // 2. Render Actual Response
      if (actualResponse.length > 0 || (turn.role === 'bot' && thinkContent.length === 0)) {
        const bodyEl = document.createElement('div');
        bodyEl.className = 'turn-content';
        bodyEl.textContent = actualResponse || (turn.role === 'bot' && !thinkContent ? '...' : '');
        bubble.appendChild(bodyEl);
      } else if (turn.role === 'bot' && thinkContent.length > 0 && actualResponse.length === 0) {
        const waitingEl = document.createElement('div');
        waitingEl.className = 'turn-content';
        waitingEl.style.color = 'var(--text-dim)';
        waitingEl.style.fontStyle = 'italic';
        waitingEl.textContent = 'Generating final response...';
        bubble.appendChild(waitingEl);
      }

      card.appendChild(bubble);
    });

    container.appendChild(card);
  });

  $('chip-turns').textContent = `${totalTurns} turns`;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function deleteTurn(targetTurn) {
  if (!confirm(`Delete this ${targetTurn.role.toUpperCase()} turn from dataset?`)) return;
  const escaped = targetTurn.rawText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const reg = new RegExp(`\\s*<${targetTurn.role}>\\s*${escaped}\\s*<\\/${targetTurn.role}>`, 'i');
  state.rawContent = state.rawContent.replace(reg, '').trim() + '\n';
  $('raw-output-text').value = state.rawContent;
  parseAndRenderConversations();
  validateXmlTags();
  showToast(`Deleted ${targetTurn.role} turn`, 'info');
}

// 6. COPY & CLEAR
function copyContent() {
  const content = state.rawContent || $('raw-output-text').value;
  if (!content) {
    showToast('Nothing to copy', 'error');
    return;
  }
  navigator.clipboard.writeText(content).then(() => {
    showToast('Copied raw XML to clipboard', 'success');
  });
}

function clearContent() {
  state.rawContent = '';
  state.thinkingContent = '';
  $('raw-output-text').value = '';
  $('think-code-content').textContent = '';
  $('conversation-container').innerHTML = '';
  $('empty-state').style.display = 'flex';
  $('chip-chars').textContent = '0 chars';
  $('chip-turns').textContent = '0 turns';
  $('chip-time').textContent = '0.0s';
  showToast('Canvas cleared', 'info');
}

// 7. DATASET SYNC
async function saveDocument() {
  const content = state.rawContent || $('raw-output-text').value;
  if (!content || content.trim().length === 0) {
    showToast('No generated content to save', 'error');
    return;
  }

  const title = $('doc-title').value.trim() || `synthetic_dialogue_${Date.now()}`;
  const destinations = [];
  if ($('dest-custom').checked) destinations.push('custom');
  if ($('dest-conv').checked) destinations.push('conversation');
  if ($('dest-exports').checked) destinations.push('exports');

  if (destinations.length === 0) {
    showToast('Select at least one destination', 'error');
    return;
  }

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: title,
        content: content,
        destinations: destinations
      })
    });

    const data = await res.json();
    if (data.success) {
      showToast(`Saved ${data.chars.toLocaleString()} chars to ${data.paths.length} destinations!`, 'success');
    } else {
      showToast(data.error || 'Failed to save', 'error');
    }
  } catch (err) {
    showToast(`Save error: ${err.message}`, 'error');
  }
}

// 8. TOAST SYSTEM
function showToast(msg, type = 'info') {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3200);
}

// =========================================================================
// 9. TEACHER ⇄ STUDENT ARENA SPARRING ENGINE
// =========================================================================

function switchMode(mode) {
  state.currentMode = mode;
  $('btn-mode-studio').classList.toggle('active', mode === 'studio');
  $('btn-mode-arena').classList.toggle('active', mode === 'arena');

  if (mode === 'studio') {
    $('sidebar-studio-view').style.display = 'block';
    $('sidebar-arena-view').style.display = 'none';
    $('studio-main-view').style.display = 'flex';
    $('arena-main-view').style.display = 'none';
    if (arenaPollInterval) {
      clearInterval(arenaPollInterval);
      arenaPollInterval = null;
    }
  } else {
    $('sidebar-studio-view').style.display = 'none';
    $('sidebar-arena-view').style.display = 'block';
    $('studio-main-view').style.display = 'none';
    $('arena-main-view').style.display = 'flex';
    selectArenaProvider(arenaState.teacherProvider);
    fetchStudentState();
    if (!arenaPollInterval) {
      arenaPollInterval = setInterval(fetchStudentState, 3500);
    }
  }
}

async function fetchStudentState() {
  try {
    const res = await fetch('/api/student/state');
    const data = await res.json();
    arenaState.studentOnline = data.online;
    const pulseDot = $('student-pulse-dot');
    const statusBadge = $('student-status-badge');

    if (data.online) {
      if (pulseDot) pulseDot.className = 'pulse-dot active';
      if (statusBadge) {
        statusBadge.className = 'student-status-badge';
        statusBadge.textContent = 'Connected';
      }
      if (data.pool) {
        const slotsEl = $('student-resident-slots');
        if (slotsEl) {
          slotsEl.textContent = `${data.pool.n_slots} / ${data.pool.n_experts} pool`;
        }
      }
      if (data.learn) {
        const learnEl = $('student-learn-status');
        const pendingEl = $('student-pending-chars');
        if (learnEl) {
          learnEl.textContent = `Online Backprop (Step ${data.learn.steps})`;
          learnEl.style.color = 'var(--accent-emerald)';
        }
        if (pendingEl) {
          pendingEl.textContent = `${data.learn.pending} / ${data.learn.chunk} chars`;
        }
      } else {
        const learnEl = $('student-learn-status');
        if (learnEl) {
          learnEl.textContent = 'Read-Only (--no-learn)';
          learnEl.style.color = 'var(--text-dim)';
        }
      }
    } else {
      if (pulseDot) pulseDot.className = 'pulse-dot';
      if (statusBadge) {
        statusBadge.className = 'student-status-badge offline';
        statusBadge.textContent = 'Offline';
      }
    }
  } catch (err) {
    arenaState.studentOnline = false;
    const statusBadge = $('student-status-badge');
    if (statusBadge) {
      statusBadge.className = 'student-status-badge offline';
      statusBadge.textContent = 'Offline';
    }
  }
}

function selectArenaProvider(providerId) {
  arenaState.teacherProvider = providerId;
  document.querySelectorAll('#sidebar-arena-view .prov-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.provider === providerId);
  });

  const pdata = state.providers[providerId];
  if (!pdata) return;

  const selectEl = $('select-arena-model');
  const openrouterBox = $('box-arena-openrouter-input');
  const selectBox = $('box-arena-model-select');

  if (pdata.type_in) {
    selectBox.style.display = 'none';
    openrouterBox.style.display = 'block';
    if (!$('input-arena-openrouter-model').value) {
      $('input-arena-openrouter-model').value = pdata.default_model || 'anthropic/claude-3.7-sonnet';
    }
    arenaState.teacherModel = $('input-arena-openrouter-model').value;
  } else {
    selectBox.style.display = 'block';
    openrouterBox.style.display = 'none';
    selectEl.innerHTML = '';
    (pdata.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      selectEl.appendChild(opt);
    });
    arenaState.teacherModel = selectEl.value;
  }
  updateArenaTeacherBadge();
}

function updateArenaTeacherBadge() {
  const badge = $('arena-teacher-badge');
  if (badge) {
    badge.textContent = arenaState.teacherModel.split('/').pop();
  }
}

function setArenaThinkingLevel(level) {
  arenaState.teacherThinking = level;
  document.querySelectorAll('#arena-thinking-levels .t-level-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.level === level);
  });
  const labelEl = $('arena-think-level-label');
  if (labelEl && THINKING_DESCRIPTIONS[level]) {
    labelEl.textContent = THINKING_DESCRIPTIONS[level].label;
  }
}

async function toggleArenaSession() {
  if (arenaState.active && !arenaState.paused) {
    // Pause
    arenaState.paused = true;
    $('btn-arena-start').innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Resume Arena</span>
    `;
    $('arena-current-round-badge').textContent = 'Session Paused';
    showToast('Arena paused', 'info');
  } else if (arenaState.active && arenaState.paused) {
    // Resume
    arenaState.paused = false;
    $('btn-arena-start').innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
      <span>Pause Arena</span>
    `;
    showToast('Resuming arena...', 'info');
    runNextArenaTurn();
  } else {
    // Start fresh
    if (!arenaState.studentOnline) {
      await fetchStudentState();
      if (!arenaState.studentOnline) {
        showToast('mini-AGI (port 8080) is not connected', 'error');
        return;
      }
    }
    arenaState.active = true;
    arenaState.paused = false;
    arenaState.currentRound = 0;
    arenaState.history = [];
    $('arena-empty-state').style.display = 'none';
    $('arena-stream-flow').innerHTML = '';
    $('btn-arena-start').innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
      <span>Pause Arena</span>
    `;
    showToast('Starting Arena sparring session', 'success');
    runNextArenaTurn();
  }
}

async function stepArenaSession() {
  if (!arenaState.studentOnline) {
    await fetchStudentState();
    if (!arenaState.studentOnline) {
      showToast('mini-AGI is offline', 'error');
      return;
    }
  }
  if (!arenaState.active) {
    arenaState.active = true;
    arenaState.paused = true;
    arenaState.currentRound = 0;
    arenaState.history = [];
    $('arena-empty-state').style.display = 'none';
    $('arena-stream-flow').innerHTML = '';
  }
  arenaState.paused = true; // Stay in manual step mode
  $('btn-arena-start').innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    <span>Resume Arena</span>
  `;
  runNextArenaTurn();
}

async function runNextArenaTurn() {
  if (arenaState.currentRound >= arenaState.maxRounds) {
    arenaState.active = false;
    arenaState.paused = false;
    $('btn-arena-start').innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Start Arena Session</span>
    `;
    $('arena-current-round-badge').textContent = 'Session Complete';
    showToast(`Completed all ${arenaState.maxRounds} sparring rounds!`, 'success');
    return;
  }

  arenaState.currentRound++;
  const round = arenaState.currentRound;
  $('arena-current-round-badge').textContent = `Round ${round} of ${arenaState.maxRounds}`;
  $('chip-arena-rounds').textContent = `${round} / ${arenaState.maxRounds} rounds`;

  // 1. Create Round Container
  const flow = $('arena-stream-flow');
  const roundCard = document.createElement('div');
  roundCard.className = 'arena-round-card';
  roundCard.id = `round-card-${round}`;

  const banner = document.createElement('div');
  banner.className = 'arena-round-banner';
  banner.textContent = `Round #${round} · Sparring Exchange`;
  roundCard.appendChild(banner);

  // 2. Create Teacher Card
  const teacherCard = document.createElement('div');
  teacherCard.className = 'teacher-turn-card';
  teacherCard.id = `teacher-card-${round}`;
  teacherCard.innerHTML = `
    <div class="turn-header">
      <div class="turn-role-group">
        <span class="turn-role-tag teacher-tag">Teacher</span>
        <span class="turn-model-name">${escapeHtml(arenaState.teacherModel)}</span>
      </div>
      <span class="model-badge" id="teacher-status-${round}">Generating...</span>
    </div>
    <details class="arena-trace-accordion" id="teacher-trace-${round}" style="display: none;" open>
      <summary class="arena-trace-summary">🧠 Teacher Reasoning Trace</summary>
      <pre class="arena-trace-body" id="teacher-trace-body-${round}"></pre>
    </details>
    <div class="turn-content" id="teacher-content-${round}"></div>
  `;
  roundCard.appendChild(teacherCard);
  flow.appendChild(roundCard);
  flow.scrollTop = flow.scrollHeight;

  // 3. Stream Teacher Turn
  let teacherText = '';
  let teacherThinking = '';
  try {
    const customGoal = $('arena-custom-curriculum-text') ? $('arena-custom-curriculum-text').value.trim() : '';
    const res = await fetch('/api/arena/teacher_turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: arenaState.teacherProvider,
        model: arenaState.teacherModel,
        thinking_level: arenaState.teacherThinking,
        curriculum: arenaState.curriculum,
        custom_goal: customGoal,
        history: arenaState.history
      })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const packet = JSON.parse(line.slice(6));
          if (packet.error) {
            showToast(`Teacher error: ${packet.error}`, 'error');
            return;
          }
          if (packet.type === 'thinking_start') {
            const trace = $(`teacher-trace-${round}`);
            if (trace) trace.style.display = 'block';
          } else if (packet.type === 'thinking') {
            teacherThinking += packet.delta;
            const tb = $(`teacher-trace-body-${round}`);
            if (tb) tb.textContent = teacherThinking;
            flow.scrollTop = flow.scrollHeight;
          } else if (packet.type === 'content') {
            teacherText += packet.delta;
            const tc = $(`teacher-content-${round}`);
            if (tc) tc.textContent = teacherText;
            flow.scrollTop = flow.scrollHeight;
          }
        } catch (e) {}
      }
    }
  } catch (err) {
    showToast(`Teacher stream error: ${err.message}`, 'error');
    return;
  }

  const teacherStatus = $(`teacher-status-${round}`);
  if (teacherStatus) teacherStatus.textContent = 'Challenge Posed';

  // Check if paused before student starts
  if (arenaState.paused && !arenaState.autoPlay) {
    return;
  }

  // 4. Create Student Card
  const studentCard = document.createElement('div');
  studentCard.className = 'student-turn-card';
  studentCard.id = `student-card-${round}`;
  studentCard.innerHTML = `
    <div class="turn-header">
      <div class="turn-role-group">
        <span class="turn-role-tag student-tag">Student</span>
        <span class="turn-model-name">mini-AGI (Local Online Learner)</span>
      </div>
      <span class="model-badge" id="student-status-${round}">Responding...</span>
    </div>
    <div class="turn-content mono-font" id="student-content-${round}"></div>
    <div class="student-telemetry-footer">
      <span class="telemetry-chip highlight" id="student-speed-${round}">⚡ Streaming...</span>
      <span class="telemetry-chip" id="student-slots-${round}">Pool: 32 slots</span>
      <span class="telemetry-chip" id="student-learn-${round}">Online SGD Active</span>
    </div>
  `;
  roundCard.appendChild(studentCard);
  flow.scrollTop = flow.scrollHeight;

  // 5. Build Student Messages History & Stream Student Turn
  const studentMessages = [];
  for (const h of arenaState.history) {
    studentMessages.push({ role: 'user', content: h.teacher });
    studentMessages.push({ role: 'assistant', content: h.student });
  }
  studentMessages.push({ role: 'user', content: teacherText });

  let studentText = '';
  let lastTelemetry = null;
  try {
    const res = await fetch('/api/student/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: studentMessages,
        max_new: arenaState.studentMaxChars
      })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const packet = JSON.parse(line.slice(6));
          if (packet.error) {
            showToast(`mini-AGI error: ${packet.error}`, 'error');
            return;
          }
          if (packet.t) {
            studentText += packet.t;
            const sc = $(`student-content-${round}`);
            if (sc) sc.textContent = studentText;
            flow.scrollTop = flow.scrollHeight;
          }
          if (packet.swap) {
            const slotsChip = $(`student-slots-${round}`);
            if (slotsChip && packet.swap.pool) {
              slotsChip.textContent = `${packet.swap.pool.n_slots} slots (${packet.swap.pool.used.length} active)`;
            }
          }
          if (packet.done) {
            lastTelemetry = packet;
            const speedChip = $(`student-speed-${round}`);
            if (speedChip) speedChip.textContent = `⚡ ${packet.cps} chars/s`;
            const speedGlobal = $('chip-arena-speed');
            if (speedGlobal) speedGlobal.textContent = `${packet.cps} chars/s`;

            if (packet.learned) {
              const learnChip = $(`student-learn-${round}`);
              const lossChip = $('chip-arena-loss');
              if (learnChip) {
                const lossText = packet.learned.loss !== undefined ? `Loss: ${packet.learned.loss}` : `Pending: ${packet.learned.pending}/${packet.learned.chunk}`;
                learnChip.textContent = `Backprop: ${lossText} (Step ${packet.learned.total})`;
              }
              if (lossChip && packet.learned.loss !== undefined) {
                lossChip.textContent = `Loss: ${packet.learned.loss}`;
              }
            }
          }
        } catch (e) {}
      }
    }
  } catch (err) {
    showToast(`mini-AGI stream error: ${err.message}`, 'error');
    return;
  }

  const studentStatus = $(`student-status-${round}`);
  if (studentStatus) studentStatus.textContent = 'Turn Completed';

  // Record into History
  arenaState.history.push({
    round: round,
    teacher: teacherText,
    teacherThinking: teacherThinking,
    student: studentText,
    telemetry: lastTelemetry
  });

  // Update total chars
  let totalChars = 0;
  for (const h of arenaState.history) {
    totalChars += (h.teacher.length + h.student.length);
  }
  $('chip-arena-chars').textContent = `${totalChars.toLocaleString()} chars`;

  // Check next step
  if (arenaState.autoPlay && !arenaState.paused && arenaState.currentRound < arenaState.maxRounds) {
    setTimeout(runNextArenaTurn, 1400);
  } else if (arenaState.currentRound >= arenaState.maxRounds) {
    arenaState.active = false;
    arenaState.paused = false;
    $('btn-arena-start').innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Start Arena Session</span>
    `;
    $('arena-current-round-badge').textContent = 'Session Complete';
    showToast('All rounds finished! Click "Save Arena to Datasets" to export.', 'success');
  }
}

function resetArenaSession() {
  arenaState.active = false;
  arenaState.paused = false;
  arenaState.currentRound = 0;
  arenaState.history = [];
  $('arena-empty-state').style.display = 'flex';
  $('arena-stream-flow').innerHTML = '';
  $('arena-current-round-badge').textContent = 'Ready to Spar';
  $('chip-arena-rounds').textContent = `0 / ${arenaState.maxRounds} rounds`;
  $('chip-arena-chars').textContent = '0 chars';
  $('chip-arena-speed').textContent = '-- chars/s';
  $('chip-arena-loss').textContent = 'Loss: --';
  $('btn-arena-start').innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
    <span>Start Arena Session</span>
  `;
  showToast('Arena reset', 'info');
}

function injectArenaTurn() {
  const input = $('arena-intervention-input');
  const text = input.value.trim();
  if (!text) {
    showToast('Enter text to inject', 'error');
    return;
  }
  input.value = '';

  // Render Human Intervention Card in the stream
  const flow = $('arena-stream-flow');
  const card = document.createElement('div');
  card.className = 'teacher-turn-card';
  card.style.borderColor = 'rgba(245, 158, 11, 0.4)';
  card.style.borderLeftColor = 'var(--accent-amber)';
  card.innerHTML = `
    <div class="turn-header">
      <div class="turn-role-group">
        <span class="turn-role-tag" style="background: rgba(245, 158, 11, 0.15); color: var(--accent-amber);">Human Director</span>
        <span class="turn-model-name">Intervention</span>
      </div>
      <span class="model-badge">Injected</span>
    </div>
    <div class="turn-content">${escapeHtml(text)}</div>
  `;
  flow.appendChild(card);
  flow.scrollTop = flow.scrollHeight;

  // Add as teacher turn in history so mini-AGI will respond to it
  arenaState.history.push({
    round: arenaState.currentRound + 1,
    teacher: text,
    teacherThinking: '',
    student: ''
  });

  showToast('Injected director turn into arena', 'success');
  if (!arenaState.active) {
    arenaState.active = true;
    arenaState.paused = false;
  }
}

async function saveArenaToDatasets() {
  if (arenaState.history.length === 0) {
    showToast('No arena dialogue to save', 'error');
    return;
  }

  // Format into standard mini-AGI multi-turn XML
  let doc = '';
  for (const h of arenaState.history) {
    if (!h.teacher || !h.student) continue;
    doc += `<user>\n${h.teacher.trim()}\n</user>\n`;
    if (h.teacherThinking) {
      doc += `<bot>\n<think>\n${h.teacherThinking.trim()}\n</think>\n${h.student.trim()}\n</bot>\n\n`;
    } else {
      doc += `<bot>\n${h.student.trim()}\n</bot>\n\n`;
    }
  }

  const title = `arena_training_${arenaState.curriculum}_${Date.now()}`;
  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: title,
        content: doc.trim(),
        destinations: ['custom', 'conversation', 'exports']
      })
    });

    const data = await res.json();
    if (data.success) {
      showToast(`Saved Arena sparring session (${data.chars.toLocaleString()} chars) to datasets!`, 'success');
    } else {
      showToast(data.error || 'Failed to save arena dialogue', 'error');
    }
  } catch (err) {
    showToast(`Save error: ${err.message}`, 'error');
  }
}

