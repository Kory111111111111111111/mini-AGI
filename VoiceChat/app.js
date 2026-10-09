/**
 * Sesame Voice Arena - Main Application Orchestrator
 * Integrates VoiceOrb, AudioManager, DebateEngine, and interactive UI controls.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Core Modules
  const audioManager = new AudioManager();
  const voiceOrb = new VoiceOrb('voice-orb-canvas');
  const debateEngine = new DebateEngine(audioManager, voiceOrb);

  // Connect Audio Analyser to Orb for fluid reactivity
  audioManager.initAudioContext().then(analyser => {
    voiceOrb.setAudioAnalyser(analyser);
  });

  // DOM Elements
  const btnStartDebate = document.getElementById('btn-start-debate');
  const btnPauseDebate = document.getElementById('btn-pause-debate');
  const btnHoldToTalk = document.getElementById('btn-hold-to-talk');
  const debateTopicInput = document.getElementById('debate-topic');
  const directorPromptInput = document.getElementById('director-prompt');
  const btnDirectorSend = document.getElementById('btn-director-send');

  const stageSpeakerTag = document.getElementById('stage-speaker-tag');
  const stageSubtext = document.getElementById('stage-subtext');
  const indicatorA = document.getElementById('indicator-agent-a');
  const indicatorB = document.getElementById('indicator-agent-b');
  const panelA = document.getElementById('panel-agent-a');
  const panelB = document.getElementById('panel-agent-b');

  const transcriptFeed = document.getElementById('transcript-feed');
  const btnClearTranscript = document.getElementById('btn-clear-transcript');
  const btnExportTranscript = document.getElementById('btn-export-transcript');
  const btnToggleDrawer = document.getElementById('btn-toggle-drawer');
  const transcriptDrawer = document.getElementById('transcript-drawer');

  // Mode Switchers
  const btnModeDebate = document.getElementById('btn-mode-debate');
  const btnModeDirect = document.getElementById('btn-mode-direct');
  let currentMode = 'debate'; // 'debate' | 'direct'

  // Settings Modal
  const settingsModal = document.getElementById('settings-modal');
  const btnToggleSettings = document.getElementById('btn-toggle-settings');
  const btnCloseSettings = document.getElementById('btn-close-settings');

  // Helper: Read Agent Configuration
  function getAgentConfigs() {
    return {
      agentA: {
        id: 'agent-a',
        name: document.getElementById('agent-a-name').value || 'Agent A',
        voiceEngine: document.getElementById('agent-a-voice-model').value,
        brain: document.getElementById('agent-a-brain-model').value,
        tone: document.getElementById('agent-a-tone').value,
        stance: document.getElementById('agent-a-stance').value
      },
      agentB: {
        id: 'agent-b',
        name: document.getElementById('agent-b-name').value || 'Agent B',
        voiceEngine: document.getElementById('agent-b-voice-model').value,
        brain: document.getElementById('agent-b-brain-model').value,
        tone: document.getElementById('agent-b-tone').value,
        stance: document.getElementById('agent-b-stance').value
      }
    };
  }

  // --- Debate Engine Callbacks ---
  debateEngine.onTurnStart = (speakerId, agent) => {
    stageSpeakerTag.textContent = `${agent.name} is speaking...`;
    stageSpeakerTag.style.color = speakerId === 'agent-a' ? 'var(--accent-cyan)' : 'var(--accent-violet)';
    stageSubtext.textContent = `Generating voice-to-voice stream via ${agent.voiceEngine}`;

    if (speakerId === 'agent-a') {
      panelA.querySelector('.agent-card').classList.add('speaking');
      panelB.querySelector('.agent-card').classList.remove('speaking');
    } else {
      panelB.querySelector('.agent-card').classList.add('speaking');
      panelA.querySelector('.agent-card').classList.remove('speaking');
    }
  };

  debateEngine.onTurnEnd = (speakerId) => {
    panelA.querySelector('.agent-card').classList.remove('speaking');
    panelB.querySelector('.agent-card').classList.remove('speaking');
    stageSpeakerTag.textContent = 'CONVERSATIONAL GAP';
    stageSpeakerTag.style.color = 'var(--text-secondary)';
    stageSubtext.textContent = 'Listening and evaluating rebuttal...';
  };

  debateEngine.onTranscriptUpdate = (entry) => {
    // Clear empty state hint if present
    const hint = transcriptFeed.querySelector('.empty-state-hint');
    if (hint) hint.remove();

    const turnDiv = document.createElement('div');
    const speakerClass = entry.speaker === 'agent-a' ? 'turn-a' : (entry.speaker === 'agent-b' ? 'turn-b' : 'turn-director');
    turnDiv.className = `transcript-turn ${speakerClass}`;

    turnDiv.innerHTML = `
      <div class="turn-header">
        <span class="turn-speaker">${escapeHTML(entry.name)}</span>
        <span class="turn-time">${entry.timestamp}</span>
      </div>
      <div class="turn-text">${escapeHTML(entry.text)}</div>
    `;

    transcriptFeed.appendChild(turnDiv);
    transcriptFeed.scrollTop = transcriptFeed.scrollHeight;
  };

  // --- Start / Pause / Stop Controls ---
  btnStartDebate.addEventListener('click', () => {
    if (!debateEngine.isRunning) {
      const topic = debateTopicInput.value.trim() || 'General AI Ethics and DSP';
      const configs = getAgentConfigs();
      debateEngine.configure(topic, configs.agentA, configs.agentB, {
        turnDelay: parseFloat(document.getElementById('cfg-turn-delay')?.value || 1.2),
        maxTurns: parseInt(document.getElementById('cfg-max-turns')?.value || 10)
      });
      debateEngine.start();

      btnStartDebate.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16"/></svg>
        <span>Stop Debate</span>
      `;
      btnStartDebate.style.background = 'linear-gradient(135deg, var(--accent-rose), #dc2626)';
      btnPauseDebate.disabled = false;
    } else {
      debateEngine.stop();
      resetDebateControls();
    }
  });

  btnPauseDebate.addEventListener('click', () => {
    if (debateEngine.isPaused) {
      debateEngine.resume();
      btnPauseDebate.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
        <span>Pause</span>
      `;
    } else {
      debateEngine.pause();
      btnPauseDebate.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Resume</span>
      `;
      stageSpeakerTag.textContent = 'DEBATE PAUSED';
      stageSubtext.textContent = 'Click Resume or Hold to Talk';
    }
  });

  function resetDebateControls() {
    btnStartDebate.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <span>Start Debate</span>
    `;
    btnStartDebate.style.background = 'linear-gradient(135deg, var(--accent-cyan), var(--accent-blue))';
    btnPauseDebate.disabled = true;
    panelA.querySelector('.agent-card').classList.remove('speaking');
    panelB.querySelector('.agent-card').classList.remove('speaking');
    stageSpeakerTag.textContent = 'IDLE / READY';
    stageSpeakerTag.style.color = 'var(--accent-cyan)';
    stageSubtext.textContent = 'Click Start Debate or speak through mic';
  }

  // --- Director Interjections ---
  function sendDirectorInterjection() {
    const text = directorPromptInput.value.trim();
    if (!text) return;
    debateEngine.interject(text);
    directorPromptInput.value = '';
  }

  btnDirectorSend.addEventListener('click', sendDirectorInterjection);
  directorPromptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendDirectorInterjection();
  });

  // --- Hold to Talk / Microphone Barge-In ---
  let isHoldingTalk = false;

  btnHoldToTalk.addEventListener('mousedown', async () => {
    isHoldingTalk = true;
    btnHoldToTalk.classList.add('active');
    stageSpeakerTag.textContent = 'LISTENING TO YOU (BARGE-IN)';
    stageSpeakerTag.style.color = 'var(--accent-rose)';
    stageSubtext.textContent = 'Speak into your microphone...';

    // Interrupt AI speech immediately
    audioManager.interrupt();
    voiceOrb.setState('listening', 'user');
    await audioManager.startMicrophone();
  });

  window.addEventListener('mouseup', () => {
    if (isHoldingTalk) {
      isHoldingTalk = false;
      btnHoldToTalk.classList.remove('active');
      audioManager.stopMicrophone();

      stageSpeakerTag.textContent = 'PROCESSING INTERVENTION';
      stageSpeakerTag.style.color = 'var(--accent-amber)';
      stageSubtext.textContent = 'Agents evaluating your remark...';

      // Simulate transcribed speech intervention
      debateEngine.interject("I want to steer both of you back to the core point: how does this affect practical latency and human musicianship?");
    }
  });

  // --- Mode Switcher ---
  btnModeDebate.addEventListener('click', () => {
    currentMode = 'debate';
    btnModeDebate.classList.add('active');
    btnModeDirect.classList.remove('active');
    panelA.style.display = 'flex';
    panelB.style.display = 'flex';
    document.getElementById('debate-topic-bar').style.display = 'block';
  });

  btnModeDirect.addEventListener('click', () => {
    currentMode = 'direct';
    btnModeDirect.classList.add('active');
    btnModeDebate.classList.remove('active');
    panelB.style.display = 'none'; // Only Agent A is active for direct chat
    document.getElementById('debate-topic-bar').style.display = 'none';
    stageSpeakerTag.textContent = 'DIRECT VOICE MODE';
    stageSubtext.textContent = 'Press Hold to Talk or click the Orb to speak with Maya';
  });

  // Click on Orb to toggle microphone talk
  document.getElementById('voice-orb-canvas').addEventListener('click', () => {
    if (!audioManager.isRecording) {
      btnHoldToTalk.dispatchEvent(new MouseEvent('mousedown'));
    } else {
      window.dispatchEvent(new MouseEvent('mouseup'));
    }
  });

  // --- Transcript Drawer & Export ---
  btnClearTranscript.addEventListener('click', () => {
    transcriptFeed.innerHTML = '<div class="empty-state-hint">Transcript cleared.</div>';
    debateEngine.history = [];
  });

  btnExportTranscript.addEventListener('click', () => {
    if (debateEngine.history.length === 0) {
      alert("No conversation history to export yet.");
      return;
    }
    let content = `# Sesame Voice Arena - Debate Transcript\n`;
    content += `**Topic:** ${debateEngine.topic}\n`;
    content += `**Date:** ${new Date().toLocaleString()}\n\n---\n\n`;

    debateEngine.history.forEach(turn => {
      content += `### [${turn.timestamp}] ${turn.name}\n${turn.text}\n\n`;
    });

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voice-debate-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });

  btnToggleDrawer.addEventListener('click', () => {
    if (transcriptDrawer.style.maxHeight === '45px') {
      transcriptDrawer.style.maxHeight = '250px';
      btnToggleDrawer.querySelector('svg').style.transform = 'rotate(0deg)';
    } else {
      transcriptDrawer.style.maxHeight = '45px';
      btnToggleDrawer.querySelector('svg').style.transform = 'rotate(180deg)';
    }
  });

  // Load saved settings from localStorage
  const deepseekInput = document.getElementById('cfg-deepseek-key');
  const geminiInput = document.getElementById('cfg-gemini-key');
  const csmUrlInput = document.getElementById('cfg-csm-url');
  const miniAgiUrlInput = document.getElementById('cfg-mini-agi-url');

  if (deepseekInput && localStorage.getItem('voice_deepseek_key')) {
    deepseekInput.value = localStorage.getItem('voice_deepseek_key');
  }
  if (geminiInput && localStorage.getItem('voice_gemini_key')) {
    geminiInput.value = localStorage.getItem('voice_gemini_key');
  }
  if (csmUrlInput && localStorage.getItem('voice_csm_url')) {
    csmUrlInput.value = localStorage.getItem('voice_csm_url');
  }
  if (miniAgiUrlInput && localStorage.getItem('voice_mini_agi_url')) {
    miniAgiUrlInput.value = localStorage.getItem('voice_mini_agi_url');
  }

  // --- Settings Modal ---
  btnToggleSettings.addEventListener('click', () => {
    settingsModal.style.display = 'flex';
  });

  btnCloseSettings.addEventListener('click', () => {
    settingsModal.style.display = 'none';
    // Save to localStorage
    if (deepseekInput) localStorage.setItem('voice_deepseek_key', deepseekInput.value);
    if (geminiInput) localStorage.setItem('voice_gemini_key', geminiInput.value);
    if (csmUrlInput) localStorage.setItem('voice_csm_url', csmUrlInput.value);
    if (miniAgiUrlInput) localStorage.setItem('voice_mini_agi_url', miniAgiUrlInput.value);
  });

  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
      settingsModal.style.display = 'none';
    }
  });

  function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag));
  }
});
