/**
 * Sesame Voice Arena - Web Audio Pipeline & Voice Synthesis
 * Handles microphone capture, audio analysis, and voice-to-voice stream playback.
 */

class AudioManager {
  constructor() {
    this.audioCtx = null;
    this.analyser = null;
    this.micStream = null;
    this.micSource = null;
    this.isPlaying = false;
    this.isRecording = false;

    // Speech synthesis instance
    this.synth = window.speechSynthesis || null;
    this.activeUtterance = null;

    // Callbacks
    this.onSpeechStart = null;
    this.onSpeechEnd = null;
    this.onMicData = null;
  }

  async initAudioContext() {
    if (!this.audioCtx) {
      const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtxClass();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;
    }
    if (this.audioCtx.state === 'suspended') {
      await this.audioCtx.resume();
    }
    return this.analyser;
  }

  /**
   * Start capturing microphone audio for real-time analysis
   */
  async startMicrophone() {
    await this.initAudioContext();
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      this.micSource = this.audioCtx.createMediaStreamSource(this.micStream);
      this.micSource.connect(this.analyser);
      this.isRecording = true;
      return true;
    } catch (err) {
      console.warn("Microphone access denied or unavailable:", err);
      return false;
    }
  }

  stopMicrophone() {
    if (this.micStream) {
      this.micStream.getTracks().forEach(track => track.stop());
      this.micStream = null;
    }
    if (this.micSource) {
      try { this.micSource.disconnect(); } catch (e) {}
      this.micSource = null;
    }
    this.isRecording = false;
  }

  /**
   * Stop any current speech or audio immediately (barge-in / interruption)
   */
  interrupt() {
    if (this.synth) {
      this.synth.cancel();
    }
    this.isPlaying = false;
    if (this.onSpeechEnd) {
      this.onSpeechEnd();
    }
  }

  /**
   * Play speech for an agent using available engines (CSM remote server or local Web Speech fallback)
   */
  async speakText(text, agentConfig, csmServerUrl = '') {
    await this.initAudioContext();
    this.interrupt();

    // 1. Try remote CSM / Voice Server if configured and available
    if (csmServerUrl && agentConfig.voiceEngine !== 'web-speech') {
      const success = await this.tryRemoteSynthesize(text, agentConfig, csmServerUrl);
      if (success) return;
    }

    // 2. High-quality browser neural speech synthesis fallback
    return this.speakBrowserTTS(text, agentConfig);
  }

  async tryRemoteSynthesize(text, agentConfig, csmServerUrl) {
    try {
      const response = await fetch(`${csmServerUrl}/sessions/demo/synthesize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agentId: agentConfig.id,
          text: text
        })
      });

      if (!response.ok) return false;

      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
      
      const source = this.audioCtx.createBufferSource();
      source.buffer = audioBuffer;

      // Connect to analyser for orb reactivity, then to speakers
      source.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);

      this.isPlaying = true;
      if (this.onSpeechStart) this.onSpeechStart();

      source.onended = () => {
        this.isPlaying = false;
        try { source.disconnect(); } catch (e) {}
        if (this.onSpeechEnd) this.onSpeechEnd();
      };

      source.start(0);
      return true;
    } catch (e) {
      console.warn("Remote voice server unavailable, falling back to local TTS:", e);
      return false;
    }
  }

  speakBrowserTTS(text, agentConfig) {
    return new Promise((resolve) => {
      if (!this.synth) {
        resolve();
        return;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      this.activeUtterance = utterance;

      // Custom voice modulation for distinct Agent personas
      if (agentConfig.id === 'agent-a') {
        // Maya / Optimist: slightly higher, bright, expressive
        utterance.pitch = agentConfig.tone === 'energetic' ? 1.25 : 1.1;
        utterance.rate = 1.05;
      } else {
        // Miles / Pragmatist: grounded, deeper, articulate
        utterance.pitch = agentConfig.tone === 'deep' ? 0.85 : 0.95;
        utterance.rate = 0.98;
      }

      // Pick distinct voice if available in system
      const voices = this.synth.getVoices();
      if (voices.length > 0) {
        if (agentConfig.id === 'agent-a') {
          const femaleVoice = voices.find(v => /female|natural|google|zira|samantha/i.test(v.name));
          if (femaleVoice) utterance.voice = femaleVoice;
        } else {
          const maleVoice = voices.find(v => /male|david|alex|daniel/i.test(v.name));
          if (maleVoice) utterance.voice = maleVoice;
        }
      }

      // Generate synthetic energy through oscillator for visualizer feedback
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      gain.gain.value = 0.001; // Silent, just drives analyser
      osc.frequency.value = agentConfig.id === 'agent-a' ? 240 : 130;
      osc.connect(this.analyser);

      utterance.onstart = () => {
        this.isPlaying = true;
        try { osc.start(); } catch (e) {}
        if (this.onSpeechStart) this.onSpeechStart();
      };

      utterance.onend = () => {
        this.isPlaying = false;
        try { osc.stop(); osc.disconnect(); } catch (e) {}
        if (this.onSpeechEnd) this.onSpeechEnd();
        resolve();
      };

      utterance.onerror = () => {
        this.isPlaying = false;
        try { osc.stop(); osc.disconnect(); } catch (e) {}
        if (this.onSpeechEnd) this.onSpeechEnd();
        resolve();
      };

      this.synth.speak(utterance);
    });
  }
}

window.AudioManager = AudioManager;
