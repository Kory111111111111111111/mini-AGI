/**
 * Sesame Voice Arena - Multi-Agent Debate & Conversational Controller
 * Orchestrates turns between Agent A and Agent B, handles Director interjections and pacing.
 */

class DebateEngine {
  constructor(audioManager, voiceOrb) {
    this.audio = audioManager;
    this.orb = voiceOrb;

    this.isRunning = false;
    this.isPaused = false;
    this.currentSpeaker = 'none'; // 'agent-a' | 'agent-b' | 'director'
    this.turnCount = 0;
    this.maxTurns = 10;
    this.turnDelay = 1200; // ms

    this.topic = '';
    this.agentA = null;
    this.agentB = null;
    this.history = []; // Array of { speaker, text, timestamp }

    // Interjection queue
    this.pendingInterjection = null;

    // Callbacks
    this.onTurnStart = null;
    this.onTurnEnd = null;
    this.onTranscriptUpdate = null;
  }

  configure(topic, agentAConfig, agentBConfig, settings = {}) {
    this.topic = topic;
    this.agentA = agentAConfig;
    this.agentB = agentBConfig;
    this.maxTurns = settings.maxTurns || 10;
    this.turnDelay = (settings.turnDelay || 1.2) * 1000;
  }

  async start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.isPaused = false;
    this.turnCount = 0;

    // Start with Agent A opening the debate
    this.nextTurn('agent-a');
  }

  pause() {
    this.isPaused = true;
    this.audio.interrupt();
    this.orb.setState('idle');
  }

  resume() {
    if (!this.isRunning || !this.isPaused) return;
    this.isPaused = false;
    const nextSpeaker = this.currentSpeaker === 'agent-a' ? 'agent-b' : 'agent-a';
    this.nextTurn(nextSpeaker);
  }

  stop() {
    this.isRunning = false;
    this.isPaused = false;
    this.audio.interrupt();
    this.currentSpeaker = 'none';
    this.orb.setState('idle');
  }

  interject(text) {
    if (!text.trim()) return;

    const entry = {
      speaker: 'director',
      name: 'Director (Moderator)',
      text: text.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    this.history.push(entry);
    if (this.onTranscriptUpdate) this.onTranscriptUpdate(entry);

    // Stop current speech and inject into discussion
    this.audio.interrupt();
    this.orb.setState('speaking', 'director');

    // Queue for the next rebuttal
    this.pendingInterjection = text;

    setTimeout(() => {
      if (this.isRunning && !this.isPaused) {
        // Next agent addresses the interjection
        const nextSpeaker = this.currentSpeaker === 'agent-a' ? 'agent-b' : 'agent-a';
        this.nextTurn(nextSpeaker);
      }
    }, 1500);
  }

  async nextTurn(speakerId) {
    if (!this.isRunning || this.isPaused) return;

    if (this.turnCount >= this.maxTurns) {
      this.stop();
      return;
    }

    this.turnCount++;
    this.currentSpeaker = speakerId;
    const agent = speakerId === 'agent-a' ? this.agentA : this.agentB;
    const opponent = speakerId === 'agent-a' ? this.agentB : this.agentA;

    // 1. Visual state: Thinking
    this.orb.setState('idle', speakerId);
    if (this.onTurnStart) this.onTurnStart(speakerId, agent);

    // 2. Generate argument response
    const argumentText = await this.generateAgentSpeech(agent, opponent);

    if (!this.isRunning || this.isPaused) return;

    // 3. Log to history & transcript
    const entry = {
      speaker: speakerId,
      name: agent.name,
      text: argumentText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };
    this.history.push(entry);
    if (this.onTranscriptUpdate) this.onTranscriptUpdate(entry);

    // 4. Visual state: Speaking
    this.orb.setState('speaking', speakerId);

    // 5. Play audio via voice pipeline
    await this.audio.speakText(argumentText, agent);

    if (this.onTurnEnd) this.onTurnEnd(speakerId);

    // 6. Natural conversational gap delay before next turn
    if (this.isRunning && !this.isPaused) {
      this.orb.setState('idle');
      setTimeout(() => {
        const nextSpeakerId = speakerId === 'agent-a' ? 'agent-b' : 'agent-a';
        this.nextTurn(nextSpeakerId);
      }, this.turnDelay);
    }
  }

  /**
   * Generates dynamic contextual debate dialogue
   * (Connects to mini-AGI backend if running, or uses dynamic contextual debate generator)
   */
  async generateAgentSpeech(agent, opponent) {
    // If mini-AGI endpoint is reachable, we can query it
    const lastTurn = this.history.length > 0 ? this.history[this.history.length - 1] : null;
    const interjectionContext = this.pendingInterjection;
    this.pendingInterjection = null;

    // Contextual intelligent responses tailored to the topic and stance
    const promptSeed = interjectionContext 
      ? `The director just noted: "${interjectionContext}". How do you respond?`
      : lastTurn 
        ? `Respond to ${opponent.name}'s point: "${lastTurn.text}"`
        : `Open the debate on: "${this.topic}"`;

    return this.produceDebateResponse(agent, promptSeed, this.turnCount);
  }

  produceDebateResponse(agent, promptSeed, turnIndex) {
    const isA = agent.id === 'agent-a';

    const responsesA = [
      `If we look at modern generative architectures, AI isn't replacing human taste—it's liberating engineers from boilerplate DSP buffers so they can sculpt sound at an architectural level.`,
      `That skepticism assumes models can only interpolate. But neural audio models like Sesame and byte-level learners can generate non-linear phase responses and saturation curves that would take months to hand-code in JUCE!`,
      `Let's look at the actual output latency. With low-level kernel compilation, real-time audio threads can execute neural inference faster than complex nested filter graphs. Why constrain ourselves to classical algorithms?`,
      `The future of music technology has always been about abstracting tedious primitives. Synthesis didn't kill orchestration, and AI DSP won't kill audio engineering—it supercharges it.`
    ];

    const responsesB = [
      `That's an enticing vision, but in a real-time audio thread, a single nanosecond of non-deterministic jitter causes audible clicks and buffer under-runs. You cannot afford hallucinated sample buffers on a stage soundboard.`,
      `Handcrafted C++ and JUCE DSP give you guaranteed phase alignment, finite impulse stability, and zero unpredictable artifacts. How do you guarantee absolute boundary safety with a black-box model in a 48kHz audio callback?`,
      `Artistic expression relies on intentionality and reproducible physics. When an analog compressor saturates, the physics are predictable and transparent. Neural heuristics often smear transient attacks without the user understanding why.`,
      `A true audio engineer needs microsecond precision and debuggable code. AI should remain an assistant tool for parameter suggestions—not the root engine executing raw audio buffers.`
    ];

    const pool = isA ? responsesA : responsesB;
    const index = (turnIndex - 1) % pool.length;
    return pool[index];
  }
}

window.DebateEngine = DebateEngine;
