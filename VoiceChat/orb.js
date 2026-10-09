/**
 * Sesame Voice Arena - Audio-Reactive Fluid Canvas Orb
 * Inspired by Sesame AI's fluid, organic conversational voice orb.
 */

class VoiceOrb {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.analyser = null;
    this.dataArray = null;

    // State: 'idle' | 'listening' | 'speaking' | 'interrupted'
    this.state = 'idle';
    this.currentSpeaker = 'none'; // 'agent-a' | 'agent-b' | 'director' | 'user'

    // Color Palettes
    this.palettes = {
      idle: {
        primary: [0, 242, 254],     // Cyan
        secondary: [168, 85, 247],  // Violet
        aura: 'rgba(0, 242, 254, 0.15)'
      },
      'agent-a': {
        primary: [0, 242, 254],     // Neon Cyan
        secondary: [79, 172, 254],  // Ocean Blue
        aura: 'rgba(0, 242, 254, 0.35)'
      },
      'agent-b': {
        primary: [168, 85, 247],    // Violet
        secondary: [236, 72, 153],  // Hot Pink
        aura: 'rgba(168, 85, 247, 0.35)'
      },
      user: {
        primary: [255, 75, 114],    // Rose
        secondary: [245, 158, 11],  // Amber
        aura: 'rgba(255, 75, 114, 0.4)'
      },
      director: {
        primary: [245, 158, 11],    // Amber
        secondary: [255, 75, 114],  // Rose
        aura: 'rgba(245, 158, 11, 0.4)'
      }
    };

    // Animation physics
    this.time = 0;
    this.baseRadius = 110;
    this.currentRadius = 110;
    this.audioEnergy = 0;
    this.targetEnergy = 0;

    // Points for organic deformable mesh
    this.numPoints = 120;
    this.points = [];
    for (let i = 0; i < this.numPoints; i++) {
      this.points.push({
        angle: (i / this.numPoints) * Math.PI * 2,
        noiseOffset: Math.random() * 100,
        currentRadius: this.baseRadius
      });
    }

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = (rect.width || 600) * dpr;
    this.canvas.height = (rect.height || 600) * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width || 600;
    this.height = rect.height || 600;
    this.center = { x: this.width / 2, y: this.height / 2 };
  }

  setAudioAnalyser(analyser) {
    this.analyser = analyser;
    if (this.analyser) {
      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    }
  }

  setState(state, speaker = 'none') {
    this.state = state;
    this.currentSpeaker = speaker;
  }

  sampleAudio() {
    if (!this.analyser || !this.dataArray) {
      return 0;
    }
    this.analyser.getByteFrequencyData(this.dataArray);
    
    // Average across speech frequencies (bins 2 to 45)
    let sum = 0;
    const count = Math.min(45, this.dataArray.length);
    for (let i = 2; i < count; i++) {
      sum += this.dataArray[i];
    }
    return (sum / (count - 2)) / 255;
  }

  animate() {
    this.time += 0.025;

    // Fetch and smooth audio energy
    const rawEnergy = this.sampleAudio();
    this.targetEnergy = rawEnergy;
    this.audioEnergy += (this.targetEnergy - this.audioEnergy) * 0.22;

    this.ctx.clearRect(0, 0, this.width, this.height);

    // Pick active color palette
    const activePaletteKey = this.currentSpeaker !== 'none' ? this.currentSpeaker : (this.state === 'listening' ? 'user' : 'idle');
    const palette = this.palettes[activePaletteKey] || this.palettes.idle;

    // Dynamic radius influenced by audio
    const pulseFactor = this.state === 'idle' 
      ? Math.sin(this.time * 1.5) * 6 
      : this.audioEnergy * 45;
    
    this.currentRadius = this.baseRadius + pulseFactor;

    // Draw background glowing aura
    this.drawAura(palette);

    // Draw fluid deformable mesh layers
    this.drawFluidBlob(palette, 1.0, 1.0, 0.85);
    this.drawFluidBlob(palette, 0.88, 1.3, 0.55);
    this.drawCoreGlow(palette);

    requestAnimationFrame(this.animate);
  }

  drawAura(palette) {
    const ctx = this.ctx;
    const auraRadius = this.currentRadius * 1.7;
    const [r, g, b] = palette.primary;

    const grad = ctx.createRadialGradient(
      this.center.x, this.center.y, this.currentRadius * 0.4,
      this.center.x, this.center.y, auraRadius
    );
    grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${0.15 + this.audioEnergy * 0.25})`);
    grad.addColorStop(0.6, `rgba(${r}, ${g}, ${b}, ${0.05 + this.audioEnergy * 0.1})`);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.save();
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(this.center.x, this.center.y, auraRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  drawFluidBlob(palette, scale, speedMultiplier, opacity) {
    const ctx = this.ctx;
    const [r1, g1, b1] = palette.primary;
    const [r2, g2, b2] = palette.secondary;
    const cx = this.center.x;
    const cy = this.center.y;

    ctx.save();
    ctx.beginPath();

    const t = this.time * speedMultiplier;
    const energy = this.audioEnergy;

    // Harmonic multi-frequency deformation
    for (let i = 0; i < this.numPoints; i++) {
      const p = this.points[i];
      const angle = p.angle;

      // Sinusoidal harmonics for liquid wave effect
      const harmonic1 = Math.sin(angle * 3 + t + p.noiseOffset) * (8 + energy * 25);
      const harmonic2 = Math.cos(angle * 5 - t * 1.4) * (5 + energy * 18);
      const harmonic3 = Math.sin(angle * 7 + t * 2.2) * (3 + energy * 12);

      const targetRadius = (this.currentRadius + harmonic1 + harmonic2 + harmonic3) * scale;
      p.currentRadius += (targetRadius - p.currentRadius) * 0.3;

      const x = cx + Math.cos(angle) * p.currentRadius;
      const y = cy + Math.sin(angle) * p.currentRadius;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        // Smooth curve through points
        const prevP = this.points[i - 1];
        const prevX = cx + Math.cos(prevP.angle) * prevP.currentRadius;
        const prevY = cy + Math.sin(prevP.angle) * prevP.currentRadius;
        const midX = (prevX + x) / 2;
        const midY = (prevY + y) / 2;
        ctx.quadraticCurveTo(prevX, prevY, midX, midY);
      }
    }
    ctx.closePath();

    // Vibrant linear / radial gradient
    const grad = ctx.createLinearGradient(
      cx - this.currentRadius, cy - this.currentRadius,
      cx + this.currentRadius, cy + this.currentRadius
    );
    grad.addColorStop(0, `rgba(${r1}, ${g1}, ${b1}, ${opacity})`);
    grad.addColorStop(1, `rgba(${r2}, ${g2}, ${b2}, ${opacity * 0.85})`);

    ctx.fillStyle = grad;
    ctx.shadowColor = `rgba(${r1}, ${g1}, ${b1}, 0.5)`;
    ctx.shadowBlur = 25 * scale;
    ctx.fill();

    // Liquid surface rim highlight
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.25 + energy * 0.5})`;
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.restore();
  }

  drawCoreGlow(palette) {
    const ctx = this.ctx;
    const [r, g, b] = palette.primary;
    const coreRadius = this.currentRadius * 0.45;

    const grad = ctx.createRadialGradient(
      this.center.x, this.center.y, 0,
      this.center.x, this.center.y, coreRadius
    );
    grad.addColorStop(0, `rgba(255, 255, 255, ${0.4 + this.audioEnergy * 0.4})`);
    grad.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, 0.2)`);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.save();
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(this.center.x, this.center.y, coreRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

window.VoiceOrb = VoiceOrb;
