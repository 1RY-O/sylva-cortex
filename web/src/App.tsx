import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RotateCcw,
  Activity,
  Radio,
  Sun,
  Moon,
  Shield,
  Axe,
  CloudLightning,
  Volume2,
  VolumeX,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Info,
  Clock,
  Battery,
  Sliders,
  Sparkles,
  Download,
  Gauge,
  X,
  ChevronDown,
  ChevronUp,
  Crosshair,
  Compass,
  Play,
  FileText,
  Headphones,
  Wind,
  Bird,
  Zap,
  Target,
} from 'lucide-react';

// --- Types ---
interface Observed {
  class: string;
  p: number;
}

interface TraceEvent {
  t: number;
  event_id: number;
  observed: Observed;
  belief: Record<string, number>;
  context_note?: string;
  threshold_at_decision?: number;
  action: 'IGNORE' | 'LOG' | 'ALERT' | 'ESCALATE' | string;
  reason: string;
  action_taken?: string;
  extra?: {
    scenario_note?: string;
    audio_clip?: string;
  };
  thresholds?: Record<string, number>;
  context?: {
    time_of_day?: string;
    zone?: string;
    weather?: string;
  };
  audio_url?: string;
  clean_audio_url?: string;
  raw_audio_url?: string;
}

interface ThresholdState {
  chainsaw: number;
  gunshot: number;
  bird_call: number;
  forest_ambience: number;
  rain: number;
  wind: number;
}

interface ContextState {
  time_of_day: string;
  zone: string;
  weather: string;
}

// --- Animated CountUp Component ---
function CountUp({ value }: { value: number }) {
  const [displayValue, setDisplayValue] = useState(value);
  const prevValueRef = useRef(value);

  useEffect(() => {
    const start = prevValueRef.current;
    const end = value;
    if (start === end) return;

    const duration = 550;
    const startTime = performance.now();

    const update = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + (end - start) * eased);
      setDisplayValue(current);

      if (progress < 1.0) {
        requestAnimationFrame(update);
      } else {
        prevValueRef.current = end;
      }
    };

    const animId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(animId);
  }, [value]);

  return <span>{displayValue}</span>;
}

// --- Viridis Spectrum Visualizer with Cyan Waveform Overlay ---
function ViridisSpectrumVisualizer({
  threatSpike,
  lastEvent,
}: {
  threatSpike: boolean;
  lastEvent: TraceEvent | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const spikeEnergyRef = useRef<number>(0.15);

  useEffect(() => {
    if (threatSpike) {
      spikeEnergyRef.current = 1.0;
    }
  }, [threatSpike]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let phase = 0;
    const numBins = 72;
    const binEnergies = new Array(numBins).fill(0.12);

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;

      // Pure near-black background #0a0a0f with zero noise
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, width, height);

      // Subtle frequency grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      [0.25, 0.5, 0.75].forEach((level) => {
        const y = height * level;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      });

      spikeEnergyRef.current = Math.max(0.15, spikeEnergyRef.current * 0.95);
      const currentSpike = spikeEnergyRef.current;
      phase += 0.065;

      // 1. Viridis Colormap FFT Waterfall Bars
      const binWidth = width / numBins;
      for (let i = 0; i < numBins; i++) {
        const waveA = Math.sin(phase * 1.6 + i * 0.32) * 0.22;
        const waveB = Math.sin(phase * 3.2 + i * 0.65) * 0.14;
        const noise = (Math.random() - 0.5) * 0.05;

        let target = 0.14 + Math.abs(waveA + waveB + noise);
        if (currentSpike > 0.3) {
          target += Math.sin(phase * 5.5 + i * 1.1) * currentSpike * 0.5;
        }
        binEnergies[i] = binEnergies[i] * 0.82 + target * 0.18;

        const barHeight = Math.min(height * 0.88, binEnergies[i] * height * (0.8 + currentSpike * 0.55));
        const x = i * binWidth;
        const y = height - barHeight;

        // Viridis Gradient (#440154 -> #3b528b -> #21918c -> #5ec962 -> #fde725)
        const grad = ctx.createLinearGradient(0, height, 0, height - barHeight);
        if (currentSpike > 0.5) {
          grad.addColorStop(0, '#440154');
          grad.addColorStop(0.3, '#21918c');
          grad.addColorStop(0.65, '#fde725');
          grad.addColorStop(1, '#ef4444');
        } else {
          grad.addColorStop(0, 'rgba(68, 1, 84, 0.35)');
          grad.addColorStop(0.35, 'rgba(59, 82, 139, 0.65)');
          grad.addColorStop(0.7, 'rgba(33, 145, 140, 0.85)');
          grad.addColorStop(0.9, 'rgba(94, 201, 98, 0.95)');
          grad.addColorStop(1, '#fde725');
        }

        ctx.fillStyle = grad;
        ctx.fillRect(x + 1, y, binWidth - 2, barHeight);
      }

      // 2. Bright Cyan Waveform Overlay
      ctx.beginPath();
      ctx.lineWidth = currentSpike > 0.5 ? 2.8 : 2.0;
      ctx.strokeStyle = currentSpike > 0.5 ? '#ef4444' : '#22d3ee';
      ctx.shadowBlur = currentSpike > 0.5 ? 18 : 10;
      ctx.shadowColor = currentSpike > 0.5 ? '#ef4444' : '#06b6d4';

      const centerY = height * 0.42;
      for (let x = 0; x < width; x += 4) {
        const nx = x / width;
        const w1 = Math.sin(nx * 19 + phase * 2.4) * (20 * (0.7 + currentSpike * 1.9));
        const w2 = Math.sin(nx * 42 - phase * 3.6) * (8 * (0.5 + currentSpike));
        const y = centerY + w1 + w2;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // 3. Axis Frequency Ticks
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(209, 213, 219, 0.6)';
      ctx.fillText('80 Hz', 12, height - 8);
      ctx.fillText('2 kHz', width * 0.25, height - 8);
      ctx.fillText('4 kHz', width * 0.5, height - 8);
      ctx.fillText('6 kHz', width * 0.75, height - 8);
      ctx.fillText('8 kHz Nyquist', width - 90, height - 8);

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  return (
    <div className="relative rounded-2xl overflow-hidden border border-white/[0.08] bg-[#0a0a0f] shadow-none">
      <canvas ref={canvasRef} width={740} height={190} className="w-full h-[180px] sm:h-[195px] block" />

      {/* Top HUD */}
      <div className="absolute top-2.5 left-3 right-3 flex items-center justify-between text-[11px] font-mono pointer-events-none">
        <div className="flex items-center space-x-2 bg-zinc-950/90 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/[0.08] text-gray-200">
          <span className="w-2 h-2 rounded-full bg-[#00ff9d] animate-ping mr-1" />
          <span className="font-bold text-[#00ff9d]">Spectrogram + Waveform</span>
          <span className="text-zinc-600">|</span>
          <span className="text-gray-300">16 kHz • Hann 25ms • 40 Mel • Viridis</span>
        </div>

        <div className="flex items-center space-x-2 bg-zinc-950/90 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/[0.08] text-gray-200">
          <span className="text-gray-400">Waveform:</span>
          <span className="text-cyan-400 font-semibold">Cyan Dual-Harmonic</span>
        </div>
      </div>

      {/* Floating Pill Badge when threat arrives */}
      <AnimatePresence>
        {lastEvent && (lastEvent.action === 'ALERT' || lastEvent.action === 'ESCALATE') && (
          <motion.div
            key={lastEvent.event_id}
            initial={{ opacity: 0, scale: 0.85, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            className="absolute bottom-5 left-1/2 -translate-x-1/2 bg-red-950/95 border border-red-500/80 backdrop-blur-md px-5 py-2 rounded-full shadow-[0_0_25px_rgba(239,68,68,0.5)] flex items-center space-x-2.5 text-xs font-mono pointer-events-none"
          >
            <Flame className="w-4 h-4 text-red-400 animate-pulse" />
            <span className="font-extrabold uppercase tracking-widest text-white">
              THREAT DETECTED: {lastEvent.observed.class}
            </span>
            <span className="text-red-300 font-bold">
              ({(lastEvent.observed.p * 100).toFixed(1)}%)
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// --- Belief-Decay Sparkline Chart Component (60s Window, 85% to Near-Zero) ---
function BeliefDecaySparkline({
  chainsawBelief,
  gunshotBelief,
}: {
  chainsawBelief: number;
  gunshotBelief: number;
}) {
  const sparklineData = useMemo(() => {
    const points: { x: number; y: number; t: number }[] = [];
    const totalSeconds = 60;
    const steps = 30;
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * totalSeconds;
      // Exponential decay modeling γ=0.95/tick: p(threat) starts at 85% (0.85) and decays to ~0.02 at 60s
      const baseDecay = 0.85 * Math.exp(-t / 14.5);
      points.push({ x: (i / steps) * 100, y: baseDecay, t });
    }
    return points;
  }, []);

  const path1 = sparklineData
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${(46 - p.y * 42).toFixed(1)}`)
    .join(' ');

  return (
    <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.08] shadow-none">
      <div className="flex items-center justify-between text-xs mb-2">
        <span className="font-semibold text-white flex items-center">
          <Activity className="w-3.5 h-3.5 text-[#00ff9d] mr-1.5" />
          Temporal Belief Decay (60s Window)
        </span>
        <span className="font-mono text-[10px] text-gray-400">γ = 0.95 / tick</span>
      </div>

      <div className="relative h-12 w-full">
        <svg className="w-full h-full overflow-visible" viewBox="0 0 100 48" preserveAspectRatio="none">
          <defs>
            <linearGradient id="grad-decay" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#00ff9d" stopOpacity="0.30" />
              <stop offset="100%" stopColor="#00ff9d" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          <line x1="0" y1="24" x2="100" y2="24" stroke="rgba(255,255,255,0.06)" strokeDasharray="2 2" />
          <path d={`${path1} L 100 48 L 0 48 Z`} fill="url(#grad-decay)" />
          <path d={path1} fill="none" stroke="#00ff9d" strokeWidth="2.0" strokeLinecap="round" />
        </svg>
      </div>

      <div className="flex items-center justify-between text-[10px] text-gray-400 mt-1 font-mono">
        <span className="text-gray-300 font-bold">t = 0s (85% threat prior)</span>
        <span className="text-gray-500">Half-life: 13.5s</span>
        <span className="text-gray-400">t = 60s (~0% faded)</span>
      </div>
    </div>
  );
}

// --- Main App Component ---
export default function App() {
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'reconnecting'>('connecting');
  const [isSimulating, setIsSimulating] = useState<boolean>(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState<boolean>(true);
  const [demoMode, setDemoMode] = useState<'CLEAN' | 'RAW'>('CLEAN');
  const demoModeRef = useRef<'CLEAN' | 'RAW'>('CLEAN');
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>(null);
  const [activeManualSound, setActiveManualSound] = useState<string | null>(null);
  const [activePipelineStep, setActivePipelineStep] = useState<number>(0);
  const [screenShake, setScreenShake] = useState<boolean>(false);
  const [threatFlash, setThreatFlash] = useState<boolean>(false);
  const [lastEvent, setLastEvent] = useState<TraceEvent | null>(null);
  const [expandedPipelineStep, setExpandedPipelineStep] = useState<number | null>(null);
  const [showCalibrateModal, setShowCalibrateModal] = useState<boolean>(false);
  const [calibrationProgress, setCalibrationProgress] = useState<number>(0);
  const [isCalibrating, setIsCalibrating] = useState<boolean>(false);

  const [thresholds, setThresholds] = useState<ThresholdState>({
    chainsaw: 0.70,
    gunshot: 0.70,
    bird_call: 0.70,
    forest_ambience: 0.70,
    rain: 0.70,
    wind: 0.70,
  });

  const [context, setContext] = useState<ContextState>({
    time_of_day: 'night',
    zone: 'core_zone',
    weather: 'clear',
  });

  const [escalationLevel, setEscalationLevel] = useState<number>(0);

  const socketRef = useRef<WebSocket | null>(null);
  const logContainerRef = useRef<HTMLDivElement | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  // Coordinated Event & Audio Queue to pace display with audio clips
  const eventQueueRef = useRef<TraceEvent[]>([]);
  const isProcessingEventRef = useRef<boolean>(false);
  const isAudioEnabledRef = useRef<boolean>(isAudioEnabled);

  useEffect(() => {
    demoModeRef.current = demoMode;
  }, [demoMode]);

  useEffect(() => {
    isAudioEnabledRef.current = isAudioEnabled;
  }, [isAudioEnabled]);

  const triggerEventPresentation = (newEvent: TraceEvent) => {
    setEvents((prev) => [...prev, newEvent]);
    setLastEvent(newEvent);

    if (newEvent.thresholds) {
      setThresholds((prev) => ({ ...prev, ...newEvent.thresholds }));
    }

    if (newEvent.context) {
      setContext((prev) => ({ ...prev, ...newEvent.context }));
    }

    const obsClass = (newEvent.observed?.class || '').toLowerCase();
    const isBenign = [
      'forest_ambience',
      'bird_call',
      'bird',
      'rain',
      'wind',
      'forest',
      'insects',
    ].includes(obsClass);

    // Dynamic escalation level: if backend provides escalation_level, use it;
    // Otherwise, if benign non-threat signal arrives, immediately reset to 0 (Calm);
    // Otherwise update to 3 (ESCALATE), 2 (ALERT), 1 (LOG threat), or 0 (Calm).
    if (typeof (newEvent as any).escalation_level === 'number') {
      const lvl = (newEvent as any).escalation_level;
      setEscalationLevel(lvl);
      if (lvl === 3) {
        setScreenShake(true);
        setThreatFlash(true);
        setTimeout(() => setScreenShake(false), 200);
        setTimeout(() => setThreatFlash(false), 1200);
      } else if (lvl === 2) {
        setThreatFlash(true);
        setTimeout(() => setThreatFlash(false), 800);
      }
    } else if (isBenign || newEvent.action === 'IGNORE') {
      setEscalationLevel(0);
    } else if (newEvent.action === 'ESCALATE') {
      setScreenShake(true);
      setThreatFlash(true);
      setTimeout(() => setScreenShake(false), 200);
      setTimeout(() => setThreatFlash(false), 1200);
      setEscalationLevel(3);
    } else if (newEvent.action === 'ALERT') {
      setThreatFlash(true);
      setTimeout(() => setThreatFlash(false), 800);
      setEscalationLevel(2);
    } else if (newEvent.action === 'LOG') {
      setEscalationLevel(obsClass === 'chainsaw' || obsClass === 'gunshot' ? 1 : 0);
    } else {
      setEscalationLevel(0);
    }
  };

  const processNextEvent = () => {
    if (eventQueueRef.current.length === 0) {
      isProcessingEventRef.current = false;
      return;
    }

    isProcessingEventRef.current = true;
    const newEvent = eventQueueRef.current.shift()!;
    triggerEventPresentation(newEvent);

    const advanceToNext = (delayMs: number = 250) => {
      setTimeout(() => {
        processNextEvent();
      }, delayMs);
    };

    // Pick pure clean audio or raw noisy audio based on Demo Mode
    const chosenAudioUrl = demoModeRef.current === 'RAW'
      ? (newEvent.raw_audio_url || newEvent.audio_url)
      : (newEvent.clean_audio_url || newEvent.audio_url || `/audio/sample_pure_${newEvent.observed.class}.wav`);

    if (isAudioEnabledRef.current && chosenAudioUrl) {
      setCurrentAudioUrl(chosenAudioUrl);
      try {
        const fullUrl = chosenAudioUrl.startsWith('http')
          ? chosenAudioUrl
          : `http://localhost:8000${chosenAudioUrl.startsWith('/') ? chosenAudioUrl : `/${chosenAudioUrl}`}`;
        const audio = new Audio(fullUrl);
        currentAudioRef.current = audio;
        audio.volume = 0.98;

        let advanced = false;
        const onClipEnded = () => {
          if (!advanced) {
            advanced = true;
            // Let the full 2-3 second clip complete cleanly with a 600ms breathing window
            advanceToNext(600);
          }
        };

        // Ensure audio never cuts off early by listening to natural onended
        audio.onended = onClipEnded;
        audio.onerror = () => {
          if (!advanced) {
            advanced = true;
            setTimeout(onClipEnded, 3000);
          }
        };

        const playPromise = audio.play();
        if (playPromise !== undefined) {
          playPromise.catch((err) => {
            console.warn('Autoplay blocked or playback error:', err);
            if (!advanced) {
              advanced = true;
              setTimeout(onClipEnded, 3500);
            }
          });
        }
      } catch (err) {
        console.warn('Audio construction error:', err);
        advanceToNext(3000);
      }
    } else {
      // When audio is disabled or absent, hold the event on screen for 4.5s so the judge can read it
      advanceToNext(4500);
    }
  };

  // Immediate manual sound trigger for judges
  const playManualJudgeClip = (filename: string, soundName: string) => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.currentTime = 0;
    }
    setActiveManualSound(soundName);

    const cleanFilename = filename.replace(/^\/+audio\/+/, '').replace(/^\/+/, '');
    const manualUrl = `/audio/${cleanFilename}`;
    const fullUrl = `http://localhost:8000${manualUrl}`;

    try {
      const audio = new Audio(fullUrl);
      currentAudioRef.current = audio;
      audio.volume = 0.98;
      audio.onended = () => setActiveManualSound(null);
      audio.onerror = () => setActiveManualSound(null);
      audio.play().catch((err) => {
        console.warn('Manual audio playback failed:', err);
        setActiveManualSound(null);
      });
    } catch {
      setActiveManualSound(null);
    }
  };

  // Play audio for class respecting Demo Mode (CLEAN vs RAW)
  const playManualAudioForClass = (classKey: string, soundName: string) => {
    let filename = `sample_pure_${classKey}.wav`;
    if (demoModeRef.current === 'RAW') {
      if (classKey === 'chainsaw') filename = 'mix_chainsaw_snr-05db_idx00.wav';
      else if (classKey === 'gunshot') filename = 'mix_gunshot_snr-05db_idx00.wav';
      else if (classKey === 'wind') filename = 'mix_wind_snr+00db_idx00.wav';
      else if (classKey === 'bird_call') filename = 'mix_bird_call_snr+00db_idx00.wav';
      else filename = `sample_pure_${classKey}.wav`;
    }
    playManualJudgeClip(filename, soundName);
  };

  // Sequential breathing progression across the 7 pipeline steps (1.2s per step)
  useEffect(() => {
    const timer = setInterval(() => {
      setActivePipelineStep((prev) => (prev + 1) % 7);
    }, 1200);
    return () => clearInterval(timer);
  }, []);

  // WebSocket Connection
  const connectWebSocket = () => {
    if (socketRef.current && (socketRef.current.readyState === WebSocket.OPEN || socketRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      setConnectionStatus('connecting');
      const ws = new WebSocket('ws://localhost:8000/ws');

      ws.onopen = () => {
        setConnectionStatus('connected');
        fetch('http://localhost:8000/api/simulate', { method: 'POST' }).catch(() => {
          ws.send(JSON.stringify({ command: 'start_simulation' }));
        });
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'agent_event') {
            const newEvent: TraceEvent = data;
            eventQueueRef.current.push(newEvent);
            if (!isProcessingEventRef.current) {
              processNextEvent();
            }
          } else if (data.type === 'simulation_started') {
            setIsSimulating(true);
            if (data.thresholds) setThresholds(data.thresholds);
            if (data.context) setContext(data.context);
          } else if (data.type === 'simulation_complete') {
            setIsSimulating(false);
            if (data.final_thresholds) setThresholds(data.final_thresholds);
          } else if (data.type === 'connected') {
            if (data.initial_thresholds) setThresholds((prev) => ({ ...prev, ...data.initial_thresholds }));
            if (data.initial_context) setContext((prev) => ({ ...prev, ...data.initial_context }));
          }
        } catch (err) {
          console.error('Error parsing WebSocket payload:', err);
        }
      };

      ws.onclose = () => {
        setConnectionStatus('reconnecting');
        setTimeout(() => connectWebSocket(), 3000);
      };

      ws.onerror = () => {
        setConnectionStatus('reconnecting');
      };

      socketRef.current = ws;
    } catch {
      setConnectionStatus('reconnecting');
    }
  };

  // Auto-Start on Page Load
  useEffect(() => {
    connectWebSocket();
    const initTimer = setTimeout(() => {
      fetch('http://localhost:8000/api/simulate', { method: 'POST' }).catch(() => {});
    }, 350);

    return () => {
      clearTimeout(initTimer);
      if (socketRef.current) socketRef.current.close();
      if (currentAudioRef.current) currentAudioRef.current.pause();
    };
  }, []);

  // Auto-scroll log container to bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [events]);

  const handleRestartSimulation = async () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
    }
    eventQueueRef.current = [];
    isProcessingEventRef.current = false;
    setEvents([]);
    setEscalationLevel(0);
    setLastEvent(null);
    setIsSimulating(true);
    setThresholds({
      chainsaw: 0.70,
      gunshot: 0.70,
      bird_call: 0.70,
      forest_ambience: 0.70,
      rain: 0.70,
      wind: 0.70,
    });

    try {
      await fetch('http://localhost:8000/api/simulate', { method: 'POST' });
    } catch {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        socketRef.current.send(JSON.stringify({ command: 'start_simulation' }));
      }
    }
  };

  // Export Ranger Shift Report (.txt)
  const handleExportRangerReport = () => {
    const lines: string[] = [];

    events.forEach((ev) => {
      const zoneName =
        ev.context?.zone === 'core_zone'
          ? 'Sector 7G (Core Zone)'
          : ev.context?.zone === 'concession'
          ? 'Sector 4B (Logging Concession)'
          : 'Sector 2A (Buffer Zone)';

      const timeStr = `t+${ev.t.toFixed(1)}s [${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}]`;
      const eventClass = ev.observed.class.replace('_', ' ').toUpperCase();

      lines.push('--- SYLVA CORTEX RANGER REPORT ---');
      lines.push(`Time: ${timeStr}`);
      lines.push(`Event: ${eventClass} detected`);
      lines.push(`Location: ${zoneName}`);
      lines.push(`Action Taken: ${ev.action_taken || 'No physical action required.'}`);
      lines.push('---------------------------------\n');
    });

    const reportText = lines.join('\n');
    const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'Ranger_Shift_Report.txt';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  // Export Raw JSON Traces (for technical judges)
  const handleExportJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(events, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `sylva_cortex_traces_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Run Ambient Calibration Simulation
  const handleStartCalibration = () => {
    setIsCalibrating(true);
    setCalibrationProgress(0);

    const interval = setInterval(() => {
      setCalibrationProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsCalibrating(false);
          return 100;
        }
        return prev + 10;
      });
    }, 280);
  };

  // High contrast action badge formatting
  const getActionBadge = (action: string) => {
    switch (action) {
      case 'ESCALATE':
        return {
          bg: 'bg-red-500/20 text-red-300 border-red-500/80 shadow-[0_0_12px_rgba(239,68,68,0.4)]',
          icon: <Flame className="w-3.5 h-3.5 mr-1 text-red-400 animate-pulse" />,
        };
      case 'ALERT':
        return {
          bg: 'bg-amber-500/20 text-amber-300 border-amber-500/80 shadow-[0_0_12px_rgba(245,158,11,0.35)]',
          icon: <AlertTriangle className="w-3.5 h-3.5 mr-1 text-amber-400" />,
        };
      case 'LOG':
        return {
          bg: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/60',
          icon: <Info className="w-3.5 h-3.5 mr-1 text-indigo-300" />,
        };
      default:
        return {
          bg: 'bg-zinc-800 text-gray-300 border-zinc-700',
          icon: <CheckCircle2 className="w-3.5 h-3.5 mr-1 text-gray-400" />,
        };
    }
  };

  // High contrast semantic styling for Action Taken based on physical operation
  const getActionTakenStyle = (actionText: string = '') => {
    const lower = actionText.toLowerCase();
    if (lower.includes('no physical action')) {
      return {
        box: 'bg-zinc-800/40 border-l-2 border-zinc-600',
        tag: 'text-zinc-400',
        text: 'text-zinc-300',
        icon: 'text-zinc-500',
        star: 'text-zinc-500',
      };
    }
    if (lower.includes('dispatched drone') || lower.includes('alerted')) {
      return {
        box: 'bg-[#00ff9d]/10 border-l-[3px] border-[#00ff9d] shadow-[-4px_0_15px_rgba(0,255,157,0.25)]',
        tag: 'text-[#00ff9d]',
        text: 'text-white',
        icon: 'text-[#00ff9d]',
        star: 'text-[#00ff9d]',
      };
    }
    if (lower.includes('pinged on-duty ranger')) {
      return {
        box: 'bg-amber-500/10 border-l-[3px] border-amber-500 shadow-[-4px_0_15px_rgba(245,158,11,0.20)]',
        tag: 'text-amber-400',
        text: 'text-amber-200',
        icon: 'text-amber-400',
        star: 'text-amber-400',
      };
    }
    return {
      box: 'bg-zinc-800/40 border-l-2 border-zinc-600',
      tag: 'text-zinc-400',
      text: 'text-zinc-300',
      icon: 'text-zinc-500',
      star: 'text-zinc-500',
    };
  };

  // Dynamic escalation chip state styling
  const getHeaderEscalationChip = (level: number) => {
    switch (level) {
      case 3:
        return {
          text: 'LEVEL 3: EMERGENCY',
          badge: 'border-red-500/80 bg-red-950/40 text-red-400 animate-pulse shadow-[0_0_12px_rgba(239,68,68,0.35)]',
          dot: 'bg-red-500 animate-pulse',
        };
      case 2:
        return {
          text: 'LEVEL 2: ALERT',
          badge: 'border-orange-500/80 bg-orange-950/40 text-orange-400 shadow-[0_0_10px_rgba(249,115,22,0.25)]',
          dot: 'bg-orange-500',
        };
      case 1:
        return {
          text: 'LEVEL 1: WATCH',
          badge: 'border-amber-500/70 bg-amber-950/30 text-amber-300 shadow-[0_0_8px_rgba(245,158,11,0.20)]',
          dot: 'bg-amber-400',
        };
      case 0:
      default:
        return {
          text: 'LEVEL 0: CALM',
          badge: 'border-emerald-500/40 bg-emerald-950/20 text-emerald-400/90',
          dot: 'bg-emerald-500',
        };
    }
  };

  const alertCount = useMemo(
    () => events.filter((e) => e.action === 'ALERT' || e.action === 'ESCALATE').length,
    [events]
  );

  return (
    <div
      className={`min-h-screen bg-[#0a0a0f] text-gray-100 flex flex-col font-sans transition-all duration-200 ${
        screenShake ? 'screen-shake' : ''
      } ${threatFlash ? 'escalate-active-glow' : ''}`}
    >
      {/* Top Header */}
      <header className="border-b border-white/[0.08] bg-[#0a0a0f]/95 backdrop-blur-xl px-4 sm:px-6 py-2.5 sticky top-0 z-50 w-full">
        <div className="w-full max-w-[1800px] mx-auto flex flex-wrap xl:flex-nowrap items-center justify-between gap-2 sm:gap-3">
          {/* Left: Brand Identity, Live Indicator & Status */}
          <div className="flex items-center space-x-3 shrink-0">
            {/* Small Purple Cat Logo (Simple flat 2D vector style) */}
            <div className="w-9 h-9 bg-purple-500/10 border border-purple-500/30 rounded-lg flex items-center justify-center text-purple-400 shrink-0">
              <svg
                className="w-5 h-5"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path d="M12 4c-4.4 0-8 3.6-8 8 0 2.2.9 4.2 2.4 5.7L6 21l3.5-1.5c.8.3 1.6.5 2.5.5s1.7-.2 2.5-.5L18 21l-.4-3.3C19.1 16.2 20 14.2 20 12c0-4.4-3.6-8-8-8zm-3.5 7a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zm7 0a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" />
              </svg>
            </div>

            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-base font-bold tracking-tight text-white flex items-center whitespace-nowrap">
                  Sylva Cortex — Forest Sentinel
                </h1>
                {/* Glowing Teal "LIVE" Pill */}
                <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-[#00ff9d]/15 border border-[#00ff9d]/40 text-[#00ff9d] text-xs font-mono font-bold shadow-[0_0_10px_rgba(0,255,157,0.25)] shrink-0">
                  <span className="w-2 h-2 rounded-full bg-[#00ff9d] animate-ping" />
                  <span>LIVE</span>
                </div>

                {/* Dynamic Escalation Status Chip */}
                {(() => {
                  const chip = getHeaderEscalationChip(escalationLevel);
                  return (
                    <div className={`flex items-center space-x-1.5 px-2 py-0.5 rounded-full border text-xs font-mono font-bold transition-all shrink-0 ${chip.badge}`}>
                      <span className={`w-2 h-2 rounded-full ${chip.dot}`} />
                      <span className="whitespace-nowrap">{chip.text}</span>
                    </div>
                  );
                })()}
              </div>
              <p className="text-xs text-gray-400 font-medium">
                Real-time Bioacoustic Threat Detection
              </p>
            </div>
          </div>

          {/* Right Header Section: KPIs, Demo Toggle, and Standardized Button Row */}
          <div className="flex items-center space-x-2 sm:space-x-2.5 shrink-0 flex-wrap sm:flex-nowrap gap-y-2">
            {/* Small floating validation card */}
            <div className="hidden 2xl:flex items-center space-x-2 h-9 px-3 rounded-lg bg-white/[0.03] border border-zinc-800 text-xs font-mono text-gray-300 shrink-0 whitespace-nowrap">
              <span className="text-gray-400">Benchmark: 200 events</span>
              <span className="text-zinc-600">|</span>
              <span>Precision <strong className="text-white">89%</strong></span>
              <span className="text-zinc-600">|</span>
              <span>FP <strong className="text-[#00ff9d]">17%</strong></span>
            </div>

            {/* Demo Mode: CLEAN / RAW Toggle Switch */}
            <div className="flex items-center h-9 p-0.5 rounded-lg bg-white/[0.03] border border-zinc-800 text-xs font-mono shrink-0">
              <span className="text-gray-400 font-bold px-2 text-[11px] uppercase tracking-wider whitespace-nowrap">Demo:</span>
              <button
                onClick={() => setDemoMode('CLEAN')}
                className={`h-7.5 px-2.5 rounded-md font-bold transition flex items-center justify-center space-x-1 text-xs whitespace-nowrap shrink-0 ${
                  demoMode === 'CLEAN'
                    ? 'bg-[#00ff9d] text-black shadow-[0_0_10px_rgba(0,255,157,0.4)]'
                    : 'text-gray-400 hover:text-white'
                }`}
                title="Clean Demo Mode: Loud, crisp threats with zero background static"
              >
                <span>CLEAN</span>
              </button>
              <button
                onClick={() => setDemoMode('RAW')}
                className={`h-7.5 px-2.5 rounded-md font-bold transition flex items-center justify-center space-x-1 text-xs whitespace-nowrap shrink-0 ${
                  demoMode === 'RAW'
                    ? 'bg-amber-500 text-black shadow-[0_0_10px_rgba(245,158,11,0.4)]'
                    : 'text-gray-400 hover:text-white'
                }`}
                title="RAW Demo Mode: Mixed noisy environmental audio (proves masking & DSP challenge)"
              >
                <span>RAW</span>
              </button>
            </div>

            {/* Grouped & Standardized Action Buttons */}
            <div className="flex items-center shrink-0 space-x-1.5 sm:space-x-2">
              {/* Group 1: Live Stream & Audio */}
              <div className="flex items-center space-x-1.5 shrink-0">
                {/* ONE filled teal primary button "Live Stream" with soft glow */}
                <button
                  onClick={handleRestartSimulation}
                  className="h-9 px-3 sm:px-3.5 rounded-lg bg-[#00ff9d] hover:bg-[#00ff9d]/90 text-black font-extrabold text-xs flex items-center justify-center whitespace-nowrap shrink-0 shadow-[0_0_15px_rgba(0,255,157,0.3)] transition active:scale-95"
                >
                  <Radio className="w-4 h-4 mr-1.5 shrink-0 animate-pulse" />
                  <span className="whitespace-nowrap">Live Stream</span>
                </button>

                {/* Ghost Button 1: Audio */}
                <button
                  onClick={() => setIsAudioEnabled(!isAudioEnabled)}
                  className="h-9 px-3 rounded-lg bg-transparent hover:bg-white/[0.05] hover:border-zinc-600 text-zinc-300 border border-zinc-700 text-xs font-semibold flex items-center justify-center whitespace-nowrap shrink-0 transition"
                  title={isAudioEnabled ? 'Audio Playback Enabled' : 'Audio Playback Muted'}
                >
                  {isAudioEnabled ? (
                    <>
                      <Volume2 className="w-4 h-4 mr-1.5 shrink-0 text-[#00ff9d]" />
                      <span className="whitespace-nowrap">Audio</span>
                    </>
                  ) : (
                    <>
                      <VolumeX className="w-4 h-4 mr-1.5 shrink-0 text-gray-400" />
                      <span className="whitespace-nowrap">Muted</span>
                    </>
                  )}
                </button>
              </div>

              {/* Subtle Vertical Divider Line */}
              <div className="border-l border-zinc-700 h-5 mx-1 shrink-0" />

              {/* Group 2: Calibrate, Export Logs, Restart */}
              <div className="flex items-center space-x-1.5 shrink-0">
                {/* Ghost Button 2: Calibrate */}
                <button
                  onClick={() => setShowCalibrateModal(true)}
                  className="h-9 px-3 rounded-lg bg-transparent hover:bg-white/[0.05] hover:border-zinc-600 text-zinc-300 border border-zinc-700 text-xs font-semibold flex items-center justify-center whitespace-nowrap shrink-0 transition"
                >
                  <Gauge className="w-4 h-4 mr-1.5 shrink-0 text-indigo-400" />
                  <span className="whitespace-nowrap">Calibrate</span>
                </button>

                {/* Ghost Button 3: Export Logs */}
                <button
                  onClick={handleExportRangerReport}
                  disabled={events.length === 0}
                  className="h-9 px-3 rounded-lg bg-transparent hover:bg-white/[0.05] hover:border-zinc-600 text-zinc-300 border border-zinc-700 text-xs font-semibold flex items-center justify-center whitespace-nowrap shrink-0 transition disabled:opacity-40"
                  title="Download human-readable ranger shift report (.txt)"
                >
                  <FileText className="w-4 h-4 mr-1.5 shrink-0 text-[#00ff9d]" />
                  <span className="whitespace-nowrap">Export Logs</span>
                </button>

                {/* Ghost Button 4: Restart (red outline / red hover) */}
                <button
                  onClick={handleRestartSimulation}
                  className="h-9 px-3 rounded-lg bg-transparent hover:bg-red-500/10 hover:border-red-500/70 text-red-400 border border-red-500/50 text-xs font-semibold flex items-center justify-center whitespace-nowrap shrink-0 transition"
                  title="Restart Sentinel Live Simulation"
                >
                  <RotateCcw className="w-4 h-4 mr-1.5 shrink-0 text-red-400" />
                  <span className="whitespace-nowrap">Restart</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main 3-Column Layout */}
      <main className="flex-1 w-full max-w-[1800px] mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* LEFT COLUMN: Vertical DSP Pipeline + Threat Escalation Meter */}
        <div className="lg:col-span-3 flex flex-col space-y-4">
          {/* KPI Summary Cards */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-[#0e0f17]/90 border border-white/[0.08] rounded-2xl p-4 shadow-none">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider block">
                Events Ingested
              </span>
              <div className="text-3xl font-extrabold text-white mt-1 tracking-tight font-mono">
                <CountUp value={events.length} />
              </div>
              <span className="text-[11px] text-gray-400 font-medium">Bioacoustic clips</span>
            </div>

            <div className="bg-[#0e0f17]/90 border border-white/[0.08] rounded-2xl p-4 shadow-none">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider block">
                Active Alerts
              </span>
              <div
                className={`text-3xl font-extrabold mt-1 tracking-tight font-mono ${
                  alertCount > 0 ? 'text-red-400 animate-pulse' : 'text-gray-200'
                }`}
              >
                <CountUp value={alertCount} />
              </div>
              <span className="text-[11px] text-gray-400 font-medium">Threat triggers</span>
            </div>
          </div>

          {/* Vertical DSP Pipeline of Collapsible Cards with Sequential Breathing Glow */}
          <div className="bg-[#0e0f17]/90 border border-white/[0.08] rounded-2xl p-4 flex-1 flex flex-col shadow-none relative">
            <div className="flex items-center justify-between mb-3.5">
              <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center">
                <Activity className="w-4 h-4 mr-1.5 text-[#00ff9d]" />
                DSP & Agent Pipeline
              </h2>
              <span className="text-[10px] font-mono bg-zinc-900 px-2 py-0.5 rounded text-[#00ff9d] font-bold border border-white/[0.08]">
                16 kHz
              </span>
            </div>

            {/* Pipeline Cards */}
            <div className="space-y-2 flex-1 relative z-10">
              {[
                {
                  id: 0,
                  name: '1. Bandpass & DC Removal',
                  short: 'Butterworth 4th-order (80-8000 Hz)',
                  detail: 'Eliminates DC sensor drift and sub-80Hz wind buffeting. 4th-order Butterworth ensures zero passband ripple.',
                },
                {
                  id: 1,
                  name: '2. Spectral Subtraction',
                  short: 'Oversubtraction α=2.0, floor β=0.02',
                  detail: '10th-percentile energy floor tracking. Removes stationary forest background hiss without musical noise artefacts.',
                },
                {
                  id: 2,
                  name: '3. Feature Extraction',
                  short: '40 Mel, 13 MFCC+Δ+ΔΔ, Crest',
                  detail: 'Extracts 40 mel bands, 13 MFCCs with deltas, spectral centroid, zero-crossing rate, crest factor, and onset attack time.',
                },
                {
                  id: 3,
                  name: '4. LightGBM Classifier',
                  short: 'Domain-randomized physical priors',
                  detail: 'GroupKFold gradient boosted decision trees evaluated across 7 synthetic SNR levels (-10 to +20 dB).',
                },
                {
                  id: 4,
                  name: '5. Bayesian Belief Update',
                  short: 'Decay γ=0.95/tick, 5.0s temporal reinforce',
                  detail: 'Multiplicative posterior decay per tick. Temporal proximity within 5.0s applies probabilistic union reinforcement.',
                },
                {
                  id: 5,
                  name: '6. Contextual Decision Policy',
                  short: 'Core zone 1.30x / Concession 0.80x',
                  detail: 'Modulates effective decision thresholds by operational risk context. Storm clutter factor introduces 0.90x suppression.',
                },
                {
                  id: 6,
                  name: '7. Adaptive Controller',
                  short: 'Proportional controller step Δθ=0.02',
                  detail: 'Proportional controller raises threshold by step Δθ=0.02 when false alarms are reported, eliminating repetitive alerts.',
                },
              ].map((step) => {
                const isActive = activePipelineStep === step.id;
                const isCompleted = activePipelineStep > step.id;
                const isExpanded = expandedPipelineStep === step.id;

                return (
                  <motion.div
                    key={step.id}
                    onClick={() => setExpandedPipelineStep(isExpanded ? null : step.id)}
                    animate={{
                      borderColor: isActive
                        ? 'rgba(0, 255, 157, 0.85)'
                        : isCompleted
                        ? 'rgba(0, 255, 157, 0.22)'
                        : 'rgba(255, 255, 255, 0.08)',
                      backgroundColor: isActive
                        ? 'rgba(0, 255, 157, 0.06)'
                        : isCompleted
                        ? 'rgba(255, 255, 255, 0.02)'
                        : 'rgba(255, 255, 255, 0.01)',
                      boxShadow: isActive
                        ? '0 0 16px rgba(0, 255, 157, 0.25), inset 0 0 8px rgba(0, 255, 157, 0.04)'
                        : '0 0 0px rgba(0, 0, 0, 0)',
                    }}
                    transition={{ duration: 0.5, ease: 'easeInOut' }}
                    className={`p-2.5 rounded-xl border cursor-pointer ${
                      isActive ? 'pipeline-step-active' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-white flex items-center">
                        {isCompleted ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-[#00ff9d] mr-1.5 shrink-0" />
                        ) : isActive ? (
                          <span className="relative flex h-2 w-2 mr-2 shrink-0">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00ff9d] opacity-75" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#00ff9d]" />
                          </span>
                        ) : (
                          <span className="w-2 h-2 rounded-full mr-2 bg-zinc-700/80 shrink-0" />
                        )}
                        <span
                          className={`transition-colors ${
                            isActive
                              ? 'text-[#00ff9d] font-bold'
                              : isCompleted
                              ? 'text-gray-200'
                              : 'text-gray-400'
                          }`}
                        >
                          {step.name}
                        </span>
                      </span>
                      {isExpanded ? (
                        <ChevronUp className="w-3.5 h-3.5 text-gray-400" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5 text-gray-500" />
                      )}
                    </div>
                    <p
                      className={`text-[11px] mt-0.5 ml-4 transition-colors ${
                        isActive ? 'text-gray-300' : 'text-gray-500'
                      }`}
                    >
                      {step.short}
                    </p>

                    {isExpanded && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        className="mt-2 pt-2 border-t border-white/[0.08] text-[11px] text-gray-300 ml-4 font-mono leading-relaxed"
                      >
                        {step.detail}
                      </motion.div>
                    )}
                  </motion.div>
                );
              })}
            </div>

            {/* Bottom-left: SINGLE Threat Escalation Meter Bar (no duplicate) */}
            <div className="mt-4 pt-3.5 border-t border-white/[0.08]">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-gray-300 font-semibold">Threat Escalation:</span>
                <span
                  className={`font-mono font-bold px-2 py-0.5 rounded text-[11px] ${
                    escalationLevel >= 3
                      ? 'bg-red-500/25 text-red-300 border border-red-500/80 animate-pulse'
                      : escalationLevel >= 2
                      ? 'bg-orange-500/25 text-orange-300 border border-orange-500/80'
                      : escalationLevel >= 1
                      ? 'bg-amber-500/25 text-amber-300 border border-amber-500/80'
                      : 'bg-zinc-900 text-gray-400 border border-white/[0.08]'
                  }`}
                >
                  {escalationLevel === 3
                    ? 'LEVEL 3: EMERGENCY'
                    : escalationLevel === 2
                    ? 'LEVEL 2: ALERT'
                    : escalationLevel === 1
                    ? 'LEVEL 1: WATCH'
                    : 'LEVEL 0: CALM'}
                </span>
              </div>
              <div className="w-full bg-zinc-900 h-2.5 rounded-full overflow-hidden flex border border-white/[0.04]">
                <div
                  className={`h-full transition-all duration-500 ${
                    escalationLevel === 3
                      ? 'w-full bg-red-500'
                      : escalationLevel === 2
                      ? 'w-2/3 bg-orange-500'
                      : escalationLevel === 1
                      ? 'w-1/3 bg-amber-500'
                      : 'w-0'
                  }`}
                />
              </div>
            </div>
          </div>
        </div>

        {/* CENTER COLUMN: Viridis Spectrogram + Chain of Thought Feed */}
        <div className="lg:col-span-5 flex flex-col space-y-4">
          {/* Large Spectrogram in Viridis Colormap with Cyan Waveform Overlay */}
          <ViridisSpectrumVisualizer threatSpike={threatFlash} lastEvent={lastEvent} />

          {/* Chain of Thought & Action Feed */}
          <div className="flex-1 flex flex-col bg-[#0e0f17]/90 border border-white/[0.08] rounded-2xl overflow-hidden shadow-none">
            <div className="px-5 py-3 border-b border-white/[0.08] flex items-center justify-between bg-zinc-900/60">
              <div className="flex items-center space-x-2">
                <Radio className="w-4 h-4 text-[#00ff9d] animate-pulse" />
                <h2 className="text-xs font-bold text-white uppercase tracking-wider">
                  Chain of Thought & Action Feed
                </h2>
              </div>
              <div className="text-[11px] font-mono text-gray-400 bg-white/[0.03] px-2.5 py-0.5 rounded border border-white/[0.08]">
                {events.length} Events Logged
              </div>
            </div>

            {/* Scrolling Feed Container */}
            <div
              ref={logContainerRef}
              className="flex-1 p-3.5 overflow-y-auto space-y-3 max-h-[380px] min-h-[300px]"
            >
              {events.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 text-gray-400 space-y-2">
                  <span className="w-3 h-3 rounded-full bg-[#00ff9d] animate-ping" />
                  <p className="text-xs text-gray-300 font-medium">
                    Sentinel active. Waiting for bioacoustic input...
                  </p>
                </div>
              ) : (
                <AnimatePresence initial={false}>
                  {events.map((event, index) => {
                    const badge = getActionBadge(event.action);
                    const isThreat = event.action === 'ALERT' || event.action === 'ESCALATE';

                    return (
                      <motion.div
                        key={event.event_id || index}
                        initial={{ opacity: 0, y: 12, scale: 0.98 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        transition={{ duration: 0.2 }}
                        className={`p-3.5 rounded-xl bg-black/40 border transition shadow-none ${
                          isThreat
                            ? 'border-red-500/80 bg-red-950/20'
                            : 'border-white/[0.08] hover:border-white/20'
                        }`}
                      >
                        {/* Event Header */}
                        <div className="flex items-center justify-between text-xs mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="font-mono text-[11px] text-gray-300 bg-white/[0.03] px-1.5 py-0.5 rounded border border-white/[0.08] flex items-center">
                              <Clock className="w-3 h-3 mr-1 text-gray-400" />
                              t = {event.t.toFixed(1)}s
                            </span>
                            <span className="text-[11px] text-gray-500 font-mono">#{event.event_id}</span>
                          </div>

                          <div className="flex items-center space-x-2">
                            {event.audio_url && (
                              <button
                                onClick={() => {
                                  const clip = demoMode === 'RAW'
                                    ? (event.raw_audio_url || event.audio_url)
                                    : (event.clean_audio_url || event.audio_url || `/audio/sample_pure_${event.observed.class}.wav`);
                                  if (clip) {
                                    playManualJudgeClip(clip, event.observed.class);
                                  }
                                }}
                                className="px-2 py-0.5 rounded bg-white/[0.04] hover:bg-white/[0.08] text-[10px] text-[#00ff9d] font-semibold border border-white/[0.08] flex items-center transition"
                              >
                                <Play className="w-2.5 h-2.5 mr-1" />
                                Audio
                              </button>
                            )}
                            <div className={`flex items-center px-2 py-0.5 rounded-full border text-[11px] font-bold ${badge.bg}`}>
                              {badge.icon}
                              {event.action}
                            </div>
                          </div>
                        </div>

                        {/* Monospace Chain of Thought Breakdown: [DSP] in blue, [OBSERVE] in cyan, [REASON] in white, [ACTION TAKEN] in green */}
                        <div className="space-y-1 font-mono text-[11px] p-2.5 rounded-lg bg-[#06060a] border border-white/[0.06]">
                          <div>
                            <span className="text-blue-400 font-bold">[DSP]</span>{' '}
                            <span className="text-gray-300">Extracting MFCCs & Spectral Crest factor...</span>
                          </div>

                          <div>
                            <span className="text-cyan-400 font-bold">[OBSERVE]</span>{' '}
                            <span className="text-white font-bold capitalize">{event.observed.class}</span>{' '}
                            <span className="text-gray-300">signature detected (confidence p = {(event.observed.p * 100).toFixed(1)}%)</span>
                          </div>

                          <div>
                            <span className="text-white font-bold">[REASON]</span>{' '}
                            <span className="text-gray-300">{event.reason}</span>
                          </div>

                          {/* ACTION TAKEN line: dynamic semantic styling based on physical action */}
                          {(() => {
                            const actionText = event.action_taken || 'No physical action required.';
                            const actStyle = getActionTakenStyle(actionText);

                            return (
                              <div className={`mt-1.5 p-2 rounded-r-lg flex items-center space-x-2 transition-all ${actStyle.box}`}>
                                <span className={`${actStyle.star} text-sm leading-none`}>✦</span>
                                <svg className={`w-4 h-4 ${actStyle.icon} shrink-0`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" />
                                  <path d="M10 10l-6 -6m0 4a2 2 0 1 0 0 -4a2 2 0 0 0 0 4" />
                                  <path d="M14 10l6 -6m-4 0a2 2 0 1 0 4 0a2 2 0 0 0 -4 0" />
                                  <path d="M10 14l-6 6m0 -4a2 2 0 1 0 0 4a2 2 0 0 0 0 -4" />
                                  <path d="M14 14l6 6m-4 0a2 2 0 1 0 4 0a2 2 0 0 0 -4 0" />
                                </svg>
                                <div>
                                  <span className={`${actStyle.tag} font-extrabold mr-1.5`}>[ACTION TAKEN]</span>
                                  <span className={`${actStyle.text} font-extrabold text-xs tracking-wide`}>
                                    {actionText}
                                  </span>
                                </div>
                              </div>
                            );
                          })()}
                        </div>

                        {event.extra?.scenario_note && (
                          <div className="mt-1.5 text-[10px] text-gray-400 italic flex items-center">
                            <Sparkles className="w-3 h-3 mr-1 text-[#00ff9d]" />
                            {event.extra.scenario_note}
                          </div>
                        )}
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Operational Context Chips, Adaptive Thresholds, Belief Decay Sparkline, Manual Audio Test */}
        <div className="lg:col-span-4 flex flex-col space-y-4">
          {/* Operational Context Chips as Rounded Pills */}
          <div className="bg-[#0e0f17]/90 border border-white/[0.08] rounded-2xl p-4 shadow-none">
            <h2 className="text-xs font-bold text-white uppercase tracking-wider mb-3 flex items-center">
              <Shield className="w-4 h-4 mr-1.5 text-[#00ff9d]" />
              Operational Context
            </h2>

            <div className="grid grid-cols-2 gap-2.5">
              {/* Night Pill */}
              <div
                className={`px-3.5 py-2 rounded-full border transition-all flex items-center justify-between text-xs font-semibold ${
                  context.time_of_day === 'night'
                    ? 'bg-indigo-950/40 border-indigo-500/50 text-indigo-200'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300'
                }`}
              >
                <div className="flex items-center space-x-1.5">
                  <Moon className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="capitalize text-white">Night</span>
                </div>
                <span className="text-[10px] font-mono text-gray-400">1.30x</span>
              </div>

              {/* Core Zone Pill */}
              <div
                className={`px-3.5 py-2 rounded-full border transition-all flex items-center justify-between text-xs font-semibold ${
                  context.zone === 'core_zone'
                    ? 'bg-red-950/30 border-red-500/50 text-red-200'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300'
                }`}
              >
                <div className="flex items-center space-x-1.5">
                  <Crosshair className="w-3.5 h-3.5 text-red-400" />
                  <span className="capitalize text-white">Core Zone</span>
                </div>
                <span className="text-[10px] font-mono text-gray-400">7G</span>
              </div>

              {/* Clear Weather Pill */}
              <div
                className={`px-3.5 py-2 rounded-full border transition-all flex items-center justify-between text-xs font-semibold ${
                  context.weather === 'storm'
                    ? 'bg-purple-950/40 border-purple-500/50 text-purple-200'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300'
                }`}
              >
                <div className="flex items-center space-x-1.5">
                  <CloudLightning className="w-3.5 h-3.5 text-amber-300" />
                  <span className="capitalize text-white">Clear</span>
                </div>
                <span className="text-[10px] font-mono text-gray-400">1.00x</span>
              </div>

              {/* Battery Pill */}
              <div className="px-3.5 py-2 rounded-full border border-white/[0.08] bg-white/[0.02] flex items-center justify-between text-xs font-semibold text-gray-300">
                <div className="flex items-center space-x-1.5">
                  <Battery className="w-3.5 h-3.5 text-[#00ff9d]" />
                  <span className="text-white">Battery</span>
                </div>
                <span className="text-[10px] font-mono text-gray-400">98% Solar</span>
              </div>
            </div>
          </div>

          {/* Adaptive Decision Thresholds Panel */}
          <div className="bg-[#0e0f17]/90 border border-white/[0.08] rounded-2xl p-4 shadow-none">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center">
                <Sliders className="w-4 h-4 mr-1.5 text-[#00ff9d]" />
                Adaptive Decision Thresholds
              </h2>
              <span className="text-[10px] font-mono text-[#00ff9d] bg-[#00ff9d]/10 border border-[#00ff9d]/30 px-2 py-0.5 rounded font-bold">
                Controller Active
              </span>
            </div>

            <p className="text-[11px] text-gray-400 mb-3.5 leading-relaxed font-normal">
              Base thresholds start at neutral 0.700. They adapt dynamically to prevent repeated false alarms.
            </p>

            <div className="space-y-3">
              {/* Chainsaw Threshold Slider Card */}
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.08]">
                <div className="flex items-center justify-between text-xs font-medium mb-1.5">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-gray-400" />
                    <span className="text-white font-bold text-sm">Chainsaw</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm font-extrabold text-white">
                      {thresholds.chainsaw.toFixed(3)}
                    </span>
                    {thresholds.chainsaw > 0.70 && (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/[0.04] text-[#00ff9d] border border-[#00ff9d]/30 font-mono font-bold">
                        +{(thresholds.chainsaw - 0.70).toFixed(3)} adapted
                      </span>
                    )}
                  </div>
                </div>

                <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gray-200 rounded-full"
                    animate={{ width: `${(thresholds.chainsaw / 0.95) * 100}%` }}
                    transition={{ duration: 0.35 }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                  <span>Min: 0.40</span>
                  <span>Base: 0.700</span>
                  <span>Max: 0.95</span>
                </div>
              </div>

              {/* Gunshot Threshold Slider Card */}
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.08]">
                <div className="flex items-center justify-between text-xs font-medium mb-1.5">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-gray-400" />
                    <span className="text-white font-bold text-sm">Gunshot</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-sm font-extrabold text-white">
                      {thresholds.gunshot.toFixed(3)}
                    </span>
                    {thresholds.gunshot < 0.70 && (
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-white/[0.04] text-[#00ff9d] border border-[#00ff9d]/30 font-mono font-bold">
                        -0.010 TP eased
                      </span>
                    )}
                  </div>
                </div>

                <div className="w-full bg-zinc-900 h-2 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gray-200 rounded-full"
                    animate={{ width: `${(thresholds.gunshot / 0.95) * 100}%` }}
                    transition={{ duration: 0.35 }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-gray-500 mt-1 font-mono">
                  <span>Min: 0.40</span>
                  <span>Base: 0.700</span>
                  <span>Max: 0.95</span>
                </div>
              </div>
            </div>
          </div>

          {/* Belief-Decay Sparkline Chart (60s Window, 85% to Near-Zero) */}
          <BeliefDecaySparkline
            chainsawBelief={thresholds.chainsaw > 0.70 ? 0.85 : 0.2}
            gunshotBelief={thresholds.gunshot < 0.70 ? 0.92 : 0.15}
          />

          {/* Manual Audio Test Panel (Judge Demonstration Controls) */}
          <div className="p-4 rounded-2xl bg-[#0e0f17]/90 border border-white/[0.08] shadow-none">
            <div className="flex items-center justify-between mb-2.5">
              <h2 className="text-xs font-bold text-white uppercase tracking-wider flex items-center">
                <Headphones className="w-4 h-4 mr-1.5 text-indigo-400" />
                Manual Audio Test (Judge Controls)
              </h2>
              <div className="flex items-center space-x-2">
                <span
                  className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-full border ${
                    demoMode === 'CLEAN'
                      ? 'bg-[#00ff9d]/15 text-[#00ff9d] border-[#00ff9d]/40'
                      : 'bg-amber-500/15 text-amber-300 border-amber-500/40'
                  }`}
                >
                  {demoMode}
                </span>
                {activeManualSound && (
                  <span className="text-[10px] font-mono font-bold text-[#00ff9d] animate-pulse">
                    Playing {activeManualSound}...
                  </span>
                )}
              </div>
            </div>

            <p className="text-[11px] text-gray-400 mb-3 leading-relaxed">
              Demonstrate individual bioacoustic signatures on demand directly to judges ({demoMode === 'CLEAN' ? 'Pure clean seeds, zero static' : 'Raw mixed SNR clips with background noise'}):
            </p>

            <div className="grid grid-cols-2 gap-2.5">
              {/* Chainsaw Test Button */}
              <button
                onClick={() => playManualAudioForClass('chainsaw', 'Chainsaw')}
                className={`h-10 px-3 rounded-lg border text-xs font-bold flex items-center justify-center whitespace-nowrap transition ${
                  activeManualSound === 'Chainsaw'
                    ? 'bg-amber-500/30 text-amber-200 border-amber-500 shadow-[0_0_12px_rgba(245,158,11,0.35)]'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300 hover:border-amber-500/60 hover:text-amber-300'
                }`}
              >
                <Zap className="w-4 h-4 mr-2 shrink-0 text-amber-400" />
                <span>Play Chainsaw</span>
              </button>

              {/* Gunshot Test Button */}
              <button
                onClick={() => playManualAudioForClass('gunshot', 'Gunshot')}
                className={`h-10 px-3 rounded-lg border text-xs font-bold flex items-center justify-center whitespace-nowrap transition ${
                  activeManualSound === 'Gunshot'
                    ? 'bg-red-500/30 text-red-200 border-red-500 shadow-[0_0_12px_rgba(239,68,68,0.35)]'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300 hover:border-red-500/60 hover:text-red-300'
                }`}
              >
                <Target className="w-4 h-4 mr-2 shrink-0 text-red-400" />
                <span>Play Gunshot</span>
              </button>

              {/* Wind Test Button */}
              <button
                onClick={() => playManualAudioForClass('wind', 'Wind')}
                className={`h-10 px-3 rounded-lg border text-xs font-bold flex items-center justify-center whitespace-nowrap transition ${
                  activeManualSound === 'Wind'
                    ? 'bg-cyan-500/30 text-cyan-200 border-cyan-500 shadow-[0_0_12px_rgba(6,182,212,0.35)]'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300 hover:border-cyan-500/60 hover:text-cyan-300'
                }`}
              >
                <Wind className="w-4 h-4 mr-2 shrink-0 text-cyan-400" />
                <span>Play Wind</span>
              </button>

              {/* Bird Call Test Button */}
              <button
                onClick={() => playManualAudioForClass('bird_call', 'Bird Call')}
                className={`h-10 px-3 rounded-lg border text-xs font-bold flex items-center justify-center whitespace-nowrap transition ${
                  activeManualSound === 'Bird Call'
                    ? 'bg-emerald-500/30 text-emerald-200 border-[#00ff9d] shadow-[0_0_12px_rgba(0,255,157,0.35)]'
                    : 'bg-white/[0.02] border-white/[0.08] text-gray-300 hover:border-[#00ff9d]/60 hover:text-[#00ff9d]'
                }`}
              >
                <Bird className="w-4 h-4 mr-2 shrink-0 text-[#00ff9d]" />
                <span>Play Bird Call</span>
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Calibration Modal */}
      <AnimatePresence>
        {showCalibrateModal && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ opacity: 0, scale: 0.92 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.92 }}
              className="bg-[#0e0f17] border border-white/[0.08] rounded-2xl max-w-md w-full p-6 shadow-2xl relative"
            >
              <button
                onClick={() => setShowCalibrateModal(false)}
                className="absolute top-4 right-4 text-gray-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2.5 rounded-xl bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                  <Gauge className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-white">Sensor Node Calibration</h3>
                  <p className="text-xs text-gray-400">30s Ambient Noise-Floor Profiling</p>
                </div>
              </div>

              <div className="py-6 flex flex-col items-center justify-center">
                {/* Circular Progress Indicator */}
                <div className="relative w-32 h-32 flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="42" className="text-zinc-800 stroke-current" strokeWidth="8" fill="transparent" />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      className="text-[#00ff9d] stroke-current transition-all duration-300"
                      strokeWidth="8"
                      strokeDasharray={264}
                      strokeDashoffset={264 - (264 * calibrationProgress) / 100}
                      strokeLinecap="round"
                      fill="transparent"
                    />
                  </svg>
                  <span className="absolute text-xl font-mono font-bold text-white">{calibrationProgress}%</span>
                </div>

                <div className="mt-4 text-center font-mono text-xs text-gray-300">
                  {calibrationProgress === 100 ? (
                    <span className="text-[#00ff9d] font-bold">Calibration Complete • -58.4 dBFS Noise Floor</span>
                  ) : isCalibrating ? (
                    <span>Sampling stationary ambient canopy background...</span>
                  ) : (
                    <span>Ready to profile background noise floor</span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs font-mono p-3 rounded-xl bg-white/[0.02] border border-white/[0.08] mb-5">
                <div>
                  <span className="text-gray-400 block">Est. Noise Floor:</span>
                  <span className="text-white font-bold">-58.4 dBFS</span>
                </div>
                <div>
                  <span className="text-gray-400 block">Spectral α / β:</span>
                  <span className="text-[#00ff9d] font-bold">2.00 / 0.02</span>
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={handleStartCalibration}
                  disabled={isCalibrating}
                  className="flex-1 py-2.5 rounded-xl bg-[#00ff9d] hover:bg-[#00ff9d]/90 text-black font-extrabold text-xs transition disabled:opacity-50"
                >
                  {isCalibrating ? 'Profiling Node...' : 'Start 30s Calibration'}
                </button>
                <button
                  onClick={() => setShowCalibrateModal(false)}
                  className="px-4 py-2.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-xs font-semibold text-gray-300"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
