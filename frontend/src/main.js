const $ = (id) => document.getElementById(id);

const REASON_LABELS = {
  off_topic: "off-topic question — outside the indexed corpus",
  unsafe: "unsafe input",
  ungrounded: "draft answer failed the grounding check",
};
const RETRIEVAL_BUDGET_MS = 200;
// The 200ms criterion measures ONLY these stages (latency.py RETRIEVAL_STAGES).
// guardrail_check/generation/stt carry no 200ms claim — see latency_report.md.
const RETRIEVAL_STAGE_NAMES = new Set(["embed_query", "vector_search", "chunk_assembly"]);
const TARGET_SAMPLE_RATE = 16000; // STT-friendly, keeps uploads small

let mediaStream = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;
let pcmChunks = [];
let recording = false;

function setStatus(text) {
  $("status").textContent = text;
}

function showError(msg) {
  const box = $("error-box");
  box.textContent = msg;
  box.classList.remove("hidden");
}

function clearTransient() {
  $("error-box").classList.add("hidden");
  $("result").classList.add("hidden");
  $("refusal-banner").classList.add("hidden");
  $("unverified-banner").classList.add("hidden");
  $("warnings-wrap").classList.add("hidden");
}

function setRecordingUI(on) {
  recording = on;
  $("record-btn").textContent = on ? "⏹ Stop & send" : "🎙 Start recording";
  $("record-btn").classList.toggle("recording", on);
  setStatus(on ? "recording… click to stop" : "idle");
}

// --- WAV capture -------------------------------------------------------------
// The STT provider (Sarvam) only accepts mp3/wav, and browsers' MediaRecorder
// produces webm/opus — which got us a 400 on the deployed backend. So we skip
// MediaRecorder entirely: capture raw mic PCM via WebAudio, downmix/resample,
// and encode a proper WAV blob client-side.

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showError(`Microphone unavailable: ${err.message}`);
    return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  // createScriptProcessor is deprecated but works everywhere; an AudioWorklet
  // would need a separate module file — overkill for this deliberately
  // minimal UI. Buffer size 4096 ≈ 85ms per callback at 48kHz.
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  pcmChunks = [];
  processor.onaudioprocess = (e) => {
    pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  sourceNode.connect(processor);
  // The processor node must reach destination or some browsers stop pulling
  // onaudioprocess — but wiring it straight to the speakers creates a mic→
  // speaker→mic feedback loop that contaminated every recording (the
  // "transcripts are gibberish" bug). Zero-gain sink keeps the graph alive
  // and silent.
  const silentSink = audioCtx.createGain();
  silentSink.gain.value = 0;
  processor.connect(silentSink);
  silentSink.connect(audioCtx.destination);
  setRecordingUI(true);
}

function stopRecording() {
  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
  }
  if (sourceNode) sourceNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  const chunks = pcmChunks;
  pcmChunks = [];
  setRecordingUI(false);
  if (!audioCtx || chunks.length === 0) {
    showError("Nothing was recorded.");
    return;
  }
  const srcRate = audioCtx.sampleRate;
  audioCtx.close();
  audioCtx = null;

  encodeWavAsync(chunks, srcRate)
    .then((blob) => {
      const fd = new FormData();
      fd.append("file", blob, "recording.wav");
      send(fd);
    })
    .catch((err) => showError(`Audio encoding failed: ${err.message}`));
}

async function encodeWavAsync(chunks, srcRate) {
  const totalLen = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(totalLen);
  let off = 0;
  for (const c of chunks) {
    merged.set(c, off);
    off += c.length;
  }

  let samples = merged;
  let rate = srcRate;
  if (srcRate !== TARGET_SAMPLE_RATE) {
    const frames = Math.ceil((totalLen * TARGET_SAMPLE_RATE) / srcRate);
    const offline = new OfflineAudioContext(1, frames, TARGET_SAMPLE_RATE);
    const buf = offline.createBuffer(1, totalLen, srcRate);
    buf.copyToChannel(merged, 0);
    const bs = offline.createBufferSource();
    bs.buffer = buf;
    bs.connect(offline.destination);
    bs.start();
    samples = (await offline.startRendering()).getChannelData(0);
    rate = TARGET_SAMPLE_RATE;
  }
  return encodeWav(samples, rate);
}

function encodeWav(samples, sampleRate) {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const writeStr = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  v.setUint32(4, 36 + dataBytes, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  v.setUint32(16, 16, true); // PCM chunk size
  v.setUint16(20, 1, true); // format = PCM
  v.setUint16(22, 1, true); // channels = mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  v.setUint32(40, dataBytes, true);
  let o = 44;
  for (let i = 0; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function sendText() {
  const q = $("text-query").value.trim();
  if (!q) return;
  const fd = new FormData();
  fd.append("query", q);
  await send(fd);
}

async function send(formData) {
  clearTransient();
  setStatus("transcribing + retrieving + generating…");
  try {
    const res = await fetch("/ask", { method: "POST", body: formData });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail || detail;
      } catch {}
      showError(`Backend error ${res.status}: ${detail}`);
    } else {
      render(await res.json());
    }
  } catch (err) {
    showError(`Request failed: ${err.message}`);
  }
  setStatus("idle");
}

function latencyRows(trace, retrievalMs, totalMs) {
  const rows = Object.entries(trace || {}).map(([stage, ms]) => ({
    // "+" prefix marks the rows that sum into the retrieval total below
    stage: RETRIEVAL_STAGE_NAMES.has(stage) ? `+ ${stage}` : stage,
    ms,
    highlight: RETRIEVAL_STAGE_NAMES.has(stage),
  }));
  const underBudget = retrievalMs < RETRIEVAL_BUDGET_MS;
  rows.push({
    stage: "retrieval_ms — sum of the + rows",
    ms: retrievalMs,
    highlight: true,
    note: underBudget ? "✓ under 200ms" : "✗ OVER 200ms",
  });
  rows.push({ stage: "total_ms", ms: totalMs });
  return rows;
}

function render(data) {
  $("result").classList.remove("hidden");
  $("transcript").textContent = data.transcript || "(no audio submitted)";

  if (data.refused) {
    const banner = $("refusal-banner");
    const reason = data.refusal_reason
      ? ` (${REASON_LABELS[data.refusal_reason] || data.refusal_reason})`
      : "";
    banner.textContent = `Refused${reason}`;
    banner.classList.remove("hidden");
    $("unverified-banner").classList.add("hidden");
  } else if (!data.grounding_verified) {
    $("unverified-banner").classList.remove("hidden");
  }

  $("answer").textContent = data.answer;

  const warnings = data.warnings || [];
  if (warnings.length > 0) {
    $("warnings-wrap").classList.remove("hidden");
    $("warnings").replaceChildren(
      ...warnings.map((w) => Object.assign(document.createElement("li"), { textContent: w }))
    );
  }

  const tbody = document.querySelector("#latency-table tbody");
  tbody.replaceChildren(
    ...latencyRows(data.latency_trace_ms, data.retrieval_ms, data.total_ms).map((r) => {
      const tr = document.createElement("tr");
      if (r.highlight && r.ms <= RETRIEVAL_BUDGET_MS) tr.className = "ok";
      if (r.highlight && r.ms > RETRIEVAL_BUDGET_MS) tr.className = "over";
      tr.append(Object.assign(document.createElement("td"), { textContent: r.stage }));
      tr.append(
        Object.assign(document.createElement("td"), {
          textContent: Number(r.ms).toFixed(1) + (r.note ? `  ${r.note}` : ""),
        })
      );
      return tr;
    })
  );
}

$("record-btn").addEventListener("click", () => (recording ? stopRecording() : startRecording()));
$("send-btn").addEventListener("click", sendText);
$("text-query").addEventListener("keydown", (e) => e.key === "Enter" && sendText());
