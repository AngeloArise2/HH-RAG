const $ = (id) => document.getElementById(id);

const REASON_LABELS = {
  off_topic: "off-topic question — outside the indexed corpus",
  unsafe: "unsafe input",
  ungrounded: "draft answer failed the grounding check",
};
const RETRIEVAL_BUDGET_MS = 200;

let recorder = null;
let chunks = [];
let recording = false;
let mediaStream = null;

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

async function toggleRecording() {
  if (recording) {
    recorder.stop(); // onstop does the send
    return;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showError(`Microphone unavailable: ${err.message}`);
    return;
  }
  const mime =
    MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "";
  recorder = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : undefined);
  chunks = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  recorder.onstop = () => {
    mediaStream.getTracks().forEach((t) => t.stop());
    setRecordingUI(false);
    if (chunks.length === 0) {
      showError("Nothing was recorded.");
      return;
    }
    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    const fd = new FormData();
    fd.append("file", blob, "recording.webm");
    send(fd);
  };
  recorder.start();
  setRecordingUI(true);
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
  const rows = Object.entries(trace || {}).map(([stage, ms]) => ({ stage, ms }));
  rows.push({ stage: "retrieval_ms (budget 200)", ms: retrievalMs, highlight: true });
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
      tr.append(Object.assign(document.createElement("td"), { textContent: Number(r.ms).toFixed(1) }));
      return tr;
    })
  );
}

$("record-btn").addEventListener("click", toggleRecording);
$("send-btn").addEventListener("click", sendText);
$("text-query").addEventListener("keydown", (e) => e.key === "Enter" && sendText());
