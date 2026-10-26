"use strict";

const EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
const EMOJI = { angry: "😠", disgust: "🤢", fear: "😨", happy: "😄", sad: "😢", surprise: "😲", neutral: "😐" };
const LIVE_INTERVAL_MS = 250;   // ~4 analyses per second in live mode
const SMOOTHING = 0.6;          // live mode: weight of the previous frame's probabilities
const MAX_CAPTURE_WIDTH = 640;

const $ = (id) => document.getElementById(id);
const video = $("video"), still = $("still"), overlay = $("overlay"), placeholder = $("placeholder");
const btnStart = $("btn-start"), btnAnalyze = $("btn-analyze"), btnStop = $("btn-stop");
const liveToggle = $("live"), fileInput = $("file"), statusEl = $("status");

const color = (e) => getComputedStyle(document.documentElement).getPropertyValue(`--${e}`).trim();

let stream = null;
let source = null;          // "camera" | "image" | null
let busy = false;
let liveTimer = null;
let lastFaces = [];         // last rendered faces (for redraws and live smoothing)
let stats = freshStats();

function freshStats() {
  return { analyses: 0, faces: 0, confidenceSum: 0, counts: Object.fromEntries(EMOTIONS.map((e) => [e, 0])) };
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function updateButtons() {
  const hasSource = source !== null;
  btnStart.disabled = source === "camera";
  btnStop.disabled = source !== "camera";
  btnAnalyze.disabled = !hasSource || liveToggle.checked;
  liveToggle.disabled = source !== "camera";
  placeholder.hidden = hasSource;
}

// ---------- sources ----------

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (err) {
    setStatus(`Could not access the camera: ${err.message}. You can still upload an image.`, true);
    return;
  }
  still.hidden = true;
  video.hidden = false;
  video.classList.add("mirrored");
  video.srcObject = stream;
  await video.play();
  source = "camera";
  clearResults();
  updateButtons();
  setStatus("Camera running. Click “Analyze Emotion” or turn on live mode.");
}

function stopCamera() {
  stopLive();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  source = null;
  clearOverlay();
  updateButtons();
  setStatus("Camera stopped.");
}

function loadImage(file) {
  if (!file) return;
  if (source === "camera") stopCamera();
  const url = URL.createObjectURL(file);
  still.onload = () => {
    source = "image";
    video.hidden = true;
    still.hidden = false;
    clearResults();
    updateButtons();
    analyze();
  };
  still.src = url;
}

// ---------- analysis ----------

function captureBlob() {
  const media = source === "camera" ? video : still;
  const w = source === "camera" ? video.videoWidth : still.naturalWidth;
  const h = source === "camera" ? video.videoHeight : still.naturalHeight;
  const scale = Math.min(1, MAX_CAPTURE_WIDTH / w);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  canvas.getContext("2d").drawImage(media, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
}

async function analyze({ live = false } = {}) {
  if (busy || source === null) return;
  busy = true;
  const t0 = performance.now();
  try {
    const form = new FormData();
    form.append("image", await captureBlob(), "frame.jpg");
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const data = await res.json();
    const faces = live ? smooth(data.faces) : data.faces;
    lastFaces = faces.map((f) => ({ ...f, imageWidth: data.image.width, imageHeight: data.image.height }));
    drawOverlay();
    renderResults(faces);
    recordStats(faces);
    const rtt = Math.round(performance.now() - t0);
    setStatus(`${faces.length} face${faces.length === 1 ? "" : "s"} · server ${data.timing_ms.total} ms ` +
      `(detect ${data.timing_ms.detect}, classify ${data.timing_ms.classify}) · round trip ${rtt} ms`);
  } catch (err) {
    setStatus(`Analysis failed: ${err.message}`, true);
  } finally {
    busy = false;
  }
}

/** Exponential smoothing of each face's probabilities against the nearest face in the previous frame. */
function smooth(faces) {
  return faces.map((face) => {
    const cx = face.box.x + face.box.w / 2, cy = face.box.y + face.box.h / 2;
    let best = null, bestDist = Infinity;
    for (const prev of lastFaces) {
      const d = Math.hypot(prev.box.x + prev.box.w / 2 - cx, prev.box.y + prev.box.h / 2 - cy);
      if (d < bestDist) { bestDist = d; best = prev; }
    }
    if (!best || bestDist > face.box.w * 0.5) return face;
    const probs = {};
    for (const e of EMOTIONS) probs[e] = SMOOTHING * best.probabilities[e] + (1 - SMOOTHING) * face.probabilities[e];
    const emotion = EMOTIONS.reduce((a, b) => (probs[a] >= probs[b] ? a : b));
    return { ...face, probabilities: probs, emotion, confidence: probs[emotion] };
  });
}

function startLive() {
  btnAnalyze.disabled = true;
  const loop = async () => {
    const t = performance.now();
    await analyze({ live: true });
    if (!liveToggle.checked) return;
    liveTimer = setTimeout(loop, Math.max(0, LIVE_INTERVAL_MS - (performance.now() - t)));
  };
  loop();
}

function stopLive() {
  liveToggle.checked = false;
  clearTimeout(liveTimer);
  liveTimer = null;
  updateButtons();
}

// ---------- rendering ----------

function clearOverlay() {
  lastFaces = [];
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
}

function clearResults() {
  clearOverlay();
  $("results").innerHTML = '<p class="empty">No analysis yet.</p>';
}

/** Draw face boxes, accounting for object-fit: contain letterboxing and the mirrored selfie view. */
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const W = overlay.clientWidth, H = overlay.clientHeight;
  overlay.width = W * dpr;
  overlay.height = H * dpr;
  const ctx = overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!lastFaces.length) return;

  const { imageWidth: iw, imageHeight: ih } = lastFaces[0];
  const s = Math.min(W / iw, H / ih);
  const ox = (W - iw * s) / 2, oy = (H - ih * s) / 2;
  const mirrored = source === "camera";

  for (const f of lastFaces) {
    const w = f.box.w * s, h = f.box.h * s;
    const x = mirrored ? ox + (iw - f.box.x - f.box.w) * s : ox + f.box.x * s;
    const y = oy + f.box.y * s;
    const c = color(f.emotion);
    ctx.lineWidth = 3;
    ctx.strokeStyle = c;
    ctx.strokeRect(x, y, w, h);

    const label = `${f.id}. ${f.emotion} ${(f.confidence * 100).toFixed(0)}%`;
    ctx.font = "600 14px system-ui, sans-serif";
    const tw = ctx.measureText(label).width + 12;
    const ty = y > 24 ? y - 24 : y + h;
    ctx.fillStyle = c;
    ctx.fillRect(x - 1.5, ty, tw, 22);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 5, ty + 16);
  }
}

function barsHtml(values) {
  return `<div class="bars">${EMOTIONS.map((e) => `
    <span class="bar-label">${e}</span>
    <span class="bar"><span style="width:${(values[e] * 100).toFixed(1)}%;background:${color(e)}"></span></span>
    <span class="bar-value">${(values[e] * 100).toFixed(1)}%</span>`).join("")}</div>`;
}

function renderResults(faces) {
  const el = $("results");
  if (!faces.length) {
    el.innerHTML = '<p class="empty">No face detected. Face the camera with good lighting.</p>';
    return;
  }
  el.innerHTML = faces.map((f) => `
    <div class="face">
      <div class="face-head">
        <span class="face-emoji" aria-hidden="true">${EMOJI[f.emotion]}</span>
        <div>
          <div class="face-title">Face ${f.id}: ${f.emotion}</div>
          <div class="face-sub">Position x ${f.box.x}, y ${f.box.y} · ${f.box.w}×${f.box.h} px</div>
        </div>
        <span class="face-conf">${(f.confidence * 100).toFixed(1)}%</span>
      </div>
      ${barsHtml(f.probabilities)}
    </div>`).join("");
}

function recordStats(faces) {
  stats.analyses += 1;
  for (const f of faces) {
    stats.faces += 1;
    stats.counts[f.emotion] += 1;
    stats.confidenceSum += f.confidence;
  }
  renderStats();
}

function renderStats() {
  $("stat-analyses").textContent = stats.analyses;
  $("stat-faces").textContent = stats.faces;
  if (stats.faces === 0) {
    $("stat-common").textContent = "—";
    $("stat-conf").textContent = "—";
    $("distribution").innerHTML = "";
    return;
  }
  const top = EMOTIONS.reduce((a, b) => (stats.counts[a] >= stats.counts[b] ? a : b));
  $("stat-common").textContent = `${EMOJI[top]} ${top}`;
  $("stat-conf").textContent = `${((stats.confidenceSum / stats.faces) * 100).toFixed(1)}%`;
  const shares = Object.fromEntries(EMOTIONS.map((e) => [e, stats.counts[e] / stats.faces]));
  $("distribution").innerHTML = barsHtml(shares);
}

// ---------- wiring ----------

btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);
btnAnalyze.addEventListener("click", () => analyze());
liveToggle.addEventListener("change", () => (liveToggle.checked ? startLive() : stopLive()));
$("btn-upload").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { loadImage(fileInput.files[0]); fileInput.value = ""; });
$("btn-reset").addEventListener("click", () => { stats = freshStats(); renderStats(); });
window.addEventListener("resize", drawOverlay);

fetch("/api/info").then((r) => r.json()).then((info) => {
  const acc = info.test_metrics ? ` · FER-2013 test acc <strong>${(info.test_metrics.accuracy * 100).toFixed(1)}%</strong>` : "";
  $("model-badge").innerHTML = `<strong>${info.model.arch}</strong> · ${info.detector} detector${acc}`;
}).catch(() => { $("model-badge").textContent = "Model info unavailable"; });

updateButtons();
renderStats();
