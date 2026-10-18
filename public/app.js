import { Detector } from "./detector.js";
import { Tracker, crossedLine, mergeDetections, removeRiders } from "./tracker.js";

const MODEL = "/models/yolo11n.onnx";

const CLASSES = {
  car: { th: "รถยนต์", color: "--car" },
  truck: { th: "กระบะ/บรรทุก", color: "--truck" },
  bus: { th: "รถบัส", color: "--bus" },
  twowheel: { th: "รถสองล้อ", color: "--bike" },
  person: { th: "คน", color: "--person" },
};

const SAMPLES = {
  video: "/samples/chiangmai.mp4",
  images: ["/samples/bkk1.jpg", "/samples/bkk2.jpg", "/samples/bkk3.jpg"],
};
// Vertical line through the middle of the Chiang Mai intersection, in media
// coordinates (0..1): traffic in the clip crosses it left <-> right.
const DEFAULT_LINE = { a: { x: 0.45, y: 0.38 }, b: { x: 0.45, y: 0.82 } };
const TILE = 640;          // target tile size (px) for sliced image detection
const TILE_OVERLAP = 0.2;

const $ = (id) => document.getElementById(id);
const video = $("video"), image = $("image"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const css = getComputedStyle(document.documentElement);
const colorOf = (cls) => css.getPropertyValue(CLASSES[cls].color).trim();
const accent = () => css.getPropertyValue("--accent").trim();

const state = {
  detector: null,
  source: null,        // "video" | "image" | "upload" | "webcam"
  threshold: 0.35,
  tiled: true,
  enabled: new Set(Object.keys(CLASSES)),
  raw: [],            // last still-image detections before filtering
  detections: [],
  tracks: [],
  line: { ...DEFAULT_LINE },
  lineOn: true,
  tracker: new Tracker(),
  crossed: {},
  log: [],
  stream: null,
  lastVideoTime: -1,
  lastTs: 0,
  fps: 0,
  busy: false,
};

// ---------- setup ----------

async function loadDetector() {
  state.detector = await Detector.load(MODEL, (p) => setStatus(`กำลังโหลดโมเดล AI… ${Math.round(p * 100)}%`));
  $("delegate").textContent = state.detector.backend === "webgpu" ? "การ์ดจอ (WebGPU)" : "CPU (WebAssembly)";
}

function buildClassToggles() {
  $("classes").innerHTML = Object.entries(CLASSES).map(([key, c]) =>
    `<label><input type="checkbox" value="${key}" checked><span class="dot" style="background:${colorOf(key)}"></span>${c.th}</label>`).join("");
  $("classes").addEventListener("change", (e) => {
    e.target.checked ? state.enabled.add(e.target.value) : state.enabled.delete(e.target.value);
    if (isStill()) refreshStill();
    else renderPanel();
  });
}

// ---------- sources ----------

const isStill = () => !image.hidden;

function stopStream() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
}

function resetCounting() {
  state.tracker = new Tracker();
  state.crossed = {};
  state.log = [];
  state.tracks = [];
  renderPanel();
}

function setTab(source) {
  state.source = source;
  document.querySelectorAll(".sources button").forEach((b) => b.classList.toggle("on", b.dataset.src === source));
  $("thumbs").hidden = source !== "image";
  $("replay").hidden = true;
  setStatus("");
}

async function showVideo(src) {
  stopStream();
  image.hidden = true;
  video.hidden = false;
  video.removeAttribute("src");
  if (src instanceof MediaStream) {
    video.srcObject = src;
    state.stream = src;
  } else {
    video.src = src;
  }
  state.lastVideoTime = -1;
  resetCounting();
  await video.play().catch(() => setStatus("กดที่ภาพเพื่อเล่นวิดีโอ"));
  document.body.classList.remove("still");
}

async function showImage(src) {
  stopStream();
  video.pause();
  video.hidden = true;
  image.hidden = false;
  image.src = src;
  try {
    await image.decode();
  } catch {
    setStatus("เปิดไฟล์รูปนี้ไม่ได้");
    return;
  }
  resetCounting();
  state.raw = [];
  document.body.classList.add("still");
  refreshStill();
  detectImage();
}

function renderThumbs(active) {
  $("thumbs").innerHTML = SAMPLES.images.map((src, i) =>
    `<button data-src="${src}" class="${src === active ? "on" : ""}" aria-label="รูปตัวอย่าง ${i + 1}"><img src="${src}" alt=""></button>`).join("");
}

async function selectSource(source) {
  if (source === "upload") {
    $("file").click();
    return;
  }
  setTab(source);
  if (source === "video") {
    await showVideo(SAMPLES.video);
  } else if (source === "image") {
    renderThumbs(SAMPLES.images[0]);
    await showImage(SAMPLES.images[0]);
  } else if (source === "webcam") {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      await showVideo(stream);
    } catch {
      setStatus("เปิดกล้องไม่ได้ กรุณาอนุญาตการใช้กล้องในเบราว์เซอร์");
    }
  }
}

// ---------- detection ----------

const visible = (dets) => dets.filter((d) => state.enabled.has(d.cls) && d.score >= state.threshold);

// Sliced detection: the model sees the image at 640 px, so small cars far
// down a road disappear. Running it on overlapping tiles as well as the whole
// image finds them; the results are then merged.
async function detectSliced(img) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const all = await state.detector.detect(img);
  const cols = Math.max(1, Math.round(w / TILE)), rows = Math.max(1, Math.round(h / TILE));
  if (cols * rows === 1) return all;
  const tw = Math.ceil(w / (cols - (cols - 1) * TILE_OVERLAP));
  const th = Math.ceil(h / (rows - (rows - 1) * TILE_OVERLAP));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = Math.min(w - tw, Math.round(c * tw * (1 - TILE_OVERLAP)));
      const y = Math.min(h - th, Math.round(r * th * (1 - TILE_OVERLAP)));
      for (const d of await state.detector.detect(img, x, y, tw, th)) {
        // A box touching an inner tile edge is a cut-off object; the
        // neighbouring tile or the full-image pass sees it whole.
        const b = d.box, m = 3;
        const cut = (b.originX - x < m && x > 0) || (b.originY - y < m && y > 0) ||
          (x + tw - (b.originX + b.width) < m && x + tw < w) || (y + th - (b.originY + b.height) < m && y + th < h);
        if (!cut) all.push(d);
      }
    }
  }
  return mergeDetections(all);
}

async function detectImage() {
  if (!state.detector || state.busy || !isStill()) return;
  state.busy = true;
  setStatus(state.tiled ? "กำลังวิเคราะห์ภาพแบบละเอียด…" : "กำลังวิเคราะห์…");
  const t0 = performance.now();
  try {
    const raw = state.tiled ? await detectSliced(image) : await state.detector.detect(image);
    state.raw = removeRiders(raw);
    $("fps").textContent = `${(performance.now() - t0).toFixed(0)} ms ต่อภาพ`;
    setStatus("");
  } catch (err) {
    console.error("CountCam:", err);
    setStatus("วิเคราะห์ภาพไม่สำเร็จ");
  } finally {
    state.busy = false;
  }
  refreshStill();
}

function refreshStill() {
  state.detections = visible(state.raw ?? []);
  draw(state.detections);
  renderPanel();
}

async function tick() {
  requestAnimationFrame(tick);
  if (!state.detector || state.busy || isStill() || video.readyState < 2 || video.paused) return;
  if (video.currentTime === state.lastVideoTime && !state.stream) return;
  state.lastVideoTime = video.currentTime;

  state.busy = true;
  const ts = performance.now();
  try {
    state.detections = visible(removeRiders(await state.detector.detect(video)));
  } catch (err) {
    console.error("CountCam:", err);
    return;
  } finally {
    state.busy = false;
  }
  if (isStill()) return;   // the source changed while the model was running
  state.tracks = state.tracker.update(state.detections);
  if (state.lineOn) countCrossings(state.tracks);

  const now = performance.now();
  if (state.lastTs) state.fps = state.fps ? state.fps * 0.9 + (1000 / (now - state.lastTs)) * 0.1 : 1000 / (now - state.lastTs);
  state.lastTs = now;
  $("fps").textContent = `${state.fps.toFixed(0)} เฟรม/วินาที (${(now - ts).toFixed(0)} ms)`;

  draw(state.tracks);
  renderPanel();
}

function countCrossings(tracks) {
  const { w, h } = mediaSize();
  const a = { x: state.line.a.x * w, y: state.line.a.y * h };
  const b = { x: state.line.b.x * w, y: state.line.b.y * h };
  for (const t of tracks) {
    if (t.counted || t.history.length < 2) continue;
    if (crossedLine(t.history[0], t.history[t.history.length - 1], a, b)) {
      t.counted = true;
      t.flash = 15;
      state.crossed[t.cls] = (state.crossed[t.cls] || 0) + 1;
      state.log.push({ time: state.stream ? new Date().toLocaleTimeString("th-TH") : video.currentTime.toFixed(1), id: t.id, cls: t.cls, score: t.score });
    }
  }
}

// ---------- drawing ----------

function mediaSize() {
  return image.hidden
    ? { w: video.videoWidth || 1, h: video.videoHeight || 1 }
    : { w: image.naturalWidth || 1, h: image.naturalHeight || 1 };
}

// Where the media sits inside the stage (object-fit: contain).
function contentRect() {
  const { w, h } = mediaSize();
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  const s = Math.min(cw / w, ch / h);
  return { x: (cw - w * s) / 2, y: (ch - h * s) / 2, s };
}

function draw(items = []) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const r = contentRect();
  const small = cw < 600;
  const dense = items.length > (small ? 6 : 12);   // crowded scene: label only the bigger boxes
  const lh = small ? 15 : 18;

  ctx.font = `600 ${small ? 10 : 12}px "IBM Plex Sans Thai", sans-serif`;
  for (const d of items) {
    const { originX, originY, width, height } = d.box;
    const x = r.x + originX * r.s, y = r.y + originY * r.s, w = width * r.s, h = height * r.s;
    const color = colorOf(d.cls);
    const flash = d.flash > 0;
    ctx.lineWidth = flash ? 4 : dense ? 1.5 : 2;
    ctx.strokeStyle = flash ? accent() : color;
    ctx.strokeRect(x, y, w, h);
    if (flash) d.flash--;
    if (dense && w < 90) continue;
    const label = `${CLASSES[d.cls].th}${d.id ? " #" + d.id : ""} ${Math.round(d.score * 100)}%`;
    const tw = ctx.measureText(label).width + 8;
    const lx = Math.max(0, Math.min(x - 1, cw - tw));
    const ly = y < lh ? y : y - lh;
    ctx.fillStyle = flash ? accent() : color;
    ctx.fillRect(lx, ly, tw, lh);
    ctx.fillStyle = "#0e1116";
    ctx.fillText(label, lx + 4, ly + lh - 5);
  }

  if (state.lineOn && !isStill()) {
    const { w, h } = mediaSize();
    const ax = r.x + state.line.a.x * w * r.s, ay = r.y + state.line.a.y * h * r.s;
    const bx = r.x + state.line.b.x * w * r.s, by = r.y + state.line.b.y * h * r.s;
    ctx.strokeStyle = accent();
    ctx.lineWidth = 3;
    ctx.setLineDash([12, 8]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
    const total = Object.values(state.crossed).reduce((s, n) => s + n, 0);
    const tag = `ข้ามเส้น ${total}`;
    ctx.font = '600 14px "IBM Plex Sans Thai", sans-serif';
    const tw = ctx.measureText(tag).width + 14;
    ctx.fillStyle = accent();
    ctx.fillRect(ax - tw / 2, ay - 30, tw, 24);
    ctx.fillStyle = "#0e1116";
    ctx.fillText(tag, ax - tw / 2 + 7, ay - 13);
  }
}

function renderPanel() {
  const now = {};
  for (const d of state.detections) now[d.cls] = (now[d.cls] || 0) + 1;
  const total = state.detections.length;
  $("nowTotal").textContent = total ? `${total}` : "";
  $("now").innerHTML = listHtml(now, "ยังไม่พบวัตถุ");
  $("crossedBox").hidden = isStill();
  $("crossed").innerHTML = listHtml(state.crossed, state.lineOn ? "ยังไม่มีรถข้ามเส้น" : "ปิดการนับเส้นอยู่");
  $("log").innerHTML = state.log.slice(-5).reverse().map((l) =>
    `<li><span class="t">${l.time}${state.stream ? "" : " วิ"}</span>${CLASSES[l.cls].th} #${l.id}</li>`).join("");
}

function listHtml(counts, empty) {
  const rows = Object.keys(CLASSES).filter((k) => counts[k]).map((k) =>
    `<li><span class="dot" style="background:${colorOf(k)}"></span>${CLASSES[k].th}<span class="n">${counts[k]}</span></li>`);
  return rows.length ? rows.join("") : `<li class="empty">${empty}</li>`;
}

function setStatus(text) {
  $("status").textContent = text;
}

// ---------- line drawing ----------

function toMedia(e) {
  const rect = canvas.getBoundingClientRect();
  const r = contentRect();
  const { w, h } = mediaSize();
  const x = (e.clientX - rect.left - r.x) / r.s / w;
  const y = (e.clientY - rect.top - r.y) / r.s / h;
  return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
}

let dragging = false;
canvas.addEventListener("pointerdown", (e) => {
  if (isStill()) return;
  if (video.paused && !video.ended) video.play().catch(() => {});
  dragging = true;
  canvas.setPointerCapture(e.pointerId);
  const p = toMedia(e);
  state.line = { a: p, b: p };
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  state.line.b = toMedia(e);
  if (video.paused) draw(state.tracks);
});
canvas.addEventListener("pointerup", () => {
  if (!dragging) return;
  dragging = false;
  const { a, b } = state.line;
  // A click (no real drag) keeps the previous line instead of a dot.
  if (Math.hypot(a.x - b.x, a.y - b.y) < 0.03) state.line = { ...DEFAULT_LINE };
  resetCounting();
});

// ---------- controls ----------

$("thr").addEventListener("input", (e) => {
  state.threshold = e.target.value / 100;
  $("thrOut").textContent = `${e.target.value}%`;
  if (isStill()) refreshStill();
});
$("tiled").addEventListener("change", (e) => {
  state.tiled = e.target.checked;
  if (isStill()) detectImage();
});
$("lineOn").addEventListener("change", (e) => {
  state.lineOn = e.target.checked;
  resetCounting();
});
$("resetLine").addEventListener("click", () => {
  state.line = { ...DEFAULT_LINE };
  resetCounting();
});
document.querySelector(".sources").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn?.dataset.src) selectSource(btn.dataset.src);
});
$("thumbs").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn || state.busy) return;
  renderThumbs(btn.dataset.src);
  showImage(btn.dataset.src);
});
$("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  setTab("upload");
  const url = URL.createObjectURL(file);
  if (file.type.startsWith("video/")) await showVideo(url);
  else await showImage(url);
});
video.addEventListener("ended", () => {
  const total = Object.values(state.crossed).reduce((s, n) => s + n, 0);
  setStatus(`จบคลิป · ข้ามเส้นทั้งหมด ${total}`);
  $("replay").hidden = false;
});
$("replay").addEventListener("click", () => {
  $("replay").hidden = true;
  setStatus("");
  resetCounting();
  state.lastVideoTime = -1;
  video.currentTime = 0;
  video.play();
});
$("csv").addEventListener("click", downloadCsv);
new ResizeObserver(() => draw(isStill() ? state.detections : state.tracks)).observe(canvas);

function downloadCsv() {
  let rows;
  if (!isStill()) {
    rows = [[state.stream ? "time" : "video_time_s", "track_id", "class", "class_th", "score"],
      ...state.log.map((l) => [l.time, l.id, l.cls, CLASSES[l.cls].th, l.score.toFixed(2)])];
  } else {
    rows = [["class", "class_th", "score", "x", "y", "width", "height"],
      ...state.detections.map((d) => [d.cls, CLASSES[d.cls].th, d.score.toFixed(2),
        ...["originX", "originY", "width", "height"].map((k) => Math.round(d.box[k]))])];
  }
  const blob = new Blob(["﻿" + rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "countcam.csv" });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- start ----------

buildClassToggles();
renderPanel();
if (new URLSearchParams(location.search).has("debug")) window.countcam = state;
try {
  await loadDetector();
  setStatus("");
  await selectSource(new URLSearchParams(location.search).get("src") === "image" ? "image" : "video");
  requestAnimationFrame(tick);
} catch (err) {
  console.error("CountCam:", err);
  setStatus(err.message);
}
