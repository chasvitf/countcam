// YOLO11n object detector running in the browser with ONNX Runtime Web.
// WebGPU when the browser has it, otherwise WebAssembly on the CPU.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// Threads need cross-origin isolation; without it ORT warns and uses one.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
ort.env.logLevel = "error";

const SIZE = 640;          // model input (square, letterboxed)
const ANCHORS = 8400;      // output columns of YOLO11 at 640 px
const MIN_SCORE = 0.2;     // drop anything weaker before NMS
const NMS_IOU = 0.5;

// COCO class index -> CountCam class. Motorcycle and bicycle are merged: the
// model mixes them up on Thai roads, and both are "รถสองล้อ" when counting.
const WANTED = { 0: "person", 1: "twowheel", 2: "car", 3: "twowheel", 5: "bus", 7: "truck" };
const WANTED_IDS = Object.keys(WANTED).map(Number);
const GROUP = { car: "vehicle", truck: "vehicle", bus: "vehicle", twowheel: "twowheel", person: "person" };

export class Detector {
  static async load(url, onProgress) {
    const model = await fetchWithProgress(url, onProgress);
    const attempts = navigator.gpu ? [["webgpu"], ["wasm"]] : [["wasm"]];
    for (const providers of attempts) {
      try {
        const session = await ort.InferenceSession.create(model, { executionProviders: providers, graphOptimizationLevel: "all" });
        const detector = new Detector(session, providers[0]);
        await detector.warmUp();
        return detector;
      } catch (err) {
        console.warn(`CountCam: ${providers[0]} backend unavailable`, err);
      }
    }
    throw new Error("เปิดโมเดลไม่สำเร็จ เบราว์เซอร์นี้อาจไม่รองรับ ลองใช้ Chrome หรือ Edge รุ่นล่าสุด");
  }

  constructor(session, backend) {
    this.session = session;
    this.backend = backend;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = SIZE;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.input = new Float32Array(3 * SIZE * SIZE);
  }

  async warmUp() {
    this.ctx.fillStyle = "#727272";
    this.ctx.fillRect(0, 0, SIZE, SIZE);
    await this.run();
  }

  // Detect objects in `source` (image, video or canvas), or in the region
  // sx, sy, sw, sh of it. Boxes come back in source pixel coordinates.
  async detect(source, sx = 0, sy = 0, sw = sourceWidth(source), sh = sourceHeight(source)) {
    const scale = Math.min(SIZE / sw, SIZE / sh);
    const dw = Math.round(sw * scale), dh = Math.round(sh * scale);
    const padX = Math.floor((SIZE - dw) / 2), padY = Math.floor((SIZE - dh) / 2);
    this.ctx.fillStyle = "#727272";
    this.ctx.fillRect(0, 0, SIZE, SIZE);
    this.ctx.drawImage(source, sx, sy, sw, sh, padX, padY, dw, dh);
    const out = await this.run();

    const found = [];
    for (let i = 0; i < ANCHORS; i++) {
      let best = 0, bestId = -1;
      for (const id of WANTED_IDS) {
        const s = out[(4 + id) * ANCHORS + i];
        if (s > best) { best = s; bestId = id; }
      }
      if (best < MIN_SCORE) continue;
      const cx = out[i], cy = out[ANCHORS + i], w = out[2 * ANCHORS + i], h = out[3 * ANCHORS + i];
      const box = {
        originX: (cx - w / 2 - padX) / scale + sx,
        originY: (cy - h / 2 - padY) / scale + sy,
        width: w / scale,
        height: h / scale,
      };
      found.push({ cls: WANTED[bestId], score: best, box: clip(box, sx, sy, sw, sh) });
    }
    return nms(found);
  }

  async run() {
    const { data } = this.ctx.getImageData(0, 0, SIZE, SIZE);
    const area = SIZE * SIZE, input = this.input;
    for (let p = 0, i = 0; p < area; p++, i += 4) {
      input[p] = data[i] / 255;
      input[p + area] = data[i + 1] / 255;
      input[p + 2 * area] = data[i + 2] / 255;
    }
    const tensor = new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]);
    const result = await this.session.run({ images: tensor });
    return result.output0.data;
  }
}

function sourceWidth(s) {
  return s.videoWidth || s.naturalWidth || s.width;
}

function sourceHeight(s) {
  return s.videoHeight || s.naturalHeight || s.height;
}

function clip(b, sx, sy, sw, sh) {
  const x1 = Math.max(sx, b.originX), y1 = Math.max(sy, b.originY);
  const x2 = Math.min(sx + sw, b.originX + b.width), y2 = Math.min(sy + sh, b.originY + b.height);
  return { originX: x1, originY: y1, width: Math.max(0, x2 - x1), height: Math.max(0, y2 - y1) };
}

function iou(a, b) {
  const x1 = Math.max(a.originX, b.originX), y1 = Math.max(a.originY, b.originY);
  const x2 = Math.min(a.originX + a.width, b.originX + b.width);
  const y2 = Math.min(a.originY + a.height, b.originY + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.width * a.height + b.width * b.height - inter || 1);
}

// Non-maximum suppression within a group, so a pickup seen as both "car"
// and "truck" is reported once.
function nms(dets) {
  dets.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const d of dets) {
    if (!kept.some((k) => GROUP[k.cls] === GROUP[d.cls] && iou(k.box, d.box) > NMS_IOU)) kept.push(d);
  }
  return kept;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`โหลดโมเดลไม่สำเร็จ (${res.status})`);
  // content-length may be the compressed size, so it only drives the progress bar.
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(Math.min(1, got / total));
  }
  const buf = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) {
    buf.set(c, at);
    at += c.length;
  }
  return buf;
}
