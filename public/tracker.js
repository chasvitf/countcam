// Box helpers and a small IoU tracker. The tracker gives each detection a
// stable id across frames, so a car that crosses the counting line is
// counted once.

const MAX_MISSED = 15;    // frames a track may go undetected before it is dropped
const MIN_IOU = 0.15;
const HISTORY = 30;

// Classes the model swaps frame to frame (car <-> truck) may match each other.
const GROUP = { car: "vehicle", truck: "vehicle", bus: "vehicle", twowheel: "twowheel", person: "person" };

function area(b) {
  return b.width * b.height;
}

function intersection(a, b) {
  const x1 = Math.max(a.originX, b.originX), y1 = Math.max(a.originY, b.originY);
  const x2 = Math.min(a.originX + a.width, b.originX + b.width);
  const y2 = Math.min(a.originY + a.height, b.originY + b.height);
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

function iou(a, b) {
  const inter = intersection(a, b);
  return inter / (area(a) + area(b) - inter || 1);
}

function center(box) {
  return { x: box.originX + box.width / 2, y: box.originY + box.height / 2 };
}

function union(a, b) {
  const x1 = Math.min(a.originX, b.originX), y1 = Math.min(a.originY, b.originY);
  const x2 = Math.max(a.originX + a.width, b.originX + b.width);
  const y2 = Math.max(a.originY + a.height, b.originY + b.height);
  return { originX: x1, originY: y1, width: x2 - x1, height: y2 - y1 };
}

// Merge detections from overlapping passes (whole image + tiles). Boxes of
// the same group that overlap a lot, or where one sits inside the other,
// are the same object: keep the best score and the outer box.
export function mergeDetections(dets, maxIou = 0.45, maxInside = 0.7) {
  const kept = [];
  for (const d of [...dets].sort((p, q) => q.score - p.score)) {
    const same = kept.find((k) => {
      if (GROUP[k.cls] !== GROUP[d.cls]) return false;
      const inter = intersection(k.box, d.box);
      return iou(k.box, d.box) > maxIou || inter / Math.min(area(k.box), area(d.box)) > maxInside;
    });
    if (same) same.box = union(same.box, d.box);
    else kept.push({ ...d, box: { ...d.box } });
  }
  return kept;
}

// A person on a motorbike is the rider, not a pedestrian: drop people whose
// box overlaps a two-wheeler so they are not counted twice.
export function removeRiders(dets) {
  const bikes = dets.filter((d) => d.cls === "twowheel");
  if (!bikes.length) return dets;
  return dets.filter((d) => d.cls !== "person" ||
    !bikes.some((b) => intersection(d.box, b.box) / area(d.box) > 0.25));
}

export class Tracker {
  constructor() {
    this.tracks = [];
    this.nextId = 1;
  }

  update(detections) {
    const pairs = [];
    this.tracks.forEach((t, ti) => detections.forEach((d, di) => {
      if (GROUP[t.cls] !== GROUP[d.cls]) return;
      let score = iou(t.box, d.box);
      if (score < MIN_IOU) {
        // Fast or briefly lost objects: fall back to a nearby centre of similar size.
        const c1 = center(t.box), c2 = center(d.box);
        const reach = 0.6 * Math.max(t.box.width, t.box.height);
        const sizeRatio = area(d.box) / area(t.box);
        if (Math.hypot(c1.x - c2.x, c1.y - c2.y) > reach || sizeRatio < 0.5 || sizeRatio > 2) return;
        score = MIN_IOU * 0.5;
      }
      pairs.push({ ti, di, score });
    }));
    pairs.sort((p, q) => q.score - p.score);

    const usedT = new Set(), usedD = new Set();
    for (const { ti, di } of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti);
      usedD.add(di);
      const t = this.tracks[ti], d = detections[di];
      t.box = d.box;
      t.score = d.score;
      t.missed = 0;
      t.votes[d.cls] = (t.votes[d.cls] || 0) + 1;
      t.cls = Object.entries(t.votes).sort((p, q) => q[1] - p[1])[0][0];
      t.history.push(center(d.box));
      if (t.history.length > HISTORY) t.history.shift();
    }

    this.tracks.forEach((t, ti) => { if (!usedT.has(ti)) t.missed++; });
    this.tracks = this.tracks.filter((t) => t.missed <= MAX_MISSED);

    detections.forEach((d, di) => {
      if (usedD.has(di)) return;
      this.tracks.push({
        id: this.nextId++, cls: d.cls, score: d.score, box: d.box,
        votes: { [d.cls]: 1 }, history: [center(d.box)], missed: 0, counted: false, flash: 0,
      });
    });

    return this.tracks.filter((t) => t.missed === 0);
  }
}

function side(p, a, b) {
  return Math.sign((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));
}

// True when the path from -> to crosses segment a-b.
export function crossedLine(from, to, a, b) {
  const s1 = side(from, a, b), s2 = side(to, a, b);
  if (s1 === 0 || s2 === 0 || s1 === s2) return false;
  return side(a, from, to) !== side(b, from, to);
}
