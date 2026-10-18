"""Count vehicles and people crossing a line in a video with YOLO11 + ByteTrack.

Example:
    python count_video.py ../public/samples/chiangmai.mp4 --out out.mp4 --csv crossings.csv

The line is given in relative coordinates (0..1) as x1,y1,x2,y2. The default is
the vertical line used by the web demo for the Chiang Mai sample clip.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO ids -> label. Motorcycle and bicycle are merged into one "two-wheeler".
CLASSES = {0: "person", 1: "two-wheeler", 2: "car", 3: "two-wheeler", 5: "bus", 7: "truck"}
COLORS = {  # BGR
    "car": (255, 179, 79),
    "truck": (67, 159, 255),
    "bus": (255, 139, 195),
    "two-wheeler": (122, 224, 126),
    "person": (139, 107, 255),
}
ACCENT = (63, 207, 255)


def side(p, a, b) -> int:
    v = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return (v > 0) - (v < 0)


def crossed(prev, cur, a, b) -> bool:
    """True when the path prev -> cur crosses segment a-b."""
    s1, s2 = side(prev, a, b), side(cur, a, b)
    if s1 == 0 or s2 == 0 or s1 == s2:
        return False
    return side(a, prev, cur) != side(b, prev, cur)


def is_rider(person, bikes) -> bool:
    """A person whose box overlaps a two-wheeler is its rider, not a pedestrian."""
    px1, py1, px2, py2 = person
    area = max(1, (px2 - px1) * (py2 - py1))
    for bx1, by1, bx2, by2 in bikes:
        inter = max(0, min(px2, bx2) - max(px1, bx1)) * max(0, min(py2, by2) - max(py1, by1))
        if inter / area > 0.25:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="input video file, or a camera index such as 0")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights (default: yolo11n.pt)")
    ap.add_argument("--line", default="0.45,0.38,0.45,0.82", help="counting line x1,y1,x2,y2 in 0..1")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum confidence (default: 0.35)")
    ap.add_argument("--out", help="write the annotated video here (.mp4)")
    ap.add_argument("--csv", help="write one row per crossing here")
    ap.add_argument("--show", action="store_true", help="show a live window (press q to stop)")
    args = ap.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    x1, y1, x2, y2 = (float(v) for v in args.line.split(","))
    a, b = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))

    model = YOLO(args.model)
    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Recent centres per track: a track counts once its path over the last
    # second crosses the line, which survives jittery boxes and short misses.
    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    counted: set[int] = set()
    riders: set[int] = set()   # person tracks seen on a two-wheeler stay riders
    totals: Counter[str] = Counter()
    rows = []
    frame_no = 0
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=args.conf,
                             classes=list(CLASSES), verbose=False)[0]

        flashes = set()
        if result.boxes.id is not None:
            dets = list(zip(result.boxes.xyxy.tolist(), result.boxes.id.int().tolist(),
                            result.boxes.cls.int().tolist(), result.boxes.conf.tolist()))
            bikes = [box for box, _, cls_id, _ in dets if CLASSES[cls_id] == "two-wheeler"]
            for box, tid, cls_id, score in dets:
                label = CLASSES[cls_id]
                if label == "person" and (tid in riders or is_rider(box, bikes)):
                    riders.add(tid)
                    continue
                bx1, by1, bx2, by2 = map(int, box)
                center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
                path = history[tid]
                path.append(center)
                if tid not in counted and len(path) > 1 and crossed(path[0], center, a, b):
                    counted.add(tid)
                    totals[label] += 1
                    flashes.add(tid)
                    rows.append({"time_s": round(frame_no / fps, 2), "track_id": tid, "class": label,
                                 "confidence": round(score, 2)})
                color = ACCENT if tid in flashes else COLORS[label]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 4 if tid in flashes else 2)
                text = f"{label} #{tid} {score:.0%}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw + 6, by1), color, -1)
                cv2.putText(frame, text, (bx1 + 3, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (22, 17, 14), 1, cv2.LINE_AA)

        cv2.line(frame, a, b, ACCENT, 3, cv2.LINE_AA)
        summary = [f"crossed: {sum(totals.values())}"] + [f"  {k}: {v}" for k, v in totals.most_common()]
        for i, line in enumerate(summary):
            cv2.putText(frame, line, (16, 34 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (16, 34 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, ACCENT, 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("CountCam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            out = csv.DictWriter(f, fieldnames=["time_s", "track_id", "class", "confidence"])
            out.writeheader()
            out.writerows(rows)

    took = time.time() - started
    print(f"{frame_no} frames in {took:.1f}s ({frame_no / took:.1f} fps)")
    print(f"crossed the line: {sum(totals.values())}", dict(totals))
    if args.out:
        print(f"video: {Path(args.out).resolve()}")
    if args.csv:
        print(f"csv:   {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()
