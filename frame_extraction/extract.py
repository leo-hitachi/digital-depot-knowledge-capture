import os
import re
import cv2
import math
import json
import shutil
import subprocess
from typing import List, Tuple


# ===== USER INPUTS =====
video_path = "../data/IMG_0381.mp4"
transcript_path = "../data/IMG_0381.vtt"   # CSV-style file (start, end, text)
output_dir = "output"

# Sampling across EACH sentence/segment
sample_stride_ms = 80       # step between samples inside [start, end] (e.g., 80ms ≈ 12.5 fps)
stabilize_reads = 1         # extra reads after a seek (0–3), helps move off keyframes

# Selection rules (use either top_k or threshold; top_k takes precedence if > 0)
top_k = 5                   # keep up to K sharpest frames per segment (set 0 to disable)
min_sharpness = 80.0        # variance of Laplacian threshold (used only if top_k == 0)
min_gap_ms = 250            # minimum time gap between saved frames to avoid near-duplicates

# Output naming
jpeg_quality = 95
max_name_len = 40
# ===============================

os.makedirs(output_dir, exist_ok=True)

def _time_to_ms_vtt(ts: str) -> int:
    ts = ts.strip()
    h, m, s = 0, 0, 0.0
    parts = ts.split(":")
    if len(parts) == 3:
        h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
    elif len(parts) == 2:
        m = int(parts[0]); s = float(parts[1])
    else:
        return int(float(ts) * 1000)
    return int((h * 3600 + m * 60 + s) * 1000)

def parse_transcript(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        raw = f.read()

    if "-->" in raw:
        entries, lines, i = [], [ln.strip("\n\r") for ln in raw.splitlines()], 0
        while i < len(lines):
            line = lines[i].strip(); i += 1
            if not line or line.upper() == "WEBVTT":
                continue
            if "-->" in line:
                try:
                    a, b = [p.strip() for p in line.split("-->")]
                    start_ms = _time_to_ms_vtt(a)
                    end_ms = _time_to_ms_vtt(b.split()[0])
                except Exception:
                    continue
                text_parts = []
                while i < len(lines):
                    nxt = lines[i].strip()
                    if not nxt or "-->" in nxt:
                        break
                    text_parts.append(nxt); i += 1
                text = " ".join(text_parts).strip()
                if text:
                    entries.append({"start": start_ms, "end": end_ms, "text": text})
        return entries

    # Numeric format: start end text ...
    entries, header_seen = [], False
    for ln in raw.splitlines():
        line = ln.strip()
        if not line:
            continue
        lower = line.lower()
        if not header_seen and ("start" in lower and "end" in lower):
            header_seen = True
            continue

        tokens = re.split(r"[ \t]+", line, maxsplit=2)
        if len(tokens) < 3:
            parts = line.split()
            if len(parts) < 3:
                continue
            start_str, end_str = parts[0], parts[1]
            text = " ".join(parts[2:])
        else:
            start_str, end_str, text = tokens[0], tokens[1], tokens[2]

        def to_ms(x: str) -> int:
            x = x.strip()
            if re.fullmatch(r"\d+", x):
                val = int(x)
                return val if val >= 1000 else val * 1000
            return int(float(x) * 1000)

        try:
            s = to_ms(start_str); e = to_ms(end_str)
        except Exception:
            continue
        if e < s:
            s, e = e, s
        entries.append({"start": s, "end": e, "text": text.strip()})
    return entries

def sanitize(s: str, max_len: int = 40) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return (s[:max_len] or "segment").strip("_-.")

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ----- Rotation & SAR via ffprobe -----
def get_video_display_corrections(path) -> Tuple[int, int, int]:
    if not shutil.which("ffprobe"):
        return 0, 1, 1
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=sample_aspect_ratio,side_data_list:stream_tags=rotate",
        "-of", "json", path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", "ignore"))
        streams = data.get("streams", [])
        if not streams:
            return 0, 1, 1
        st = streams[0]

        rot = 0
        tags = st.get("tags") or {}
        if "rotate" in tags:
            try: rot = int(tags["rotate"]) % 360
            except Exception: pass
        for sd in st.get("side_data_list") or []:
            if sd.get("side_data_type", "").lower() == "displaymatrix":
                try: rot = int(sd.get("rotation", 0)) % 360
                except Exception: pass

        sar = st.get("sample_aspect_ratio", "1:1")
        try:
            num, den = sar.split(":"); num = int(num); den = int(den)
            if num <= 0 or den <= 0: num, den = 1, 1
        except Exception:
            num, den = 1, 1
        return rot, num, den
    except Exception:
        return 0, 1, 1

def apply_display_corrections(frame, rotation_deg, sar_num, sar_den):
    if rotation_deg in (90, 270):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if rotation_deg == 90 else cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_deg == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    if sar_num != 1 or sar_den != 1:
        h, w = frame.shape[:2]
        new_w = int(round(w * (sar_num / sar_den)))
        if new_w > 0 and new_w != w:
            interp = cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC
            frame = cv2.resize(frame, (new_w, h), interpolation=interp)
    return frame

# ----- Sharpness & grabbing -----
def frame_sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def read_frame_at_msec(cap, msec: int, stabilize_reads: int = 0):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
    ok, frame = cap.read()
    for _ in range(stabilize_reads if ok else 0):
        _ok, _frame = cap.read()
        if _ok:
            ok, frame = _ok, _frame
        else:
            break
    actual = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    return ok, frame, actual

def nonmax_suppress_temporal(cands: List[Tuple[int, float]], min_gap_ms: int, top_k: int = 0, min_sharpness: float = 0.0):
    """
    cands: list of (timestamp_ms, sharpness_score)
    Greedy: take best score, suppress neighbors within ±min_gap_ms.
    If top_k>0 -> keep up to K; else keep all >= min_sharpness.
    Returns list of kept (timestamp_ms, sharpness_score) sorted by time.
    """
    if not cands:
        return []
    # sort by score desc
    cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)
    kept = []
    for ts, sc in cands_sorted:
        if top_k == 0 and sc < min_sharpness:
            continue
        if any(abs(ts - kts) < min_gap_ms for kts, _ in kept):
            continue
        kept.append((ts, sc))
        if top_k > 0 and len(kept) >= top_k:
            break
    return sorted(kept, key=lambda x: x[0])

def main():
    print("Using OpenCV from:", cv2.__file__)
    rotation_deg, sar_num, sar_den = get_video_display_corrections(video_path)
    print(f"Display corrections -> rotation: {rotation_deg}°, SAR: {sar_num}:{sar_den}")

    segments = parse_transcript(transcript_path)
    if not segments:
        raise SystemExit(f"No usable segments parsed from: {transcript_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = 0
    if total_frames > 0:
        duration_ms = int(math.floor((total_frames / fps) * 1000))
    if duration_ms <= 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, 10_000_000)
        duration_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or 0

    print(f"Video FPS: {fps:.3f}, frames: {total_frames}, duration ~ {duration_ms/1000:.2f}s")

    for i, seg in enumerate(segments):
        s = clamp(seg["start"], 0, max(0, duration_ms - 1)) if duration_ms > 0 else seg["start"]
        e = clamp(seg["end"],   0, max(0, duration_ms - 1)) if duration_ms > 0 else seg["end"]
        if e < s: s, e = e, s
        if e <= s:
            print(f"Skip tiny segment {i} [{s}-{e}]"); continue

        # Sample timestamps across the entire segment
        span = e - s
        stride = max(1, int(sample_stride_ms))
        candidates_ts = list(range(s, e + 1, stride))

        # Score each sampled frame
        scored: List[Tuple[int, float]] = []
        for ts in candidates_ts:
            ok, frame, actual = read_frame_at_msec(cap, ts, stabilize_reads=stabilize_reads)
            if not ok or frame is None:
                # fallback via frame index
                frame_idx = int(round((ts / 1000.0) * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                actual = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if ok and frame is not None:
                corrected = apply_display_corrections(frame, rotation_deg, sar_num, sar_den)
                score = frame_sharpness(corrected)
                scored.append((actual, score))

        if not scored:
            print(f"⚠️  No frames scored for segment {i} [{s}-{e}]")
            continue

        # Select multiple sharp frames
        kept = nonmax_suppress_temporal(
            scored,
            min_gap_ms=min_gap_ms,
            top_k=top_k,
            min_sharpness=min_sharpness
        )

        # base = f"{i:04d}_{sanitize(seg.get('text','segment'), max_name_len)}"
        base = f"{i:04d}"
        if not kept:
            print(f"ℹ️  No frames passed selection for segment {i}; saving the single best as fallback.")
            best_ts, best_sc = max(scored, key=lambda x: x[1])
            ok, frame, _ = read_frame_at_msec(cap, best_ts, stabilize_reads=stabilize_reads)
            if ok and frame is not None:
                corrected = apply_display_corrections(frame, rotation_deg, sar_num, sar_den)
                out_file = os.path.join(output_dir, f"{base}_{best_ts:08d}.jpg")
                cv2.imwrite(out_file, corrected, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            continue

        # Save all selected frames
        for n, (ts, sc) in enumerate(kept, start=1):
            ok, frame, _ = read_frame_at_msec(cap, ts, stabilize_reads=stabilize_reads)
            if ok and frame is not None:
                corrected = apply_display_corrections(frame, rotation_deg, sar_num, sar_den)
                # out_file = os.path.join(output_dir, f"{base}_{n:02d}_{ts:08d}ms_{sc:.1f}.jpg")
                out_file = os.path.join(output_dir, f"{base}_{ts:08d}.jpg")
                cv2.imwrite(out_file, corrected, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    cap.release()
    print(f"✅ Done. Frames saved under: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import sys
    print("Python:", sys.executable)
    try:
        main()
    except Exception as ex:
        print("❌ Error:", ex)
        raise