import os
import re
import cv2
import math

# ===== USER INPUTS =====
video_path = "../data/IMG_0383.mp4"
transcript_path = "../data/IMG_0383.vtt"   # CSV-style file (start, end, text)
output_dir = "output"
# =======================

os.makedirs(output_dir, exist_ok=True)

def _time_to_ms_vtt(ts: str) -> int:
    """
    Convert VTT timestamp 'HH:MM:SS.mmm' or 'MM:SS.mmm' to milliseconds.
    """
    ts = ts.strip()
    h, m, s = 0, 0, 0.0
    parts = ts.split(":")
    if len(parts) == 3:
        h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
    elif len(parts) == 2:
        m = int(parts[0]); s = float(parts[1])
    else:
        # Fallback: seconds as float
        return int(float(ts) * 1000)
    return int((h * 3600 + m * 60 + s) * 1000)

def parse_transcript(path):
    """
    Returns a list of dicts: [{"start": ms, "end": ms, "text": str}, ...]
    Supports:
      1) Numeric lines: <start_ms><space/tab><end_ms><space><text...>
         Optionally with a header containing 'start' and 'end'.
      2) WebVTT cues: lines like '00:00:08.120 --> 00:00:17.280' followed by one or more text lines.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        raw = f.read()

    # Heuristic: WebVTT if it contains '-->'
    if "-->" in raw:
        entries = []
        lines = [ln.strip("\n\r") for ln in raw.splitlines()]
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.upper() == "WEBVTT":
                continue
            # Timestamp line?
            if "-->" in line:
                try:
                    a, b = [p.strip() for p in line.split("-->")]
                    start_ms = _time_to_ms_vtt(a)
                    end_ms = _time_to_ms_vtt(b.split()[0])  # ignore settings after end time
                except Exception:
                    continue
                # Gather following text lines until blank or next timestamp
                text_parts = []
                while i < len(lines):
                    nxt = lines[i].strip()
                    # stop on blank line or next time cue
                    if not nxt or "-->" in nxt:
                        break
                    text_parts.append(nxt)
                    i += 1
                text = " ".join(text_parts).strip()
                if text:
                    entries.append({"start": start_ms, "end": end_ms, "text": text})
            # else ignore (cue ids, etc.)
        return entries

    # Otherwise: numeric rows like "start end text"
    entries = []
    header_seen = False
    for ln in raw.splitlines():
        line = ln.strip()
        if not line:
            continue
        # Allow a header row
        lower = line.lower()
        if not header_seen and ("start" in lower and "end" in lower):
            header_seen = True
            continue

        # Split the first two numeric tokens; the rest is text (which may contain spaces)
        # Handle tabs or multiple spaces.
        tokens = re.split(r"[ \t]+", line, maxsplit=2)
        if len(tokens) < 3:
            # Try again with more liberal splitting; skip if still invalid
            parts = line.split()
            if len(parts) < 3:
                continue
            start_str, end_str = parts[0], parts[1]
            text = " ".join(parts[2:])
        else:
            start_str, end_str, text = tokens[0], tokens[1], tokens[2]

        # Coerce to ms; if numbers look like seconds, convert to ms
        def to_ms(x: str) -> int:
            x = x.strip()
            if re.fullmatch(r"\d+", x):  # integer
                val = int(x)
                # heuristic: treat >= 1000 as ms, else seconds
                return val if val >= 1000 else val * 1000
            else:
                # float seconds
                return int(float(x) * 1000)

        try:
            start_ms = to_ms(start_str)
            end_ms = to_ms(end_str)
        except Exception:
            continue

        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms

        entries.append({"start": start_ms, "end": end_ms, "text": text.strip()})

    return entries

def sanitize(s: str, max_len: int = 40) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return (s[:max_len] or "segment").strip("_-.")

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def save_frame_at_millis(cap, millis: int, out_path: str):
    # Seek by time (ms). Some backends only seek to nearest keyframe; we also do a safety read.
    cap.set(cv2.CAP_PROP_POS_MSEC, float(millis))
    ok, frame = cap.read()
    if not ok or frame is None:
        # Fallback: try seeking by frame index approximation
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = int(round((millis / 1000.0) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    if ok and frame is not None:
        cv2.imwrite(out_path, frame)
    else:
        print(f"⚠️  Could not grab frame at {millis} ms -> {out_path}")

def main():
    print("Using OpenCV from:", cv2.__file__)
    segments = parse_transcript(transcript_path)
    if not segments:
        raise SystemExit(f"No usable segments parsed from: {transcript_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_ms = int(math.floor((total_frames / fps) * 1000)) if total_frames > 0 else int(cap.get(cv2.CAP_PROP_POS_MSEC))  # fallback
    if not duration_ms or duration_ms <= 0:
        # last resort: try to probe by seeking to a very large time then reading back position
        cap.set(cv2.CAP_PROP_POS_MSEC, 10_000_000)
        duration_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or 0

    print(f"Video FPS: {fps:.3f}, frames: {int(total_frames)}, duration ~ {duration_ms/1000:.2f}s")

    for i, seg in enumerate(segments):
        s = clamp(seg["start"], 0, max(0, duration_ms - 1))
        e = clamp(seg["end"],   0, max(0, duration_ms - 1))
        if e < s:
            s, e = e, s
        mid = s + (e - s) // 2

        base = f"row{i:03d}_{sanitize(seg['text'])}"
        for tag, t in (("start", s), ("mid", mid), ("end", e)):
            out_file = os.path.join(output_dir, f"{base}_{tag}_{t:08d}ms.jpg")
            save_frame_at_millis(cap, t, out_file)

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