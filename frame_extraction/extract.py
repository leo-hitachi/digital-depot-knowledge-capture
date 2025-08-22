import os
import re
import cv2
import math
import json
import shutil
import subprocess

# ===== USER INPUTS =====
video_path = "../data/IMG_0383.mp4"
transcript_path = "../data/IMG_0383.vtt"   # CSV-style file (start, end, text)
output_dir = "output"
# Search settings around the midpoint of each segment
window_ms = 1000           # look ± this many milliseconds around the midpoint
num_candidates = 20        # how many timestamps to sample within the window
stabilize_reads = 2       # after seeks, throw away this many reads to stabilize decoding (0–3)
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

    # WebVTT?
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

# ----------- Video metadata via ffprobe (rotation/SAR) -----------
def get_video_display_corrections(path):
    """
    Returns (rotation_deg, sar_num, sar_den).
    If ffprobe not available, returns (0, 1, 1).
    """
    if not shutil.which("ffprobe"):
        return 0, 1, 1
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,sample_aspect_ratio,display_aspect_ratio,side_data_list:stream_tags=rotate",
        "-of", "json", path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", "ignore"))
        streams = data.get("streams", [])
        if not streams:
            return 0, 1, 1
        st = streams[0]

        # rotation
        rot = 0
        tags = st.get("tags") or {}
        if "rotate" in tags:
            try:
                rot = int(tags["rotate"]) % 360
            except Exception:
                pass
        # some ffprobe builds put rotation in side_data_list
        sdl = st.get("side_data_list") or []
        for sd in sdl:
            if sd.get("side_data_type", "").lower() == "displaymatrix":
                deg = sd.get("rotation", 0)
                try:
                    rot = int(deg) % 360
                except Exception:
                    pass

        # sample aspect ratio (sar), format like "num:den"
        sar = st.get("sample_aspect_ratio", "1:1")
        try:
            num, den = sar.split(":")
            num = int(num); den = int(den)
            if num <= 0 or den <= 0:
                num, den = 1, 1
        except Exception:
            num, den = 1, 1

        return rot, num, den
    except Exception:
        return 0, 1, 1

def apply_display_corrections(frame, rotation_deg, sar_num, sar_den):
    """
    Applies rotation and SAR scaling so saved image matches display aspect.
    """
    # Rotate first (so SAR scaling targets correct axes)
    if rotation_deg in (90, 270):
        # cv2.ROTATE_90_CLOCKWISE is 90 deg CW; metadata rotate is typically CW
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if rotation_deg == 90 else cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_deg == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Scale width by SAR so pixels become square (setsar=1)
    if sar_num != 1 or sar_den != 1:
        h, w = frame.shape[:2]
        new_w = int(round(w * (sar_num / sar_den)))
        if new_w <= 0:
            new_w = w
        # Interpolation: area for downscale, cubic for upscale
        interp = cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC
        frame = cv2.resize(frame, (new_w, h), interpolation=interp)

    return frame

# ----------- Sharpness & frame grabbing -----------
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

def pick_clearest_around_mid(cap, start_ms: int, end_ms: int, window_ms: int, num_candidates: int,
                             rotation_deg: int, sar_num: int, sar_den: int):
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    mid = start_ms + (end_ms - start_ms) // 2
    lo = clamp(mid - window_ms, start_ms, end_ms)
    hi = clamp(mid + window_ms, start_ms, end_ms)

    if num_candidates <= 1 or hi == lo:
        candidates = [mid]
    else:
        step = (hi - lo) / (num_candidates - 1)
        candidates = [int(round(lo + i * step)) for i in range(num_candidates)]

    best = (None, None, -1.0)  # (frame, ts, score)
    for ts in candidates:
        ok, frame, actual = read_frame_at_msec(cap, ts, stabilize_reads=stabilize_reads)
        if not ok or frame is None:
            # Fallback: try approximate by frame index
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_idx = int(round((ts / 1000.0) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            actual = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        if ok and frame is not None:
            # Apply display corrections BEFORE scoring/saving
            corrected = apply_display_corrections(frame, rotation_deg, sar_num, sar_den)
            score = frame_sharpness(corrected)
            if score > best[2]:
                best = (corrected, actual, score)
    return best

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
        if e < s:
            s, e = e, s

        frame, ts, score = pick_clearest_around_mid(
            cap, s, e, window_ms=window_ms, num_candidates=num_candidates,
            rotation_deg=rotation_deg, sar_num=sar_num, sar_den=sar_den
        )

        base = f"row{i:03d}_{sanitize(seg.get('text','segment'))}"
        if frame is not None and ts is not None:
            out_file = os.path.join(output_dir, f"{base}_midbest_{ts:08d}ms.jpg")
            cv2.imwrite(out_file, frame)
        else:
            print(f"⚠️  Could not select a frame for segment {i} [{s}–{e} ms] -> {base}")

    cap.release()
    print(f"✅ Done. Best frames saved under: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    import sys
    print("Python:", sys.executable)
    try:
        main()
    except Exception as ex:
        print("❌ Error:", ex)
        raise