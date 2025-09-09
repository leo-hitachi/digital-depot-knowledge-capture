import json
import os
import glob
import csv
import re

files = {
    "base": "../data/transcription_compare/whisper/IMG_0381_base.json",
    "large": "../data/transcription_compare/whisper/IMG_0381_large.json",
    "medium": "../data/transcription_compare/whisper/IMG_0381_medium.json",
    "small": "../data/transcription_compare/whisper/IMG_0381_small.json",
    "tiny": "../data/transcription_compare/whisper/IMG_0381_tiny.json",
    "turbo": "../data/transcription_compare/whisper/IMG_0381_turbo.json",
}

# files = {
#     "base": "../data/whisper vs x/whisper_x_IMG_0381.json",
#     "large": "../data/whisper vs x/whisper_IMG_0381.json"     
# }

# Normalization: lowercase + remove punctuation
def normalize_word(w):
    return re.sub(r"[^\w\s]", "", w.lower()).strip()

def load_words(path):
    with open(path, "r") as f:
        data = json.load(f)
    words = []
    for seg in data["segments"]:
        for w in seg["words"]:
            orig = w["word"].strip()
            norm = normalize_word(orig)
            if norm:  # skip empty after stripping punctuation
                words.append({
                    "word": orig,
                    "norm": norm,
                    "start": w.get("start"),
                    "end": w.get("end")
                })
    return words

# Needleman–Wunsch alignment at word level
def align_words(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        back[i][0] = "up"
    for j in range(m + 1):
        dp[0][j] = j
        back[0][j] = "left"
    back[0][0] = None

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1]["norm"] == hyp[j - 1]["norm"]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "diag"
            else:
                choices = [
                    (dp[i - 1][j] + 1, "up"),
                    (dp[i][j - 1] + 1, "left"),
                    (dp[i - 1][j - 1] + 1, "diag"),
                ]
                dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])

    # Traceback
    i, j = n, m
    aligned = []
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag":
            aligned.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif move == "up":
            aligned.append((ref[i - 1], None))
            i -= 1
        elif move == "left":
            aligned.append((None, hyp[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned, dp[n][m]

def wer_accuracy(ref, hyp):
    _, dist = align_words(ref, hyp)
    return 100 * (1 - dist / max(len(ref), 1))

# Load reference
ref_words = load_words(files["large"])

summary = {}

for name, path in files.items():
    if name == "large":
        continue
    hyp_words = load_words(path)
    aligned, dist = align_words(ref_words, hyp_words)
    acc = 100 * (1 - dist / max(len(ref_words), 1))
    summary[name] = acc

    # Write to CSV
    out_path = f"comparison_{name}.csv"
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "ref_word", "ref_start", "ref_end",
            "hyp_word", "hyp_start", "hyp_end",
            "comparison"
        ])
        for r, h in aligned:
            if r and h:
                comp = "MATCH" if r["norm"] == h["norm"] else "SUBSTITUTION"
                writer.writerow([r["word"], r["start"], r["end"],
                                 h["word"], h["start"], h["end"], comp])
            elif r and not h:
                writer.writerow([r["word"], r["start"], r["end"],
                                 "", "", "", "DELETION"])
            elif h and not r:
                writer.writerow(["", "", "",
                                 h["word"], h["start"], h["end"], "INSERTION"])

    print(f"{name}: {acc:.2f}% (saved to {out_path})")

# Summary
print("\n=== Overall Word-Level Accuracy vs Large ===")
for model, acc in summary.items():
    print(f"{model:6s}: {acc:.2f}%")