import json
import os
import glob

files = {
    "base": "../data/transcription_compare/IMG_0381_base.json",
    "large": "../data/transcription_compare/IMG_0381_large.json",
    "medium": "../data/transcription_compare/IMG_0381_medium.json",
    "small": "../data/transcription_compare/IMG_0381_small.json",
    "tiny": "../data/transcription_compare/IMG_0381_tiny.json",
    "turbo": "../data/transcription_compare/IMG_0381_turbo.json",
}

# Load words from WhisperX JSON
def load_words(path):
    with open(path, "r") as f:
        data = json.load(f)
    words = []
    for seg in data["segments"]:
        for w in seg["words"]:
            # normalize punctuation/case
            word = w["word"].strip().lower()
            # drop punctuation-only tokens
            if word.isalnum():
                words.append(word)
    return words

# Levenshtein distance at word-level
def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,   # deletion
                dp[i][j - 1] + 1,   # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[n][m]

def wer_accuracy(ref, hyp):
    dist = levenshtein(ref, hyp)
    return 100 * (1 - dist / max(len(ref), 1))

# Load reference (large model)
ref_words = load_words(files["large"])

results = {}
for name, path in files.items():
    if name == "large":
        continue
    hyp_words = load_words(path)
    acc = wer_accuracy(ref_words, hyp_words)
    results[name] = acc

# Print results
print("Model Word-Level Accuracy vs Large:")
for model, acc in results.items():
    print(f"{model:6s}: {acc:.2f}%")