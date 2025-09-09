clear
time whisper "../data/IMG_0381.mp4" \
  --model large-v3 \
  --fp16 False \
  --language English \
  --condition_on_previous_text False \
  --temperature 0 \
  --no_speech_threshold 0.6 \
  --logprob_threshold -1.0 \
  --word_timestamps True \
  --output_format all