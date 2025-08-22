whisper "../data/IMG_0381.mp4" \
  --model large-v3 \
  --condition_on_previous_text False \
  --temperature 0 \
  --no_speech_threshold 0.6 \
  --logprob_threshold -1.0 \
  --output_format all