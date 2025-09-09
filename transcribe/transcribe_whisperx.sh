clear
time uvx whisperx "../data/IMG_0381.mp4" \
  --model large-v3 \
  --align_model WAV2VEC2_ASR_LARGE_LV60K_960H \
  --compute_type int8 \
  --print_progress True \
  --segment_resolution chunk \
  --max_line_width 50 \
  --language en