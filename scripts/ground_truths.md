# Manual Ground Truths — Speaker Diarization Test Clips

Obtained by user manually listening and timing each clip (2026-03-06).

## Clip 1: clip1_theater_conversation.mp3
- Duration: 202s
- **Speaker B total** (including overlap): ~35s (revised upward from initial 26-30s estimate)
- **Overlap** (both speaking): ~10s
- **Silence**: 5-7s

Derived values (using B=35s, overlap=10s, silence=6s):
- B exclusive: 35 - 10 = 25s
- A exclusive: 202 - 6 - 25 - 10 = 161s
- **A total** (including overlap): 161 + 10 = **~171s**

## Clip 2: clip2_tv_study_interview.mp3
- Duration: 751s
- **Speaker A total** (including overlap): ~195s (3 min 15 sec)
- **Overlap** (both speaking): 30-40s
- **Silence**: >10s (pipeline measures 12.1s, which feels right)

Derived values (using overlap=35s, silence=12s):
- A exclusive: 195 - 35 = 160s
- Total speech: 751 - 12 = 739s
- B exclusive: 739 - 160 - 35 = 544s
- B total (including overlap): 544 + 35 = 579s

## Notes
- "Total" for a speaker means wall-clock time they are speaking, including moments of overlap.
- In our dual-assignment model, "Speech A" already includes overlap credited to A, so it is directly comparable to the ground truth totals above.
- SpeechBrain's silence (1.7s clip 2, 2.4s clip 1) is too low — user confirms pauses/hesitations are genuine silence.
