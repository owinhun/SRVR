# SVHalluc Phase-0 audio control

This directory implements only the fixed 48-sample CMTB Phase 0:

1. freeze `data/phase0_cmtb.jsonl` from the official `samples.json`;
2. create the speech/background separation-and-recombination control `O_proc`;
3. mux the recombined audio with the untouched video stream;
4. write per-sample QC to `outputs/audio_phase0.jsonl`.

It does not generate B, T_env, T_rev, silence, or shifted speech.

```bash
/usr/bin/python src/audio/phase0.py manifest
/usr/bin/python src/audio/phase0.py process --device cuda
```

Selection is deterministic (`seed=20260728`) and the manifest is never
overwritten. Existing valid result rows and media are reused. A failed sample is
recorded as failed and is never replaced with another sample.

QC gates:

- absolute container-duration difference at most 0.05 seconds;
- O_proc RMS within 0.5 dB of decoded original;
- no decoded O_proc samples at or above 0.9999 absolute peak;
- VAD speech present with first/last boundary shift at most 0.25 seconds;
- 44.1 kHz audio;
- successful ffprobe and media decode.

The output video stream is copied (`-c:v copy`); only the recombined audio is
encoded as 192 kbps AAC.
