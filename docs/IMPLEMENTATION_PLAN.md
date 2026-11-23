# IMPLEMENTATION_PLAN - CogniFusion

## Phase status (all shipped)
1. Core pipeline: feature extraction, dataset builder, trainer, eval - DONE
2. Real-time inference + logging: realtime_pipeline, DataLogger, SessionManager - DONE
3. Colab notebook: 100 cells, 18 sections, verified - DONE
4. Research variants: seq_len, ablate, grl, swa, ema, kd, feat-noise/dropout, cutmix, gce, sam, rdrop, LOSO - DONE
5. Deployment: ONNX export, INT8 quantization, Dockerfile, CI - DONE

## Remaining phases
- Phase 6: real-data end-to-end run (needs hours of compute; parquet path ready)
- Phase 7: hardware validation with Muse + webcam (needs devices)
- Phase 8: paper writeup / benchmark release

## Rules for agents
Work one phase at a time; verify with a short training run before pushing; update this doc's status table when a phase completes.

---
## For AI agents
Work one phase at a time. Finish and verify it (compile + smoke train), summarize, then stop before the next phase. Update the status table when a phase completes.
