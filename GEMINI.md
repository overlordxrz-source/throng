# THRONG Agent — Mandatory Context

## SESSION START (non-negotiable)
Before ANY code or architectural decision:
1. Read THRONG.md §0, §3, §4 in full
2. Do NOT rely on KI summaries alone — they lose scientific nuance

## HARD PROHIBITIONS
- Never remove the VQ bottleneck from the primary signal channel
- Never wipe /mnt/throng-runs/ckpt_step_* without explicit Cam approval
- Never replace GloVe with hand-crafted primitives in the Rosetta Stone
- Never mark architectural changes COMPLETE in THRONG.md before Cam review
- Never pkill + relaunch without verifying process count with pgrep afterward
- Never interpret ATE = 0 as anything except: receivers ignore the signal

## WORKFLOW
- Cam sets architecture. Will implements and verifies.
- Smoke test must pass before Modal compute is spent.
- All architectural decisions go through User → Cam first.
