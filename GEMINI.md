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
- ANY launch cell generated must include `start_new_session=True` as a mandatory non-negotiable parameter in `subprocess.Popen`.
- Before telling the user it's safe to stop a monitoring cell, first confirm the training process was launched with `start_new_session=True`.

## WORKFLOW
- Cam sets architecture. Will implements and verifies.
- Smoke test must pass before Modal compute is spent.
- All architectural decisions go through User → Cam first.
