# BRIEFING — 2026-06-16T01:44:27+03:00

## Mission
Coordinate the InverseTribe project, decomposing work into milestones, executing worker iterations, and verifying the integration.

## 🔒 My Identity
- Archetype: Project Orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: /Users/overlord/CascadeProjects/throng/.agents/orchestrator/
- Original parent: main agent
- Original parent conversation ID: 8ab62324-4d0b-4fae-b344-a541dedbdd7d

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: /Users/overlord/CascadeProjects/throng/PROJECT.md
1. **Decompose**: Decompose the InverseTribe project into milestones. Define cross-module interface contracts.
2. **Dispatch & Execute**:
   - **Delegate**: Spawn sub-orchestrators for milestones or run the iteration loop (Explorer -> Worker -> Reviewer -> Challenger -> Auditor -> Gate).
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (sub-orchestrators only, last resort)
4. **Succession**: Self-succeed when cumulative sub-agent spawn count >= 16 and all subagents are complete.
- **Work items**:
  1. Setup environment [pending]
  2. Implement tribe_grad.py [pending]
  3. Implement roi_atlas.py and baseline_cache.py [pending]
  4. Implement generate.py [pending]
  5. Implement FastAPI server.py [pending]
  6. Implement React ui [pending]
  7. E2E verification [pending]
- **Current phase**: 1
- **Current focus**: Setup environment

## 🔒 Key Constraints
- Always call me(The User) by my name, Dimitar.
- Never remove the VQ bottleneck from the primary signal channel.
- Never wipe /mnt/throng-runs/ckpt_step_* without explicit Cam approval.
- Never replace GloVe with hand-crafted primitives in the Rosetta Stone.
- Never mark architectural changes COMPLETE in THRONG.md before Cam review.
- Never pkill + relaunch without verifying process count with pgrep afterward.
- Never interpret ATE = 0 as anything except: receivers ignore the signal.
- ANY launch cell generated must include `start_new_session=True` as a mandatory non-negotiable parameter in `subprocess.Popen`.
- Before telling the user it's safe to stop a monitoring cell, first confirm the training process was launched with `start_new_session=True`.
- The 'dragonbg' server is hosted on Modal. Connect using `modal shell ta-01KV6PEESNZMDNR82XFKR143JX` or use `modal run` for executions.

## Current Parent
- Conversation ID: 8ab62324-4d0b-4fae-b344-a541dedbdd7d
- Updated: 2026-06-16T02:00:00+03:00

## Key Decisions Made
- Initial setup
- Course-corrected subagents to use Modal shell/run instead of SSH for dragonbg connectivity

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_env_wrapper | teamwork_preview_explorer | Environment check and wrapper exploration | completed | a482eb3b-4a7c-4b64-b7b9-b4fe0fcd9adb |
| self_e2e_testing | self | E2E Testing Track Orchestrator | in-progress | 2622d0b8-a2d6-48d3-bbbe-043d93d42331 |
| self_m1_env_wrapper | self | Milestone 1 Sub-orchestrator | in-progress | 3d64c6cf-cda7-4244-a25e-faa8d5255153 |

## Succession Status
- Succession required: no
- Spawn count: 3 / 16
- Pending subagents: 2622d0b8-a2d6-48d3-bbbe-043d93d42331, 3d64c6cf-cda7-4244-a25e-faa8d5255153
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: 9eac3848-8fc5-4cf4-adbd-c717cc7055dd/task-29
- Safety timer: 9eac3848-8fc5-4cf4-adbd-c717cc7055dd/task-145

## Artifact Index
- /Users/overlord/CascadeProjects/throng/PROJECT.md — Global index for the project
- /Users/overlord/CascadeProjects/throng/.agents/orchestrator/progress.md — Progress tracking
- /Users/overlord/CascadeProjects/throng/.agents/orchestrator/plan.md — Detailed plan
