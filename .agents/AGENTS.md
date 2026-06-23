# THRONG Project Rules

## 1. Cam Context Window Resets (Horcrux Protocol)
If the user indicates that Cam has lost context or has hallucinated an outdated project state (e.g., asking to plan a phase that is already live):
- **DO NOT** just summarize the chat.
- **DO** write an authoritative "Context Sync" prompt for the user to copy-paste to Cam.
- The prompt MUST use Cam's analytical lenses (Software / Physics / RL) and explicitly state the exact current Phase, the active Git branch, the current codebook/action dimensionality, and the precise empirical telemetry from the cluster.

## 2. JAX Parameter Grafting (Amnesia Prevention)
When modifying the architecture to expand the output dimensions of existing neural network heads (e.g., increasing actions from 8 to 12):
- **NEVER** allow the layer to reinitialize due to shape mismatch. This causes catastrophic amnesia of the agents' survival policies.
- **ALWAYS** ensure `jax_sim/network_jax.py:graft_missing_param_subtrees` gracefully zero-pads the expansion. Kernels must be padded on `axis=1` (outputs), and biases on `axis=0`.

## 3. Multi-Slot Array Handling
With the transition to the Phase 18 multi-slot communication architecture, agent tokens are 2D arrays `(N, slots)`.
- When writing offline analysis or telemetry scripts that use scalar operations like `np.bincount()`, **always check `ndim`** and explicitly slice the active slot (e.g., `toks[:, 0]`) to prevent `ValueError: object too deep for desired array`.
- Do not assume `vq_token` or `token_ids` are scalar integers.

## 4. Signal Corpus Field Names
The signal corpus (`signal_corpus.jsonl`) uses these field names. Do NOT guess alternatives:
- `scout` (boolean) — NOT `is_scout`
- `vq_token` (list of 3 ints for [slot_0, slot_1, slot_2]) — NOT `token_ids`
- `nb_scout_token_lag1` (int or None) — lag-1 scout token received by this agent
- `red_dist`, `red_bear`, `resource`, `energy`, `neighbors` — context features
- `sig` (list of 40 floats) — raw 40D signal vector (8D cont + 32D discrete)
- `action` (int 0–11) — executed action index

## 5. Modal Account Migration
When switching between Modal accounts, you must follow this exact 4-step sequence to avoid data loss or permission errors:
1. **Auth to Old Account:** Run `.modal-cli/bin/modal token new` and authenticate in the browser to the account that currently holds the volume.
2. **Download Data:** Run `./scripts/migrate_modal.sh download ~/throng_backup` to pull all checkpoints and corpora locally. Wait for this to fully complete.
3. **Auth to New Account:** Run `.modal-cli/bin/modal token new` AGAIN and authenticate in the browser to the target destination account.
4. **Upload Data:** Run `./scripts/migrate_modal.sh upload ~/throng_backup` to create the new volume and push the local data up.

**Log location:** When training is launched from a Modal Notebook (Jupyter), logs go to cell output, NOT to `/mnt/throng-runs/train.log` on the volume. Use `modal volume get` to fetch volume files, or ask the user to paste cell output.

## 6. Strategic Roadmap Awareness
When onboarding or making high-level architectural decisions, you **MUST** consult `docs/STRATEGIC_ROADMAP.md`. 
- Do not mistake "adding features" (like crafting) for progress towards AGI. Features exist only to increase the **environmental complexity ceiling** and force **compositional syntax**.
- The true path to AGI lies in: 1) Persistent Culture (Phase 19 writing system), 2) Open-Ended Complexity (Phase 21 procedural environments), 3) Internal Reasoning (Phase 22 language as a cognitive tool), and 4) Cross-Domain Transfer (Phase 23).
- Always contextualize the current phase within this broader strategic trajectory.

## 7. Modal Launch Cell Constraints
Any Python cell generated to launch training (`run_bg.py`) on Modal MUST adhere to these exact constraints:
1. **Fresh Container Bootstrap:** Always include `git clone https://github.com/overlordxrz-source/throng.git /root/throng || true` and `pip install -r requirements.txt` before fetching or running.
2. **Bulletproof `cwd`:** Never use `cd` in the `subprocess.Popen` bash string. Always use the `cwd="/root/throng"` argument in the Python call.
3. **Regex `pgrep`:** When verifying the process wipe, use `pgrep -af '[p]ython -u'` (the brackets prevent `pgrep` from matching its own command string).
4. **Detachment:** You must pass `start_new_session=True` to `subprocess.Popen` to prevent Jupyter `SIGINT`s from killing the run when the user stops a `tail -f` log cell.
