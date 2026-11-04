# Config audit triage -- 44 unread keys (`scripts/config_audit.py` at a675253)

For Cam's ruling. Rule to apply (Cam): allowlist only a feature that is deliberately off, with a reason;
everything else gets wired or deleted. Each read below comes from a grep of jax_sim/ and scripts/. It is
my read, not a ruling. "Dead" = nothing in the code implements it, or the code uses a different source,
so the key can be deleted without changing behaviour.

## PPO / learning (3)
- `ppo_epochs: 1` -- **should-be-wired**. `ppo_update` always does exactly one pass, whatever the value.
  Matches config today, so there is no behaviour change at 1. It is Cam's next lever after the estimator.
- `ppo_warmup_steps: 15` -- **dead**. There is no learning-rate warmup anywhere; the only warmup is the comms
  unfreeze ramp, which has its own key.
- `action_entropy_coef: 0.05` -- **should-be-wired or deleted** (Cam: not wired yet; entropy is rising on its
  own). `ppo_loss` uses `ppo_entropy_coef` 0.02 only.

## Channel / signal (9) -- resolved by the channel fix
- `gumbel_tau`, `gumbel_tau_start`, `gumbel_tau_end`, `gumbel_tau_anneal_start`, `gumbel_tau_anneal_end` --
  **should-be-wired** as the token-sampling temperature in the channel fix. Today tokens are a deterministic
  argmin with no noise, so nothing reads a temperature.
- `signal_entropy_coef_start/_end`, `signal_entropy_anneal_start/_end` -- **should-be-wired** as the token
  entropy bonus in the same fix. Both coefficients are 0.0 today, so there is no behaviour change either way.

## Theory of mind (4)
- `tom_enabled: true`, `tom_lr`, `tom_reward_coef: 0.0`, `tom_head_independent` -- **dead**. `head_tom` is
  computed every forward pass (`b_tom = b_outs.tom_logits`), but no loss trains it and nothing reads it.
  `tom_enabled: true` claims a feature that isn't running.

## Puzzle (4) -- config claims control it doesn't have
- `puzzle_enabled`, `puzzle_n_nodes: 3`, `puzzle_switch_distance`, `puzzle_timeout` -- **should-be-wired or the
  puzzle deleted**. The puzzle runs **unconditionally**: `generate_puzzle_nodes(..., n_nodes=3)` is hardcoded,
  and `puzzle_grid` is a live observation channel. Setting `puzzle_enabled: false` would silently do nothing.

## Legacy curriculum / spawn / population (8)
- `curriculum_red_count`, `curriculum_blue_threshold`, `curriculum_sustain_steps` -- **dead**, superseded by
  `ctd_competence_ramp`.
- `red_spawn_step: 2000` -- **dead**. Reds exist from step 0.
- `base_maturation_steps` -- **dead**. Nothing implements maturation.
- `max_population: 200` -- **dead and misleading**. `max_pop` comes from `population_size`
  (`_normalize_config`).
- `brain_token_dim: 128` -- **dead**.
- `force_brain_layers: 4` -- **dead**. `n_layers` is the one read.

## Memory / culture / misc mechanisms (5)
- `memory_buffer_enabled: true` -- **dead flag on a live feature**. `update_memory_buffer` runs
  unconditionally; delete the key or wire it as a real switch.
- `culture_intensity: 0.3` -- **dead**. Culture is driven by `phase15_cumulative_culture` and the decay keys.
- `hebb_lr` -- **dead**. Nothing Hebbian exists in the code.
- `red_policy: true` -- **dead**. Red always has a PPO policy.
- `alarm_min_group: 20` -- **dead**.

## Rendering / logging / analysis (10)
- `window_size`, `steps_per_render`, `target_fps`, `dashboard_update_interval`, `log_dir`, `log_interval`
  -- **dead**, from the old renderer and logger.
- `mi_analysis_interval`, `mi_window`, `signal_cluster_k` -- **dead** in jax_sim/scripts (any MI tooling lives
  in tools/, which reads its own arguments).
- `exit_on_graduation: false` -- **dead**. Graduation is logged, and nothing exits on it.

## Stale and dangerous (1)
- `phase16_combinatorial_syntax.num_actions: 8` -- **dead and wrong**. The action space is 12
  (`n_actions`). Delete it before anyone reads it: `_normalize_config` exists precisely because an
  8-action default once built the wrong model.

**Tally (44):** should-be-wired 16 (ppo_epochs, action_entropy_coef, 9 channel, 4 puzzle, memory switch if
kept); dead 28 (including num_actions); intended-off 0 -- none of the 44 is a deliberately-off feature with a reason, so on Cam's
rule the allowlist starts empty.
