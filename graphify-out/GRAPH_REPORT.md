# Graph Report - .  (2026-05-31)

## Corpus Check
- 124 files · ~139,814 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 914 nodes · 1568 edges · 78 communities (61 shown, 17 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 120 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Toroidal Grid & Resources|Toroidal Grid & Resources]]
- [[_COMMUNITY_PyTorch Transformer Brain|PyTorch Transformer Brain]]
- [[_COMMUNITY_Genome & Parameter Ops|Genome & Parameter Ops]]
- [[_COMMUNITY_JAX Network & Cross-Attn|JAX Network & Cross-Attn]]
- [[_COMMUNITY_JAX Training Loop|JAX Training Loop]]
- [[_COMMUNITY_Grid Physics & Moves|Grid Physics & Moves]]
- [[_COMMUNITY_Decode & Vocabulary Tests|Decode & Vocabulary Tests]]
- [[_COMMUNITY_MAPPO Rollout Buffer|MAPPO Rollout Buffer]]
- [[_COMMUNITY_JAX Grid & Neighbors|JAX Grid & Neighbors]]
- [[_COMMUNITY_Corpus & MI Analysis|Corpus & MI Analysis]]
- [[_COMMUNITY_Dashboard Rendering|Dashboard Rendering]]
- [[_COMMUNITY_Run Logging|Run Logging]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]

## God Nodes (most connected - your core abstractions)
1. `ToroidalGrid` - 44 edges
2. `run()` - 38 edges
3. `TorchBrain` - 31 edges
4. `_run_simulation_impl()` - 31 edges
5. `PopulationState` - 27 edges
6. `RunLogger` - 22 edges
7. `SignalCorpusWriter` - 22 edges
8. `ndarray` - 21 edges
9. `PopState` - 20 edges
10. `Renderer` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Decode Pipeline` --semantically_similar_to--> `topographic_similarity()`  [INFERRED] [semantically similar]
  tools/decode_signals.py → communication/analysis.py
- `Decode Pipeline` --semantically_similar_to--> `CommunicationAnalyser`  [INFERRED] [semantically similar]
  tools/decode_signals.py → communication/analysis.py
- `int` --uses--> `PopulationState`  [INFERRED]
  evolution/selection.py → agents/population.py
- `KeyArray` --uses--> `PopulationState`  [INFERRED]
  evolution/selection.py → agents/population.py
- `ndarray` --uses--> `PopulationState`  [INFERRED]
  evolution/selection.py → agents/population.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **JAX PPO Training Pipeline** — jax_sim_main_jax_lax_scan_rollout, jax_sim_main_jax_rollout_to_cpu, jax_sim_rl_jax_ppo_flow, jax_sim_rl_jax_auxiliary_update, jax_sim_main_jax_vq_dead_code_reset [EXTRACTED 1.00]
- **Phase 9/11 Auxiliary Learning Stack** — jax_sim_network_jax_auxiliary_heads, jax_sim_rl_jax_auxiliary_update, jax_sim_main_jax_phase9_auxiliary, jax_sim_network_jax_carry_fusion [EXTRACTED 1.00]
- **Modal GPU Deployment Stack** — run_bg_background_training, scripts_modal_train_p10_5, scripts_modal_train_gpu_env, jax_sim_main_jax_local_jax_cache, jax_sim_main_jax_orbax_checkpoint [INFERRED 0.85]
- **Phase 9 Canvas Unlock Sequence** — throng_neighbor_cross_attention, config_phase7_cross_attn_enabled, docs_phase9_canvas_graft_nb_cross_attn, throng_checkpoint_390, throng_confidence_head_9_1 [EXTRACTED 1.00]
- **Corpus Decode Pass Criteria (P10.6+)** — throng_decode_signals_tool, throng_lag1_omnibus_decode, throng_vq_token_direction_test, config_phase7_alarm_scout_range [EXTRACTED 1.00]
- **Phase 11 Imagination Failure Modes** — throng_phase_11_2_concluded, throng_stay_collapse_failure, throng_solipsistic_world_model, throng_checkpoint_393_forbidden, throng_phase_9_canvas_active [EXTRACTED 1.00]
- **MAPPO Training Stack** — main_run, main_mappo, main_rolloutbuffer, agents_network_torch_torchbrain, agents_rl_torch_ppo_update_torch, agents_rl_torch_compute_gae [EXTRACTED 0.95]
- **Signal Science Pipeline** — agents_network_torch_vq_token_bottleneck, communication_analysis_signcorpuswriter, communication_analysis_signal_corpus, tools_decode_signals_decode_pipeline, tools_decode_signals_vq_token_direction_test [INFERRED 0.85]
- **Online vs Offline MI Analysis** — communication_analysis_communicationanalyser, communication_analysis_misnapshot, tools_decode_signals_decode_pipeline, tools_decode_signals_load_corpus [INFERRED 0.75]

## Communities (78 total, 17 thin omitted)

### Community 0 - "Toroidal Grid & Resources"
Cohesion: 0.06
Nodes (29): float, int, ndarray, environment/grid.py — Toroidal grid with continuous float symbol culture layer,, Return (max_pop, W*2) normalised presence — [blue_density, red_density]., Return (N,) bool — True where position is a wall., Procedural wall generation: cellular automata cave generation.         Produces, Return (max_pop, W) bool — wall presence in local window. (+21 more)

### Community 1 - "PyTorch Transformer Brain"
Cohesion: 0.06
Nodes (44): AgentNetworkTorch, _AttentionBlock, _best_device(), layer_entropy_torch(), bool, device, float, int (+36 more)

### Community 2 - "Genome & Parameter Ops"
Cohesion: 0.07
Nodes (42): copy_agent_params(), crossover_params(), flatten_params(), get_agent_params(), mutate_params(), float, int, KeyArray (+34 more)

### Community 3 - "JAX Network & Cross-Attn"
Cohesion: 0.07
Nodes (36): Debug NaN issue in JAX model., AgentNetworkJax, dead_code_reset_codebook_params(), ensure_aux_head_params(), graft_missing_param_subtrees(), init_agent_params(), _is_nested_param_dict(), NeighborCrossAttention (+28 more)

### Community 4 - "JAX Training Loop"
Cohesion: 0.07
Nodes (39): Debug script to check PPO metrics on Kaggle., Contested Resource Cooperation, Grid Ecology Layers, Brain Vote (Capacity-Based Layer Growth), Evolutionary Distillation Outer Loop, JAX Simulation Loop (Init→Rollout→PPO→Repeat), lax.scan JIT Rollout Kernel, Phase 9 Auxiliary Loss Pipeline (+31 more)

### Community 5 - "Grid Physics & Moves"
Cohesion: 0.11
Nodes (35): apply_catches(), apply_moves(), chebyshev_dist(), check_puzzle_solved(), consume_resources(), decay_grid(), decay_puzzle_timeout(), generate_contested_nodes() (+27 more)

### Community 6 - "Decode & Vocabulary Tests"
Cohesion: 0.12
Nodes (33): 145k Scaffolding Withdrawal Test, _cardinal_bin(), _cardinal_dist(), categorical_vocabulary_tests(), cluster_analysis(), Decode Pipeline, dim_correlations(), dim_mi() (+25 more)

### Community 7 - "MAPPO Rollout Buffer"
Cohesion: 0.14
Nodes (27): compute_obs_dim_torch(), Collects T steps × N agents of (obs, action, log_prob, value, reward, done, aliv, RolloutBuffer, bool, int, ndarray, PopulationState, str (+19 more)

### Community 8 - "JAX Grid & Neighbors"
Cohesion: 0.13
Nodes (20): AgentNetworkJax, GridState, get_local_patches(), get_neighbour_signals(), GridState, All environment layers packed into a pytree-friendly dict., Extract (max_pop, W, [D]) local patches via toroidal indexing.     W = (2*radius, For each agent, find k nearest alive neighbours and return their signals.     Re (+12 more)

### Community 9 - "Corpus & MI Analysis"
Cohesion: 0.12
Nodes (18): CommunicationAnalyser, granger_causality_lags(), MISnapshot, float, int, ndarray, Queue, str (+10 more)

### Community 10 - "Dashboard Rendering"
Cohesion: 0.17
Nodes (20): _ax_style(), _dashboard_process_main(), DashboardUpdate, _draw_fitness(), _draw_lineage(), _draw_mi_heatmap(), _draw_population(), _legend() (+12 more)

### Community 11 - "Run Logging"
Cohesion: 0.19
Nodes (9): bool, float, int, Path, str, utils/logging.py — Structured JSON run logging.  All log records are written as, Log an MISnapshot.  mi_matrix is serialised as a nested list so it         can b, Thread-safe JSONL logger for one simulation run.      Creates a new run director (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (12): LineageRecord, MetricsTracker, int, ndarray, evolution/metrics.py — Fitness tracking and lineage tree management.  Lineage tr, Return top-n lineages by longevity (last_seen - birth_step).         Includes bo, Return the lineage_id currently with the most living members.         Returns No, Append a mutual-information snapshot. (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (17): apply_auto_reproduce(), apply_mind_meld(), init_population(), kill_agents(), Array, float, int, ndarray (+9 more)

### Community 14 - "Community 14"
Cohesion: 0.12
Nodes (4): get_current_branch(), get_feature_paths(), has_git(), common.sh script

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (7): bool, int, ndarray, visualization/renderer.py — Beautiful dark-space renderer for THRONG v2.  Visual, Glowing circles — radius scales with brain depth (n_layers)., Symbol presence → violet/indigo glow; base is deep space., Renderer

### Community 16 - "Community 16"
Cohesion: 0.24
Nodes (17): create_population(), distill_population(), expand_brain(), _find_free_slot(), inject_offspring(), inject_random_agent(), kill_agent(), PopulationState (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (16): away_delta(), _away_row(), crowding_split(), in_quadrant(), load(), main(), bool, float (+8 more)

### Community 18 - "Community 18"
Cohesion: 0.18
Nodes (11): compute_gae(), _get_loss_fn(), ppo_update_step(), bool, float, GradientTransformation, int, ndarray (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.15
Nodes (9): float, ndarray, resource.py — Resource patch generation and regeneration logic.  Resources are d, Manages resource cluster positions and per-step regeneration., Place cluster centres randomly, generate the initial resource grid,         and, Apply one step of resource regeneration in-place.         Adds regen_map to the, Shift each cluster centre by a random walk step.         This prevents agents fr, Precompute the (grid_size, grid_size) float32 regen contribution map.          W (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.17
Nodes (15): compute_fwd_env_dim(), loc_env_flat_bounds(), Start/end indices of flattened loc_env in the observation vector., Flat loc_env size (W × 8 env channels)., _normalize_config(), int, Map PyTorch config names to JAX config names., Move scan outputs off GPU so PPO backward has room (obs alone ~2.4GB). (+7 more)

### Community 21 - "Community 21"
Cohesion: 0.13
Nodes (16): phase9_canvas.cross_attn_enabled, Modal Cell 1 Clone feature/phase9-canvas, Phase 9.4 Cross-Attention Receiver, graft_missing_param_subtrees nb_cross_attn, Dedicated Other Pathway, Dreamer Imagination Loop (9.3), README Phase 9 Roadmap (9.1–9.4), Checkpoint 390 (200k Metrics-Only) (+8 more)

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (14): files, .specify/scripts/bash/check-prerequisites.sh, .specify/scripts/bash/common.sh, .specify/scripts/bash/create-new-feature.sh, .specify/scripts/bash/setup-plan.sh, .specify/scripts/bash/setup-tasks.sh, .specify/templates/checklist-template.md, .specify/templates/constitution-template.md (+6 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (15): Shelter Catch Protection, Dual-Team Blue/Red PPO Training, Local JAX Compilation Cache (Modal), Orbax Checkpoint Resume, Phase 9.4 Cross-Attention Receiver, Predator Jitter (catch_prob), Red Population Curriculum, Signal Corpus Recording (+7 more)

### Community 24 - "Community 24"
Cohesion: 0.14
Nodes (13): files, .cursor/skills/speckit-analyze/SKILL.md, .cursor/skills/speckit-checklist/SKILL.md, .cursor/skills/speckit-clarify/SKILL.md, .cursor/skills/speckit-constitution/SKILL.md, .cursor/skills/speckit-implement/SKILL.md, .cursor/skills/speckit-plan/SKILL.md, .cursor/skills/speckit-specify/SKILL.md (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.19
Nodes (13): _dict_to_pop(), find_latest_checkpoint(), load_checkpoint(), _pop_to_dict(), int, ndarray, Path, str (+5 more)

### Community 26 - "Community 26"
Cohesion: 0.23
Nodes (12): aggregate_neighbour_signals(), compute_signal_similarity_pairs(), get_neighbour_indices_padded(), get_neighbour_signals_padded(), float, int, ndarray, communication/channel.py — Neighbour signal aggregation.  Each agent broadcasts (+4 more)

### Community 27 - "Community 27"
Cohesion: 0.21
Nodes (9): _assert_corpus_writer(), _assert_observations_module(), _evict_stale_modules(), Any, int, Import run_simulation from here after `git pull` (no kernel restart needed).  Cl, Run JAX training with a fresh import of jax_sim + communication from disk., run_simulation() (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.17
Nodes (12): alarm_scout_range: 8, red_detection_radius: 0 (Blind Blues), corpus_every_n_steps: 4, P11 Staging Merge Gate (≥100k decode), Cardinal Lexicon Decode (χ² Pass), decode_signals.py, Lag-1 Omnibus Direction LRT, Modal Volume throng-runs (dragonbgnx) (+4 more)

### Community 29 - "Community 29"
Cohesion: 0.24
Nodes (11): find_latest_run(), get_main_py_pids(), get_process_info(), main(), Find PIDs of running main.py processes., Get basic info about a process without psutil., Find the most recent run directory., Show last N lines of science.log with key metrics. (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.20
Nodes (3): _extract_highest_number(), get_highest_from_branches(), create-new-feature.sh script

### Community 31 - "Community 31"
Cohesion: 0.20
Nodes (9): invoke_separator, script, default_integration, installed_integrations, integration, integration_settings, cursor-agent, integration_state_schema (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (9): schema_version, description, installed_at, name, source, updated_at, version, workflows (+1 more)

### Community 33 - "Community 33"
Cohesion: 0.31
Nodes (9): dissociation_table(), load(), main(), int, ndarray, str, tools/es_probe.py — Scout/blind dissociation probe for E/S-encoding dims.  For a, Spearman r, returning (r, p). NaN-safe: drops rows where either is nan. (+1 more)

### Community 34 - "Community 34"
Cohesion: 0.39
Nodes (7): ConvertTo-CleanBranchName(), Get-BranchName(), Get-HighestNumberFromBranches(), Get-HighestNumberFromNames(), Get-HighestNumberFromRemoteRefs(), Get-HighestNumberFromSpecs(), Get-NextBranchNumber()

### Community 35 - "Community 35"
Cohesion: 0.22
Nodes (8): ai, ai_skills, branch_numbering, context_file, here, integration, script, speckit_version

### Community 36 - "Community 36"
Cohesion: 0.25
Nodes (3): _extract_highest_number(), get_highest_from_branches(), create-new-feature.sh script

### Community 37 - "Community 37"
Cohesion: 0.33
Nodes (8): bearing_histogram(), load(), main(), print_histogram(), int, ndarray, str, tools/bearing_analysis.py — West-cluster bearing analysis.  Tests Cam's predicti

### Community 38 - "Community 38"
Cohesion: 0.47
Nodes (3): Periodically samples alive blue agents and writes one JSONL record per     sampl, Flush Python buffer and fsync — call after each PPO rollout on Modal., SignalCorpusWriter

### Community 40 - "Community 40"
Cohesion: 0.40
Nodes (5): carry_fwd_coef: 0.05, head_fwd_dyn Carry Prediction, stop_gradient(carry_{t+1}) Target, Forward Dynamics Head (9.2), Phase 11.0 Carry Forward Dynamics

### Community 42 - "Community 42"
Cohesion: 0.40
Nodes (5): Bottom-Up Emergence Bet, THRONG Project Overview, Cam / Will / User Triad, Emergent Proto-Language Hypothesis, No Comm Reward Shaping

### Community 43 - "Community 43"
Cohesion: 0.50
Nodes (3): Force-reload jax_sim after git pull (use when the kernel cached old modules)., Drop cached jax_sim.* modules and re-import main_jax from disk., reload_throng_jax()

### Community 45 - "Community 45"
Cohesion: 0.50
Nodes (4): distill_enabled: false, P10.5 Hard-Ceiling Population 150–200, Lotka-Volterra Population Oscillator, NB_GAIN↔surv nan at Ceiling

### Community 49 - "Community 49"
Cohesion: 0.67
Nodes (3): B200 Rollout OOM (Fragmentation), CPU Rollout Offload PPO (H2D), Phase 11.1 GPU Rollouts (Abandoned)

## Ambiguous Edges - Review These
- `Phase 9.4 Cross-Attention Receiver` → `Red Sense API v2`  [AMBIGUOUS]
  jax_sim/main_jax.py · relation: conceptually_related_to
- `run()` → `SignalCorpusWriter`  [AMBIGUOUS]
  main.py · relation: conceptually_related_to

## Knowledge Gaps
- **118 isolated node(s):** `bool`, `ndarray`, `int`, `str`, `bool` (+113 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Phase 9.4 Cross-Attention Receiver` and `Red Sense API v2`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `run()` and `SignalCorpusWriter`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `SignalCorpusWriter` connect `Community 38` to `MAPPO Rollout Buffer`, `JAX Grid & Neighbors`, `Corpus & MI Analysis`, `Community 20`, `Community 27`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `ToroidalGrid` connect `Toroidal Grid & Resources` to `Genome & Parameter Ops`, `MAPPO Rollout Buffer`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `run()` connect `MAPPO Rollout Buffer` to `Toroidal Grid & Resources`, `PyTorch Transformer Brain`, `Genome & Parameter Ops`, `Community 38`, `Corpus & MI Analysis`, `Run Logging`, `Community 15`, `Community 16`, `Community 25`, `Community 26`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `ToroidalGrid` (e.g. with `bool` and `int`) actually correct?**
  _`ToroidalGrid` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `TorchBrain` (e.g. with `float` and `int`) actually correct?**
  _`TorchBrain` has 13 INFERRED edges - model-reasoned connections that need verification._