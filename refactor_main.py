import re

with open("main.py", "r") as f:
    content = f.read()

# 1. Imports
content = re.sub(
    r"from evolution\.selection\s+import run_evolution_step, compute_fitness\n",
    "", content
)

# 2. Initializations
init_target = r"    metrics  = MetricsTracker\(\)\n    analyser = CommunicationAnalyser\("
init_replacement = """    # ── RL Setup ─────────────────────────────────────────────────────────────
    blue_opt = optax.adam(3e-4)
    blue_opt_state = blue_opt.init(blue_pop.params)
    ppo_update_blue = make_ppo_update_fn(forward_fn, blue_opt)
    blue_buffer = RolloutBuffer(blue_pop.max_pop)

    red_opt = optax.adam(3e-4)
    red_opt_state = None
    if red_pop is not None:
        red_opt_state = red_opt.init(red_pop.params)
    ppo_update_red = make_ppo_update_fn(forward_fn, red_opt)
    red_buffer = RolloutBuffer(config.get("red_population_size", 40))

    metrics  = MetricsTracker()
    analyser = CommunicationAnalyser("""

content = content.replace(init_target, init_replacement)

# 3. Inside step loop (lines 313 onwards)
# Let's replace the whole block from "3. Blue forward pass" down to "9. Periodic: evolution"
# Since it's large, I'll use a regex that matches the start and end.

start_marker = "                # 2. Blue forward pass"
end_marker = "                # 10. Periodic: MI analysis"

match = re.search(re.escape(start_marker) + r".*?" + re.escape(end_marker), content, re.DOTALL)
if match:
    old_block = match.group(0)
    
    new_block = """                # 2. Blue forward pass
                prng_key, b_prng_key, r_prng_key = jax.random.split(prng_key, 3)
                
                b_sym   = grid.get_local_symbols(blue_pop.positions, radius=obs_radius)
                b_pres  = grid.get_local_presence(blue_pop.positions, blue_map, red_map, radius=obs_radius)
                b_neigh = get_neighbour_signals_padded(
                    blue_pop.positions, blue_pop.signals, blue_pop.alive,
                    k=K, grid_size=gs,
                )
                b_own = build_own_state(blue_pop, config)

                b_own_jnp = jnp.array(b_own)
                b_neigh_jnp = jnp.array(b_neigh)
                b_sym_jnp = jnp.array(b_sym)
                b_pres_jnp = jnp.array(b_pres)
                b_sig_jnp = jnp.array(blue_pop.signals[:, None])
                b_nl = jnp.array(blue_pop.n_layers.astype(np.int32))

                b_new_c, b_logits_tuple = forward_fn(
                    blue_pop.params, blue_pop.carries, b_nl, b_own_jnp, b_neigh_jnp, b_sym_jnp, b_pres_jnp, b_sig_jnp
                )
                b_acts, b_log_prob = sample_actions_and_log_probs(b_prng_key, b_logits_tuple)
                b_action, b_sigs_act, b_sym_w_act = [np.array(a) for a in b_acts]
                b_value = np.array(b_logits_tuple[3])
                
                b_obs_cache = (
                    b_own, b_neigh, b_sym, b_pres, blue_pop.signals[:, None].copy(), blue_pop.n_layers.copy(),
                    np.array(blue_pop.carries), b_action, b_sigs_act, b_sym_w_act, np.array(b_log_prob), b_value, blue_pop.alive.copy()
                )

                blue_pop.carries = b_new_c
                b_sigs = np.array(b_sigs_act, dtype=np.int32)
                b_sigs[~blue_pop.alive] = 0
                blue_pop.signals = b_sigs

                b_actions = b_action.copy()
                b_actions[~blue_pop.alive] = A_STAY
                apply_moves(blue_pop, b_actions, gs)

                b_sym_w = np.array(b_sym_w_act, dtype=np.int32)
                grid.write_symbols(
                    blue_pop.positions, b_sym_w, blue_pop.alive
                )
                blue_pop.ages[blue_pop.alive] += 1
                blue_pop.alive[blue_pop.alive & (blue_pop.ages >= config["max_age"])] = False

                # 3. Red forward pass
                r_obs_cache = None
                if red_pop is not None and red_pop.alive.any():
                    bm2, rm2 = grid.build_presence_maps(
                        blue_pop.positions, blue_pop.alive, blue_pop.team
                    )
                    _, rm2r = grid.build_presence_maps(
                        red_pop.positions, red_pop.alive, red_pop.team
                    )
                    rm2 = rm2 + rm2r

                    r_sym   = grid.get_local_symbols(red_pop.positions, radius=obs_radius)
                    r_pres  = grid.get_local_presence(red_pop.positions, bm2, rm2, radius=obs_radius)
                    r_neigh = get_neighbour_signals_padded(
                        red_pop.positions, red_pop.signals, red_pop.alive,
                        k=K, grid_size=gs,
                    )
                    r_own = build_own_state(red_pop, config)

                    r_own_jnp = jnp.array(r_own)
                    r_neigh_jnp = jnp.array(r_neigh)
                    r_sym_jnp = jnp.array(r_sym)
                    r_pres_jnp = jnp.array(r_pres)
                    r_sig_jnp = jnp.array(red_pop.signals[:, None])
                    r_nl = jnp.array(red_pop.n_layers.astype(np.int32))

                    r_new_c, r_logits_tuple = forward_fn(
                        red_pop.params, red_pop.carries, r_nl, r_own_jnp, r_neigh_jnp, r_sym_jnp, r_pres_jnp, r_sig_jnp
                    )
                    r_acts, r_log_prob = sample_actions_and_log_probs(r_prng_key, r_logits_tuple)
                    r_action, r_sigs_act, r_sym_w_act = [np.array(a) for a in r_acts]
                    r_value = np.array(r_logits_tuple[3])
                    
                    r_obs_cache = (
                        r_own, r_neigh, r_sym, r_pres, red_pop.signals[:, None].copy(), red_pop.n_layers.copy(),
                        np.array(red_pop.carries), r_action, r_sigs_act, r_sym_w_act, np.array(r_log_prob), r_value, red_pop.alive.copy()
                    )

                    red_pop.carries = r_new_c
                    r_sigs = np.array(r_sigs_act, dtype=np.int32)
                    r_sigs[~red_pop.alive] = 0
                    red_pop.signals = r_sigs

                    r_actions = r_action.copy()
                    r_actions[~red_pop.alive] = A_STAY
                    apply_moves(red_pop, r_actions, gs)

                    r_sym_w = np.array(r_sym_w_act, dtype=np.int32)
                    grid.write_symbols(
                        red_pop.positions, r_sym_w, red_pop.alive
                    )
                    red_pop.ages[red_pop.alive] += 1
                    red_pop.alive[red_pop.alive & (red_pop.ages >= config["max_age"])] = False

                # 4. Catch detection
                if red_pop is not None:
                    caught_b, catching_r = apply_catches(blue_pop, red_pop, config)
                else:
                    caught_b = np.zeros(blue_pop.max_pop, dtype=bool)
                    catching_r = np.zeros(0, dtype=bool)
                
                # 5. RL Trajectory Collection & Rewards
                b_rewards = np.zeros(blue_pop.max_pop, dtype=np.float32)
                b_rewards[b_obs_cache[-1]] = 0.01
                b_rewards[caught_b] = -1.0
                blue_buffer.add(*b_obs_cache[:-1], b_rewards, b_obs_cache[-1])
                
                if r_obs_cache is not None:
                    r_rewards = np.zeros(red_pop.max_pop, dtype=np.float32)
                    r_rewards[r_obs_cache[-1]] = -0.01
                    r_rewards[catching_r] = 1.0
                    red_buffer.add(*r_obs_cache[:-1], r_rewards, r_obs_cache[-1])

                # 6. Auto-reproduction
                blue_pop, prng_key = apply_auto_reproduce(
                    blue_pop, config, prng_key, model, team_id=0
                )
                if red_pop is not None:
                    red_pop, prng_key = apply_auto_reproduce(
                        red_pop, config, prng_key, model, team_id=1
                    )

                # 7. Symbol grid decay
                grid.decay_symbols(config.get("culture_fade_steps", 100))

                # 8. Population floors
                blue_pop, prng_key = enforce_population_floor(
                    blue_pop, config, prng_key, model, team_id=0
                )
                if red_pop is not None:
                    red_pop, prng_key = enforce_population_floor(
                        red_pop, config, prng_key, model, team_id=1
                    )

                # 9. Spawn reds at scheduled step
                if red_pop is None and step >= config["red_spawn_step"]:
                    prng_key, rk = jax.random.split(prng_key)
                    red_pop = create_population(config, model, rk, gs, team_id=1)
                    red_opt_state = red_opt.init(red_pop.params)
                    slog(f"[step {step:,}] *** RED PREDATORS SPAWNED — arms race begins! ***")

                # 9b. Periodic: PPO Update
                if step % config.get("rl_update_interval", 500) == 0:
                    # Get next value for GAE
                    _, b_logits_t = forward_fn(
                        blue_pop.params, blue_pop.carries, jnp.array(blue_pop.n_layers.astype(np.int32)),
                        jnp.array(build_own_state(blue_pop, config)),
                        jnp.array(get_neighbour_signals_padded(blue_pop.positions, blue_pop.signals, blue_pop.alive, k=K, grid_size=gs)),
                        jnp.array(grid.get_local_symbols(blue_pop.positions, radius=obs_radius)),
                        jnp.array(grid.get_local_presence(blue_pop.positions, blue_map, red_map, radius=obs_radius)),
                        jnp.array(blue_pop.signals[:, None])
                    )
                    b_next_val = np.array(b_logits_t[3])[:, 0]
                    
                    b_batch = blue_buffer.finalize(b_next_val)
                    if len(b_batch['own']) > 0:
                        for epoch in range(config.get("ppo_epochs", 4)):
                            blue_pop.params, blue_opt_state, b_aux = ppo_update_blue(blue_pop.params, blue_opt_state, b_batch)
                        logger.log_evo_event({"evo_step": evo_count, "pg_loss": float(b_aux[0]), "v_loss": float(b_aux[1]), "ent": float(b_aux[2])})
                    
                    if red_pop is not None:
                        _, r_logits_t = forward_fn(
                            red_pop.params, red_pop.carries, jnp.array(red_pop.n_layers.astype(np.int32)),
                            jnp.array(build_own_state(red_pop, config)),
                            jnp.array(get_neighbour_signals_padded(red_pop.positions, red_pop.signals, red_pop.alive, k=K, grid_size=gs)),
                            jnp.array(grid.get_local_symbols(red_pop.positions, radius=obs_radius)),
                            jnp.array(grid.get_local_presence(red_pop.positions, blue_map, red_map, radius=obs_radius)),
                            jnp.array(red_pop.signals[:, None])
                        )
                        r_next_val = np.array(r_logits_t[3])[:, 0]
                        r_batch = red_buffer.finalize(r_next_val)
                        if len(r_batch['own']) > 0:
                            for epoch in range(config.get("ppo_epochs", 4)):
                                red_pop.params, red_opt_state, r_aux = ppo_update_red(red_pop.params, red_opt_state, r_batch)
                    
                    evo_count += 1

                # 10. Periodic: MI analysis"""
    content = content.replace(old_block, new_block)
else:
    print("Failed to match block!")

with open("main.py", "w") as f:
    f.write(content)
