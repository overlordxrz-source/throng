"""Learnability 2 arms x 5 seeds (registered Nov 10 before data; RESEARCH_PROTOCOL.md). Per seed: eligible lives,
eligibility rate, pass fraction over eligible lives (bar >= 50%, >= 50 eligible), stratified 5 / 6 / 7-8, first reliance
by age vs median lifespan. Per arm: pooled over seeds, and seeds passing. usage: summarize.py <dir with learn_*_62b6758.json>"""
import glob, json, os, sys
import numpy as np
D = sys.argv[1]
arm_el = {}
for arm in ("nolev", "allon"):
    print(f"== arm {arm}")
    pooled_k = pooled_n = 0; seeds_pass = 0; el_rates = []
    for f in sorted(glob.glob(f"{D}/learn_{arm}_s*_62b6758.json")):
        S = json.load(open(f))["summary"]; h = S["heard25"]
        n, pf = h["eligible"], h["learner_pass_frac_eligible"]; ok = h["enough_eligible"] and pf >= 0.5
        seeds_pass += ok; pooled_k += round(pf * n) if n else 0; pooled_n += n; el_rates.append(h["eligibility_rate"])
        fr = h.get("first_reliance", {}); ba = h.get("by_age", {})
        print(f"  {os.path.basename(f)}: lives {h['lives']} eligible {n} ({h['eligibility_rate']:.0%}) pass {pf:.3f} CI {[round(x, 3) for x in h['learner_pass_ci95']]}"
              f" -> {'PASS' if ok else 'fail'} | strat {{{', '.join(f'{k}: {v['pass_frac']}' for k, v in h['stratified'].items())}}}"
              f" | by age {{{', '.join(f'{k}: n {v.get('n')} p {round(v['pass_frac'], 3) if v.get('pass_frac') is not None else None}' for k, v in ba.items())}}}"
              f" | ever {fr.get('ever_frac')} first-age median {fr.get('median_age_among_ever')} vs median lifespan {fr.get('median_lifespan')}"
              f" | grad steps/life {S['median_grad_steps_per_life']}")
    arm_el[arm] = float(np.mean(el_rates)) if el_rates else float("nan")
    print(f"  pooled: {pooled_k}/{pooled_n} = {pooled_k / max(pooled_n, 1):.3f}; seeds passing {seeds_pass}/5; mean eligibility {arm_el[arm]:.0%}")
d = abs(arm_el.get("allon", 0) - arm_el.get("nolev", 0))
print(f"\neligibility difference all-on vs no-lever: {d * 100:.1f} points -> {'REPORT STRATIFIED (> 15)' if d > 0.15 else 'within 15 points'}")
