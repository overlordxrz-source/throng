"""Cam Nov 12 (ruling on the ~22:50 report), registered BEFORE the 3e8 readout: CONTINUE s1kl03 to 1B if ANY of
(a) hungry life test, memory kept: pass > 0 AND its Wilson 95% lower bound > the memory-wiped arm's pass rate;
(b) token influence at 3e8 >= 3x its value at 1e8;
(c) the probe's memory-minus-wiped sector advantage (acc_all) is >= 3 points larger at 3e8 than at 1e8.
Otherwise STOP after the 3e8 readout.
Operationalisation (Will, registered here): token influence "at" a milestone = mean of the last 5 logged values
(every 100 updates) at or before the update where that milestone's readout fired. Readouts are matched by order
(1st = 1e8, 2nd = 3e8). Prints one JSON line with the verdict.
usage: python decide_3e8.py s1kl03_a595b35.json"""
import json, sys

d = json.load(open(sys.argv[1])); S = d["summary"]; R = S["readouts"]; L = d["log"]
assert len(R) >= 2, "the 3e8 readout has not landed"
r1, r3 = R[0], R[1]
T = S.get("T", 64); W = S.get("worlds", 64); spu = W * T * 4          # agent-steps per update (4 learners per world)


def ti_at(steps):
    u = int(steps // spu)
    v = [x["token_influence"]["kl"] for x in L if x.get("token_influence") and x["update"] <= u]
    return sum(v[-5:]) / len(v[-5:]) if v else None


h, hw = r3["tests"]["hungry_memory_carried"], r3["tests"]["hungry_memory_wiped"]
ca = bool(h["pass_frac"] and h["pass_frac"] > 0 and h["ci95"][0] > (hw["pass_frac"] or 0.0))
t1, t3 = ti_at(r1["at_agent_steps"]), ti_at(r3["at_agent_steps"])
cb = bool(t1 is not None and t3 is not None and t1 > 0 and t3 >= 3 * t1)
adv = lambda r: r["linear_probe"]["sector_memory"]["acc_all"] - r["linear_probe"]["sector_wiped"]["acc_all"]  # noqa: E731
a1, a3 = adv(r1), adv(r3); cc = bool(a3 - a1 >= 0.03)
print(json.dumps(dict(verdict="CONTINUE" if (ca or cb or cc) else "STOP",
                      a=dict(hold=ca, hungry_kept=h["pass_frac"], ci95=h["ci95"], hungry_wiped=hw["pass_frac"]),
                      b=dict(hold=cb, ti_1e8=t1, ti_3e8=t3, ratio=(t3 / t1 if t1 else None)),
                      c=dict(hold=cc, adv_1e8=a1, adv_3e8=a3, diff=a3 - a1),
                      forage=[r.get("forage", {}).get("decision") for r in R])))
