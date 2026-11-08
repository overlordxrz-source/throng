"""North star (Cam, 2026-11-08): dictionary + "speak to them", exploratory, $0.

usage: analyze_speak.py <speak.jsonl> ...  (comm_toy.py --speak-test on saved actor params; frozen, own keys)
Dictionary from 4 frozen control rollouts: fire -> modal sender triple ("dictionary"), and each slot's modal token
("slot_dictionary"). Speaking: the listener hears our word for X = T + s instead of its sender (s = 0: the truth).
Specificity = Crafts at X / Crafts at any fire (babble makes listeners Craft everywhere, so the count alone is not
specific). Conditions: control (real sender), own_words (the sender's own triples for X), dictionary, slot_dictionary, babble
(uniform random triples). Readout: listener Crafts at X per world (s = 1..3) and deliveries (s = 0), 10 paired episodes.
"Understood" (descriptive, not a registered bar): for every s in 1..3, dictionary Crafts at X >= 0.8x own_words AND
dictionary specificity >= 0.5 AND >= babble specificity + 0.25.
"""
import json
import sys
from collections import defaultdict

import numpy as np

C = ("control", "own_words", "dictionary", "slot_dictionary", "babble")
for p in sys.argv[1:]:
    R = [json.loads(l) for l in open(p)]
    d = next(r for r in R if r["phase"] == "dictionary"); t = next(r for r in R if r["phase"] == "triple_mi")
    S = defaultdict(lambda: defaultdict(list)); A = defaultdict(lambda: defaultdict(list))
    for r in (r for r in R if r["phase"] == "speak"):
        for c in C:
            S[r["shift"]][c].append(r[f"crafts_at_X_{c}"] if r["shift"] else r[f"dep_{c}"])
            A[r["shift"]][c].append(r[f"crafts_any_{c}"])
    m = {s: {c: float(np.mean(S[s][c])) for c in C} for s in S}
    sp = {s: {c: float(np.sum(S[s][c]) / max(np.sum(A[s][c]), 1e-9)) for c in C} for s in S}
    und = lambda w: all(m[s][w] >= 0.8 * m[s]["own_words"] and sp[s][w] >= 0.5 and sp[s][w] >= sp[s]["babble"] + 0.25 for s in (1, 2, 3))  # noqa: E731
    ok, oks = und("dictionary"), und("slot_dictionary")
    print(f"{p.split('/')[-1]}: slot MI {np.round(d['mi_slot'], 3).tolist()} triple MI(MM) {t['mi_triple_mm']:.3f} ({t['distinct_triples']} "
          f"distinct triples) | modal-triple purity {np.round(d['purity'], 3).tolist()} | slot purity (mean) "
          f"{np.round(np.mean(d['slot_purity'], 1), 3).tolist()}")
    print(f"   s=0 (truth) deliveries: " + " ".join(f"{c} {m[0][c]:.2f}" for c in C))
    for s in (1, 2, 3):
        print(f"   say T+{s}: crafts at X (specificity): " + " ".join(f"{c} {m[s][c]:.2f} ({sp[s][c]:.2f})" for c in C))
    print(f"   -> understood: dictionary {'YES' if ok else 'no'} | slot dictionary {'YES' if oks else 'no'}")
