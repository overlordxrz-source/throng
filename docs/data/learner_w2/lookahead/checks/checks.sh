#!/bin/zsh
# after the label count: p = 0 bit-identity (2 checks), then the two lookahead gate controls (Cam Nov 16 ~01:00)
until grep -q CDONE /private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/labels/count.log; do sleep 30; done
cd /Users/overlord/CascadeProjects/throng-toy; PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; C=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/checks
G="--oracle --hunger-gate --item-size 0.1 --W 32 --steps 1.5e7 --policy follower --v2 /Users/overlord/throng-backup/evo2/evo2.ckpt --gate honest --curve-every 100"
$PY w2/port_w2.py ${=G} --out $C/id_port_la0.json > $C/id_port_la0.log 2>&1
X="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --steps 3.3e5 --curve-every 5 --seed 9"
$PY w2/learner_w2_kv.py ${=X} --out $C/id_kv.json > $C/id_kv.log 2>&1
$PY w2/learner_w2_la.py ${=X} --out $C/id_la0.json > $C/id_la0.log 2>&1
$PY w2/port_w2.py ${=G} --lookahead 0.25 --out $C/ctl_pos_la.json > $C/ctl_pos_la.log 2>&1
$PY w2/port_w2.py ${=G} --lookahead 0.25 --giveup 0.3 --out $C/ctl_neg_la.json > $C/ctl_neg_la.log 2>&1
/Users/overlord/CascadeProjects/throng/.venv/bin/python - <<'PY'
import json
C="/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/checks"; strip=lambda c:[{k:v for k,v in d.items() if k not in ("sec","n_hungry_lookahead")} for d in c]
old=json.load(open("/Users/overlord/CascadeProjects/throng/docs/data/port_w2/honest_gate/ctl_gu0.json")); new=json.load(open(C+"/id_port_la0.json"))
i1 = strip(old["curve"])==strip(new["curve"]) and old["gate_scores"]==new["gate_scores"]
a=json.load(open(C+"/id_kv.json")); b=json.load(open(C+"/id_la0.json")); i2 = strip(a["curve"])==strip(b["curve"]) and a["gate_scores"]==b["gate_scores"] and len(a["curve"])>0
pos=json.load(open(C+"/ctl_pos_la.json")); neg=json.load(open(C+"/ctl_neg_la.json"))
la_share=lambda d: sum(c["n_hungry_lookahead"] for c in d["curve"])/max(sum(c["n_hungry"] for c in d["curve"]),1)
r=dict(identity_port_vs_committed_ctl_gu0=i1, n_curve_port=len(new["curve"]), identity_la0_vs_kv=i2, n_curve_kv=len(a["curve"]),
       pos_frontier=pos["frontier"], pos_windows=len(pos["gate_scores"]), pos_min_window=min(pos["gate_scores"]) if pos["gate_scores"] else None, pos_la_share_logged=la_share(pos),
       neg_frontier=neg["frontier"], neg_windows=len(neg["gate_scores"]), neg_mean=sum(neg["gate_scores"])/max(len(neg["gate_scores"]),1), neg_max=max(neg["gate_scores"]) if neg["gate_scores"] else None, neg_la_share_logged=la_share(neg))
r["controls_pass"] = bool(r["pos_frontier"]>=24 and r["neg_frontier"]==1)
r["all_pass"] = bool(i1 and i2 and r["controls_pass"])
json.dump(r, open(C+"/summary.json","w"), indent=1); print(json.dumps(r))
if r["all_pass"]: open("/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad/learner_la/controls_pass","w").write("ok\n")
PY
echo CHKDONE
