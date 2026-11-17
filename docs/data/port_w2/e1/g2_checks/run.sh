#!/bin/zsh
# G2 + merged-learner checks on the merged code (worktree m8wt = moves8-learner: comm-toy + learner 8-way + rl2 target-KL +
# lr-anneal). Run AFTER G1 (one training job at a time). Compared by check.py.
SP=/private/tmp/claude-501/-Users-overlord-CascadeProjects-throng/aad77c1d-d7ac-437e-bef4-aa525802ccc0/scratchpad; I=$SP/g2chk2; mkdir -p $I   # rerun 2 after the upd_kl fix (first run kept in g2chk)
PY=/Users/overlord/CascadeProjects/throng/.venv/bin/python; cd $SP/m8wt
B="--oracle --hunger-gate --item-size 0.1 --W 32 --gate honest --no-nav --moves8 --ppo-log"
AN="--target-kl 0.015 --lr-anneal --lr-anneal-frontier 2 --lr-anneal-after 81920 --lr-anneal-len 163840"
PAN="--target-kl 0.015 --lr-anneal --lr-anneal-frontier 2 --lr-anneal-after 81920 --lr-anneal-len 81920"   # rerun 2: len 81920 so the port check reaches lr_end inside 60 iterations
$PY w2/test_lr_anneal.py > $I/unit.txt 2>&1; echo "unit $(tail -1 $I/unit.txt)"
$PY w2/port_w2.py ${=B} --steps 491520 --out $I/p_off.json > $I/p_off.log 2>&1; echo "p_off $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} --target-kl 0.015 --steps 491520 --out $I/p_tkl.json > $I/p_tkl.log 2>&1; echo "p_tkl $(date '+%H:%M:%S')"
$PY w2/port_w2.py ${=B} ${=PAN} --steps 491520 --out $I/p_anA.json > $I/p_anA.log 2>&1; echo "p_anA $(date '+%H:%M:%S')"
for i in $(seq 1 10); do $PY w2/port_w2.py ${=B} ${=PAN} --steps 491520 --ckpt $I/p_anB.ckpt --max-seconds 1 --out $I/p_anB.json >> $I/p_anB.log 2>&1; [ -f $I/p_anB.json ] && break; done; echo "p_anB $(date '+%H:%M:%S')"
cp $SP/e1/e1s1_ckpt_5e7.ckpt $I/old5e7.ckpt
$PY w2/port_w2.py ${=B} --target-kl 0.015 --lr-anneal --seed 1 --steps 50e6 --ckpt $I/old5e7.ckpt --out $I/p_old.json > $I/p_old.log 2>&1; echo "p_old $(date '+%H:%M:%S')"
XT="--learner ep --slots 96 --shared-tok --hindsight 0.5 --head-own-meal --demonstrator --parent-waits --gate honest --hunger-gate --item-size 0.1 --W 32 --lookahead 0.25 --la-max 12 --head-relabel-token"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --seed 7 --steps 491520 --out $I/l_off.json > $I/l_off.log 2>&1; echo "l_off $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --seed 8 --steps 163840 --out $I/l_on.json > $I/l_on.log 2>&1; echo "l_on $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --target-kl 1e9 --seed 8 --steps 163840 --out $I/l_big.json > $I/l_big.log 2>&1; echo "l_big $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --target-kl 0.015 --tkl-synthetic-shift 1.0 --seed 8 --steps 40960 --out $I/l_syn.json > $I/l_syn.log 2>&1; echo "l_syn $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 --target-kl 0.015 --seed 8 --steps 491520 --out $I/l_tkl.json > $I/l_tkl.log 2>&1; echo "l_tkl $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl2.py ${=XT} --no-nav --moves8 ${=AN} --lr-anneal-stop --seed 8 --steps 1e6 --out $I/l_an.json > $I/l_an.log 2>&1; echo "l_an $(date '+%H:%M:%S')"
$PY w2/learner_w2_rl_traj.py ${=XT} --moves8 --seed 8 --init-params $SP/m8lid/rl2_on.params --d-fixed 3 --eval-necessity 0 --traj-out $I/traj_on_d3.npz --out $I/traj_on_d3.json > $I/traj_on.log 2>&1; echo "traj $(date '+%H:%M:%S')"
python3 /Users/overlord/CascadeProjects/throng/docs/data/port_w2/e1/g2_checks/check.py > $I/RESULT.txt 2>&1; cat $I/RESULT.txt
echo G2CHK_DONE
