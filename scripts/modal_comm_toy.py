"""Modal launcher for the CPU comm toy (scripts/comm_toy.py on the comm-toy branch). CPU containers only; Cam's
cap for toy runs on Modal: $2 total. One container per run; results to the volume under testbed/toy/.

    MODAL_PROFILE=rapidcrisp modal run scripts/modal_comm_toy.py --plan erosion
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")

PINNED_SHA = "24f3925c34e70ff2adb6613ab77b6ca08695c56d"  # comm-toy branch (Nov 10: + --spoil-wrong / --material-rows; default path bit-identical)
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_NAME = "throng-runs"
VOLUME_MOUNT = "/mnt/throng-runs"

app = modal.App("throng-comm-toy")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /root/throng",
        f"cd /root/throng && git checkout {PINNED_SHA}",
        f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"",
    )
    .pip_install_from_requirements(_REQUIREMENTS)
)


@app.function(image=image, cpu=2.0, memory=4096, timeout=2 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})   # tight (Cam Nov 9)
def run_toy(tag: str, args: list) -> str:
    import os
    import subprocess

    os.chdir("/root/throng")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_SHA, f"box did not clone the pinned SHA (got {head})"
    os.makedirs(f"{VOLUME_MOUNT}/testbed/toy", exist_ok=True)
    out = f"{VOLUME_MOUNT}/testbed/toy/{tag}_{head[:7]}.jsonl"
    args = [x.replace("{VOL}", VOLUME_MOUNT) for x in args]
    if "--save-at" not in args:   # Cam 2026-11-08: every toy run saves actor (+ critic) params at 1,500 / 1,750 / 2,000
        os.makedirs(f"{VOLUME_MOUNT}/testbed/toy/params", exist_ok=True)
        args += ["--save-at", "1500", "1750", "2000", "--save-prefix", f"{VOLUME_MOUNT}/testbed/toy/params/{tag}_{head[:7]}"]
    cmd = ["python", "scripts/comm_toy.py", *args, "--out", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    volume.commit()
    if proc.returncode != 0:
        print(proc.stdout[-3000:], proc.stderr[-3000:], flush=True)
        raise RuntimeError(f"{tag} exited {proc.returncode}")
    return open(out).read()


_W = "toy_out/ORACLE0.msgpack"
_BASE = ["--updates", "2000", "--warm-start", _W, "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
         "--ps-anneal", "1000", "1500", "--erode-from", "1500"]
_T3 = ["--warm-start", "toy_out/ORACLE0.msgpack", "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
       "--ps-anneal", "1000", "1500", "--reset-from", "1750", "--reset-every", "250", "--reset-until", "2750",
       "--eval-at", "1500", "1750", "2000", "2250", "2500", "2750", "--imitate"]
_ERO = ["--updates", "2000", "--warm-start", _W, "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
        "--ps-anneal", "1000", "1500", "--erode-from", "1500", "--erode-wd", "3e-4", "--eval-at", "1500"]
_ERO_JOBS = ([(f"erosion_on_s{s}", ["--arm", "COMM", "--seed", str(s), *_ERO, "--tag", "EROSION_ON"]) for s in (70, 71, 72)]
             + [(f"erosion_off_s{s}", ["--arm", "NOSIG", "--seed", str(s), *_ERO, "--tag", "EROSION_OFF"]) for s in (80, 81, 82)])
PLANS: dict = {
    # registered erosion test (2026-11-08): probe one container first (Cam: fan out only if < $0.01/min/container)
    "erosion_probe": _ERO_JOBS[:1],
    # step B (2026-11-08): regenerate T3 wire-on seeds to 1,750 with a full-state snapshot, then branch to 2,000
    **{f"stepB_pre_s{s}": [(f"stepB_pre_s{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "1750", *_T3,
                                               "--snapshot-at", "1750", "--snapshot-path", f"{{VOL}}/testbed/toy/snap/t3_s{s}_u1750.pkl",
                                               "--tag", "T3_ON"])] for s in (140, 141, 142)},
    "stepB_branch": [(f"stepB_{br}_s{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "2000", *_T3,
                                           "--resume", f"{{VOL}}/testbed/toy/snap/t3_s{s}_u1750.pkl", "--branch", br,
                                           "--tag", f"STEPB_{br.upper()}"])
                     for br in ("none", "frozen_listener", "stopgrad") for s in (140, 141, 142)],
    # step 6 (2026-11-08): compositionality pilot on step 2's winner (VQ, wire-blind), vocabulary cut to 4 per slot
    "step6": [(f"step6_comp_s{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "3000", "--warm-start", _W,
                                    "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99", "--wire-blind", "1",
                                    "--comp", "--vq-vocab", "4", "--eval-at", "1000", "2000", "--tag", "S6_COMP"])
              for s in range(210, 215)],
    # step 2 (2026-11-08): channel A/B, wire-blind head in both arms (C1-T3 confirmed the coupling); PS on, 2,000 updates
    "step2": [(f"step2_{tag.lower()}_s{s}", ["--arm", arm, "--seed", str(s), "--updates", "2000", "--warm-start", _W,
                                             "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
                                             "--wire-blind", "1", "--eval-at", "1000", *extra, "--tag", tag])
              for tag, arm, extra, ss in (("S2_VQ", "COMM", [], range(190, 195)),
                                          ("S2_SYM", "COMM", ["--channel", "sym"], range(195, 200)),
                                          ("S2_NOSIG", "NOSIG", [], (200, 201))) for s in ss],
    # C1-E replication (Cam item 4, 2026-11-08 night): fresh seeds; full-state snapshot at 2,000 so that a continuation (if
    # Cam approves) resumes faithfully instead of regenerating (C1 saved actor params only)
    # Cam Nov 9 approved the continuation to 3,000 (option B). Run straight to 3,000 instead of 2,000 + resume: nothing in
    # comm_toy.py depends on --updates except the loop bound and the final eval, mid-run evals leave training bit-identical
    # and snapshot+resume is bit-identical, so the 2,000 readout (eval-at 2000) is the same; the 2,000 snapshot is still written.
    "c1rep": [(f"c1rep_{w}_s{s}", ["--arm", arm, "--seed", str(s),
                                   *[x if x != "2000" else "3000" for x in _ERO], "--eval-at", "1500", "2000", "2500",
                                   "--save-at", "1500", "1750", "2000", "2500", "3000",
                                   "--save-prefix", f"{{VOL}}/testbed/toy/params/c1rep_{w}_s{s}",
                                   "--wire-blind", "1", "--snapshot-at", "2000",
                                   "--snapshot-path", f"{{VOL}}/testbed/toy/snap/c1rep_{w}_s{s}_u2000.pkl", "--tag", f"C1REP_{w.upper()}"])
              for w, arm, ss in (("on", "COMM", range(230, 235)), ("off", "NOSIG", (235, 236))) for s in ss],
    # NEXT ACCOUNT (Cam Nov 9 ~07:00) (b): erosion calibration. Grow 3 fresh wire-off seeds with stop-grad ON to a
    # full-state snapshot at 1,500 (weight decay starts at 1,501 so the snapshot is pre-erosion), then resume each at
    # wd 3e-4 / 1e-3 / 3e-3 to 2,500 with params saved every 250 (shuffle-corrected joint MI computed from them, free).
    "ero_grow": [(f"erocal_grow_s{s}", ["--arm", "NOSIG", "--seed", str(s), *[x if x != "2000" else "1500" for x in _ERO],
                                         "--erode-from", "1501", "--wire-blind", "1", "--msg-grad", "sender",
                                         "--snapshot-at", "1500", "--snapshot-path", f"{{VOL}}/testbed/toy/snap/erocal_s{s}_u1500.pkl",
                                         "--tag", "EROCAL_GROW"]) for s in (270, 271, 272)],
    "ero_cal": [(f"erocal_wd{wd}_s{s}", ["--arm", "NOSIG", "--seed", str(s), *_ERO, "--updates", "2500", "--erode-from", "1501",
                                          "--erode-wd", wd, "--wire-blind", "1", "--msg-grad", "sender",
                                          "--resume", f"{{VOL}}/testbed/toy/snap/erocal_s{s}_u1500.pkl",
                                          "--save-at", "1750", "2000", "2250", "2500",
                                          "--save-prefix", f"{{VOL}}/testbed/toy/params/erocal_wd{wd}_s{s}", "--tag", f"EROCAL_{wd}"])
                for wd in ("3e-4", "1e-3", "3e-3") for s in (270, 271, 272)],
    # (c) rung-3 rerun (Cam Nov 9 ~07:00 / ~07:55): fresh seeds, the C1-E erosion protocol to 3,000 at the CALIBRATED
    # weight decay (env RUNG3_WD, set from the registered pick); C1 baseline x5, stop-grad x5, wire-off x5 (stop-grad on)
    "rung3": [(f"r3_{arm_t}_s{s}", ["--arm", arm, "--seed", str(s), *[x if x != "2000" else "3000" for x in _ERO],
                                     "--erode-wd", os.environ.get("RUNG3_WD", "UNSET"), "--wire-blind", "1", *extra,
                                     "--eval-at", "1500", "2000", "2500",
                                     "--save-at", "1500", "1750", "2000", "2500", "3000",
                                     "--save-prefix", f"{{VOL}}/testbed/toy/params/r3_{arm_t}_s{s}", "--tag", f"R3_{arm_t.upper()}"])
              for arm_t, arm, extra, ss in (("base", "COMM", [], range(290, 295)),
                                            ("sg", "COMM", ["--msg-grad", "sender"], range(295, 300)),
                                            ("off", "NOSIG", ["--msg-grad", "sender"], range(300, 305))) for s in ss],
    # (f) step-6 rerun (Cam ~00:40 / Nov 9 ~14:45: tau 0.63 approved): one arm, fresh seeds; ORACLE0 for navigation,
    # message head + material input rows re-initialised (--reinit-msg), tau 0.63 (step 2's starting fraction of max entropy)
    "step6r": [(f"step6r_comp_s{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "3000", "--warm-start", _W,
                                      "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99", "--wire-blind", "1",
                                      "--comp", "--vq-vocab", "4", "--reinit-msg", "--tau", "0.63",
                                      "--eval-at", "1000", "2000", "--tag", "S6R_COMP"]) for s in range(310, 315)],
    # (f) step-6 rerun 2 (Cam Nov 10 ~02:25): world fix (a wrong-material Craft re-rolls the pair) + material on fresh input
    # rows; otherwise as step6r (reinit message head, tau 0.63); NOSIG x2 for the new rules' baseline
    "step6f": ([(f"step6f_comp_s{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "3000", "--warm-start", _W,
                                       "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99", "--wire-blind", "1",
                                       "--comp", "--vq-vocab", "4", "--reinit-msg", "--tau", "0.63", "--spoil-wrong",
                                       "--material-rows", "--eval-at", "1000", "2000", "--tag", "S6F_COMP"]) for s in range(330, 335)]
               + [(f"step6f_nosig_s{s}", ["--arm", "NOSIG", "--seed", str(s), "--updates", "3000", "--warm-start", _W,
                                          "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99", "--wire-blind", "1",
                                          "--comp", "--vq-vocab", "4", "--reinit-msg", "--tau", "0.63", "--spoil-wrong",
                                          "--material-rows", "--tag", "S6F_NOSIG"]) for s in (335, 336)]),
    # (d) separate brains: 8 G-12 navigators by ORACLE0's own recipe (founders 260-263, newborn pool 264-267), then the G0
    # word-bias gate: the C1 formation recipe to 1,500 from navigators 260 (seeds 280 / 281) and 261 (seeds 282 / 283)
    "nav": [(f"nav_s{s}", ["--arm", "ORACLE", "--seed", str(s), "--updates", "1500", "--craft-cost", "0.45",
                           "--save-params", f"{{VOL}}/testbed/toy/nav/nav_s{s}.msgpack", "--tag", "NAV"]) for s in range(260, 268)],
    "g0": [(f"g0_n{n}_s{s}", ["--arm", "COMM", "--seed", str(s), "--updates", "1500",
                              "--warm-start", f"{{VOL}}/testbed/toy/nav/nav_s{n}.msgpack", "--wire-init-scale", "0.06",
                              "--craft-cost", "0.45", "--ps", "1.99", "--ps-anneal", "1000", "1500", "--wire-blind", "1",
                              "--msg-grad", "sender", "--tag", f"G0_N{n}"])
           for n, ss in ((260, (280, 281)), (261, (282, 283))) for s in ss],
    # step 3b value-of-information check (Cam item 6, 2026-11-08 night): lookout world, PS off, wire off in both arms
    "s3b_voi": [(f"s3b_{w}_s{s}", ["--arm", "NOSIG", "--seed", str(s), "--updates", "2000", "--warm-start", _W, "--craft-cost", "0.45",
                                  "--grid", "24", "--hearths", "3,3;3,9;9,3;9,9", "--lookout", "18,18,1", "--reroll", "10", *extra,
                                  "--tag", f"S3B_{w.upper()}"])
                for w, extra, ss in (("oracle", ["--lookout-oracle"], (220, 221)), ("nosig", [], (222, 223))) for s in ss],
    # C1 (2026-11-08): wire-blind message head from the start; T3 protocol (to 2,000) and erosion protocol
    "c1": ([(f"c1t3_{w}_s{s}", ["--arm", arm, "--seed", str(s), "--updates", "2000", *_T3, "--wire-blind", "1", "--tag", f"C1T3_{w.upper()}"])
            for w, arm, ss in (("on", "COMM", (180, 181, 182)), ("off", "NOSIG", (183, 184))) for s in ss]
           + [(f"c1e_{w}_s{s}", ["--arm", arm, "--seed", str(s), *_ERO, "--wire-blind", "1", "--tag", f"C1E_{w.upper()}"])
              for w, arm, ss in (("on", "COMM", (185, 186, 187)), ("off", "NOSIG", (188, 189))) for s in ss]),
    # build-order step 3 (approved 2026-11-08): no assigned sender; PS on (formation), local critic
    "step3": [(f"step3_{tag.lower()}_s{s}", ["--arm", arm, "--seed", str(s), "--updates", "2000", "--warm-start", _W,
                                             "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
                                             "--eval-at", "1000", *extra, "--tag", tag])
              for tag, arm, extra, ss in (("PROX", "COMM", ["--prox", "2"], range(170, 175)),
                                          ("ASSIGNED", "COMM", [], range(175, 178)),
                                          ("PROX_NOSIG", "NOSIG", ["--prox", "2"], (178, 179))) for s in ss],
    # approved T4b regeneration (2026-11-08): bit-identical re-runs of the T4b seeds with params saved, for check 3 (b)/(c)
    "t4b_regen": [(f"t4b_regen_{w}_s{s}", ["--arm", arm, "--seed", str(s), *_ERO, "--credit", "Bg", "--tag", f"T4B_{w.upper()}"])
                  for w, arm, ss in (("on", "COMM", (160, 161)), ("off", "NOSIG", (163, 164))) for s in ss],
    # registered T4 (2026-11-08): centralized critic for message credit, erosion protocol, reliance-first bar
    "t4": [(f"t4{fx[0].lower()}_{w}_s{s}", ["--arm", arm, "--seed", str(s), *_ERO, "--credit", fx, "--tag", f"T4{fx[0]}_{w.upper()}"])
           for fx, base in (("Ag", 150), ("Bg", 160))
           for w, arm, ss in (("on", "COMM", (0, 1, 2)), ("off", "NOSIG", (3, 4))) for s in [base + i for i in ss]],
    # registered T1 / T2 / T3 (2026-11-08)
    "tbatch": ([(f"t1_{w}_s{s}", ["--arm", arm, "--seed", str(s), *_ERO, "--credit", "B", "--tag", f"T1_{w.upper()}"])
                for w, arm, ss in (("on", "COMM", range(120, 125)), ("off", "NOSIG", (125, 126))) for s in ss]
               + [(f"{t.lower()}_{w}_s{s}", ["--arm", arm, "--seed", str(s), "--updates", "3000", "--warm-start", _W,
                                             "--wire-init-scale", "0.06", "--craft-cost", "0.45", "--ps", "1.99",
                                             "--ps-anneal", "1000", "1500", "--reset-from", "1750", "--reset-every", "250",
                                             "--reset-until", "2750", "--eval-at", "1500", "1750", "2000", "2250", "2500",
                                             "2750", *(["--imitate"] if t == "T3" else []), "--tag", f"{t}_{w.upper()}"])
                  for t, base in (("T2", 130), ("T3", 140))
                  for w, arm, ss in (("on", "COMM", (0, 1, 2)), ("off", "NOSIG", (3, 4))) for s in [base + i for i in ss]]),
    # registered credit fixes under the erosion protocol (2026-11-08): wire-on x3 + wire-off x2 per fix, fresh seeds
    "credit": [(f"credit_{fx}_{w}_s{s}", ["--arm", arm, "--seed", str(s), *_ERO, "--credit", fx,
                                          "--tag", f"CREDIT_{fx}_{w.upper()}"])
               for fx, base in (("A", 100), ("B", 110))
               for w, arm, ss in (("on", "COMM", (0, 1, 2)), ("off", "NOSIG", (3, 4)))
               for s in [base + i for i in ss]],
    # registered erosion diagnostic (2026-11-08): fresh seeds 90 (wire on) / 91 (wire off), log 1,500-1,800
    "diag": [(f"diag_{tag}_s{s}", ["--arm", arm, "--seed", str(s), *[x if x != "2000" else "1800" for x in _ERO],
                                    "--diag-from", "1500", "--diag-to", "1800", "--diag-every", "10", "--diag-pairs", "4",
                                    "--swap-pairs", "4", "--tag", f"DIAG_{tag}"])
             for tag, arm, s in (("on", "COMM", 90), ("off", "NOSIG", 91))],
    "erosion_rest": _ERO_JOBS[1:],
    # exploratory (2026-11-08): weight-decay erosion calibration, wire OFF, 2 calibration seeds per strength.
    # The token-entropy force failed calibration (none < 0.5 bit); this is reported to Cam, not registered.
    "wdcal": [(f"wdcal_wd{wd}_s{s}", ["--arm", "NOSIG", "--seed", str(s), *_BASE, "--erode-wd", wd, "--tag", "WDCAL"])
              for wd in ("1e-4", "3e-4", "1e-3", "3e-3") for s in (60, 61)],
}


@app.local_entrypoint()
def main(plan: str, outdir: str = "toy_out_modal", only: str = ""):
    Path(outdir).mkdir(exist_ok=True)
    jobs = PLANS[plan]
    if plan == "rung3" and os.environ.get("RUNG3_WD", "UNSET") == "UNSET":
        raise SystemExit("rung3 needs RUNG3_WD (the registered erosion-calibration pick)")
    if only:   # comma-separated tags (e.g. a price probe of one job)
        jobs = [j for j in jobs if j[0] in only.split(",")]
    if not only and plan == "c1":
        jobs = [j for j in jobs if j[0] != "c1t3_on_s180"]   # the probe job runs separately
    for (tag, _), text in zip(jobs, run_toy.starmap(jobs)):
        (Path(outdir) / f"{tag}.jsonl").write_text(text)
        print(f"[toy] {tag}: {len(text.splitlines())} records", flush=True)
