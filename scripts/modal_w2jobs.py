"""Modal launcher for the W2 learnability / fast-learner jobs (Cam Nov 10: jobs over ~10 min go to Modal, small CPU
containers, --ckpt on). throng-toy w2/ on the comm-toy branch, pinned.

    MODAL_PROFILE=roadtoagi111 modal deploy scripts/modal_w2jobs.py
    MODAL_PROFILE=roadtoagi111 python scripts/spawn_jobs.py w2 <plan> [--only tag,...]
Each job writes testbed/w2jobs/<tag>_<sha7>.json (fast learner: a directory) and checkpoints to testbed/w2jobs/<tag>.ckpt;
the volume is committed every 2 minutes so a preempted job resumes from its last checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import modal

_REQUIREMENTS = str(Path(__file__).resolve().parents[1] / "requirements.txt")
PINNED_SHA = "1256389b75b8d89b774f3b80a8c13aa44444e881"  # comm-toy: D' (evolution on food) + E (hungry-adult oracle)
REPO_URL = "https://github.com/overlordxrz-source/throng.git"
VOLUME_MOUNT = "/mnt/throng-runs"
OUTD = f"{VOLUME_MOUNT}/testbed/w2jobs"

app = modal.App("throng-w2jobs")
volume = modal.Volume.from_name("throng-runs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .run_commands(f"git clone {REPO_URL} /root/throng", f"cd /root/throng && git checkout {PINNED_SHA}",
                       f"cd /root/throng && test \"$(git rev-parse HEAD)\" = \"{PINNED_SHA}\"")
         .pip_install_from_requirements(_REQUIREMENTS))


@app.function(image=image, cpu=4.0, memory=6144, timeout=4 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_w2(tag: str, args: list) -> str:
    import os
    import subprocess
    import time
    os.chdir("/root/throng")
    assert subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == PINNED_SHA
    os.makedirs(OUTD, exist_ok=True)
    script, rest = args[0], list(args[1:])
    out = f"{OUTD}/{tag}_{PINNED_SHA[:7]}" + (".json" if script.endswith("learnability.py") else "")
    cmd = ["python", script, *rest, "--out", out, "--ckpt", f"{OUTD}/{tag}.ckpt"]
    log = open(f"{OUTD}/{tag}_{PINNED_SHA[:7]}.log", "a")
    log.write(f"=== start {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n"); log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    while p.poll() is None:
        time.sleep(120); log.flush(); volume.commit()
    log.write(f"=== exit {p.returncode} {time.strftime('%H:%M:%S')}\n"); log.close(); volume.commit()
    return f"{tag}: exit {p.returncode}"


@app.function(image=image, cpu=4.0, memory=8192, timeout=24 * 60 * 60, retries=0, volumes={VOLUME_MOUNT: volume})
def run_w2_long(tag: str, args: list) -> str:
    """the stage-1 pilot: up to 24 h (a timeout is not a preemption, so it would not be restarted); --ckpt makes a
    preempted run resume where it stopped"""
    return run_w2.local(tag, args)


_NAVS = "w2_out/nav_w13/*.msgpack"
_WORLD = ["--world", "0.0013,8,0.001"]
_APP = ["--beacon-walk", "--young-seek", "--beacon-spread", "6", "--test", "heard25", "--score", "last-heard", "--min-sectors", "5",
        "--eval-eps", "48", "--test-window", "600"]
_ARM = ["w2/learnability.py", "--navs", _NAVS, *_WORLD, *_APP, "--seek-age", "500", "--updates", "150", "--worlds", "16",
        "--snap-ages", "250,500,1000"]
_NOLEV = ["--juvenile", "0"]
_ALLON = ["--juvenile", "500", "--pred", "1.0", "--replay", "8", "--copy", "1.0", "--copy-replay", "--young-lr", "4"]
_DOSE = [*_ARM[:-2], "--juvenile", "500", "--pred", "1.0", "--copy", "1.0", "--copy-replay", "--young-lr", "4"]   # all-on base minus snap ages / replay
_ICP = ["w2/learnability.py", "--navs", _NAVS, *_WORLD, "--beacon-walk", "--young-seek", "--beacon-spread", "6", "--test", "heard25",
        "--score", "last-heard", "--min-sectors", "5", "--seek-age", "500", "--code-per-life", "--no-train", "--ic-probe",
        "--updates", "150", "--worlds", "64", "--ic-steps", "2000", "--ic-len", "640"]
_OBSL = ["w2/learnability.py", "--navs", _NAVS, *_WORLD, "--beacon-walk", "--young-seek", "--beacon-spread", "6", "--test", "heard25",
         "--score", "last-heard", "--min-sectors", "5", "--seek-age", "500", "--code-per-life", "--no-train", "--ic-probe", "--parent",
         "--eval-eps", "48", "--test-window", "600", "--updates", "150", "--worlds", "32", "--eval-eligible-only"]
_S1 = ["w2/learnability.py", "--navs", _NAVS, *_WORLD, "--beacon-walk", "--young-seek", "--beacon-spread", "6", "--test", "heard25",
       "--score", "last-heard", "--min-sectors", "5", "--seek-age", "500", "--parent", "--agent", "recurrent"]
_PILOT = [*_S1, "--parent-leave", "1000", "--code-family", "A", "--rec-hidden", "128", "--worlds", "64", "--updates", "61100",
          "--readout-at", "1e8,3e8,1e9", "--readout-worlds", "64", "--eval-updates", "60", "--eval-eps", "48", "--test-window", "600",
          "--test-energy", "0.3", "--ckpt-every", "500"]
PLANS = {
    # Cam Nov 12 stage-1 pilot (cap \$6; registered tree); launched only after hunger_ctl passes
    "s1pilot": [("s1pilot", _PILOT)],
    # Cam Nov 12: the ONE allowed relaunch = the registered pilot + the screen's pick (c) KL 0.3 to the navigator;
    # readouts halved at 1e8 / 3e8 (32 worlds), full at 1e9 (64); token influence logged every 100 updates; $6 cap
    # Cam Nov 13: RELAUNCH of the GRU arm on the fixed code (ratio-1, NaN guard, fixed forage rule); same 3e8 conditional; cap $2.4
    # Cam Nov 13 branch 3 (only if no A-C arm qualifies). Same screen readouts; 25M steps each.
    "dprime": [("klsDp_evolve_food", [*_PILOT, "--oracle-input", "--updates", "1526", "--readout-at", "1.25e7,2.5e7", "--readout-worlds-at", "32,32",
                                      "--readout-tests", "hungry_kept", "--readout-no-probe", "--ti-fixed", "--forage-worlds", "64", "--forage-updates", "200",
                                      "--evolve", "--evolve-fitness", "food", "--evolve-budget", "25e6", "--evolve-sigma", "0.3"])],
    "earm": [("klsE_hungry_adult", [*_PILOT, "--oracle-input", "--updates", "1526", "--readout-at", "1.25e7,2.5e7", "--readout-worlds-at", "32,32",
                                    "--readout-tests", "hungry_kept", "--readout-no-probe", "--ti-fixed", "--token-influence-every", "100",
                                    "--forage-stop", "--forage-worlds", "64", "--forage-updates", "200",
                                    "--juvenile", "0", "--parent-leave", "0", "--start-energy", "0.3", "--kl-nav", "0.3"])],
    "s1kl03r": [("s1kl03r", [*_PILOT, "--kl-nav", "0.3", "--token-influence-every", "100", "--readout-worlds-at", "32,32,64",
                             "--forage-stop", "--forage-worlds", "64", "--forage-updates", "200"])],
    "s1kl03": [("s1kl03", [*_PILOT, "--kl-nav", "0.3", "--token-influence-every", "100", "--readout-worlds-at", "32,32,64",
                           "--forage-stop", "--forage-worlds", "64", "--forage-updates", "200"])],
    # Cam Nov 12: scripted controls under the HUNGRY test start must reproduce their frozen numbers (EXPLORE, ~\$0.2)
    "hunger_ctl": [("hunger_gate", ["w2/learnability.py", "--navs", _NAVS, *_WORLD, *_APP, "--updates", "150", "--worlds", "16",
                                    "--no-train", "--sanity", "--test-energy", "0.3"]),
                   ("hunger_parent_true", [*_OBSL, "--obs-listener", "true", "--parent-leave", "1000", "--test-energy", "0.3"])],
    # Nov 12 price probe (EXPLORE, ~$0.01): stage-1 training throughput on a 4-CPU container
    "s1tp": [("s1tp_w64", [*_S1, "--worlds", "64", "--updates", "12", "--eval-updates", "0"])],
    # Cam Nov 12 plug-in check (EXPLORE, < $1): counting-observer listener vs the true-code control, frozen test, parent arm
    "obsl": [("obsl_observer", [*_OBSL, "--obs-listener", "observer"]), ("obsl_true", [*_OBSL, "--obs-listener", "true"])],
    "obsl1000": [("obsl1000_observer", [*_OBSL, "--obs-listener", "observer", "--parent-leave", "1000"]),
                 ("obsl1000_true", [*_OBSL, "--obs-listener", "true", "--parent-leave", "1000"])],
    # Nov 12, EXPLORE lane (< $1, no bars): the stage-1 supervised in-context probe at ~10x the laptop's lives
    "icp": [("icp_listen", [*_ICP]), ("icp_demo", [*_ICP, "--demonstrators"])],
    # Cam Nov 10 ~18:20: dose-response (R32 x5, R128 x3 after the price probe, J2000 x5); item cap $4, stopped at $5
    "dose": [(f"dose_r32_s{s}", [*_DOSE, "--snap-ages", "250,500,1000", "--replay", "32", "--seed", str(s)]) for s in range(5)]
            + [(f"dose_r128_s{s}", [*_DOSE, "--snap-ages", "250,500,1000", "--replay", "128", "--seed", str(s)]) for s in range(3)]
            + [(f"dose_j2000_s{s}", [*_DOSE[:_DOSE.index("--juvenile")], "--juvenile", "2000", *_DOSE[_DOSE.index("--juvenile") + 2:],
                                     "--snap-ages", "250,500,1000,2000", "--replay", "8", "--seed", str(s)]) for s in range(5)],
    # Cam Nov 10: the appetite-free (random-fallback) VoI on the full gate lives, REPORTED (same deterministic gate config)
    "app_voi_r": [("app_gate_voir", ["w2/learnability.py", "--navs", _NAVS, *_WORLD, *_APP, "--updates", "150",
                                     "--worlds", "16", "--no-train", "--sanity"])],
    # Cam Nov 10 (6): learnability 2 arms x 5 seeds after the frozen test; bars as registered
    "learn_arms": [(f"learn_nolev_s{s}", [*_ARM, *_NOLEV, "--seed", str(s)]) for s in range(5)]
                  + [(f"learn_allon_s{s}", [*_ARM, *_ALLON, "--seed", str(s)]) for s in range(5)],
    # apparatus gate (ii) (Cam Nov 10): scripted control >= 90% on eligible lives, non-listener <= 10%, VoI >= 1.5x
    "app_gate": [("app_gate_scripted", ["w2/learnability.py", "--navs", _NAVS, *_WORLD, *_APP, "--updates", "150",
                                        "--worlds", "16", "--no-train", "--sanity"])],
}
