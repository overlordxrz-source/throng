"""THRONG spend guard as a scheduled Modal function (Cam, Nov 9 ~07:00, next-account item (a)).

Every 2 minutes: read the workspace's month-to-date METERED cost (`modal billing summary --json`), list live apps,
and if metered >= CAP, stop every running app except this guard. It is the Mac guard (~/throng-smoke/cap_guard_rc.sh)
moved onto Modal, so it no longer depends on the Mac staying awake.

The token never passes through the agent: M creates the secret `throng-guard-token` in the Modal dashboard (see
docs/GUARD_ON_MODAL.md). This file only names the secret.

    modal run    scripts/modal_guard.py::check         # one dry check: prints metered, CAP, live apps; stops nothing
    modal deploy scripts/modal_guard.py                # start the 2-minute schedule
    modal app logs throng-guard                        # its log
    modal app stop throng-guard                        # stop the schedule

CAP comes from the GUARD_CAP environment variable at deploy time (default 15.0, Cam's next-account guard).
"""
from __future__ import annotations

import os

import modal

CAP = float(os.environ.get("GUARD_CAP", "15.0"))
GUARD_APP = "throng-guard"

app = modal.App(GUARD_APP)
image = modal.Image.debian_slim(python_version="3.11").pip_install("modal")
SECRETS = [modal.Secret.from_name("throng-guard-token"), modal.Secret.from_dict({"GUARD_CAP": str(CAP)})]


def _cli(*args: str) -> str:
    """run the modal CLI with ONLY the guard token as credentials (the container's own task identity is dropped, so the
    CLI acts as the workspace token, as on the Mac)"""
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith("MODAL_")}
    env["MODAL_TOKEN_ID"] = os.environ["MODAL_TOKEN_ID"]; env["MODAL_TOKEN_SECRET"] = os.environ["MODAL_TOKEN_SECRET"]
    r = subprocess.run(["modal", *args], capture_output=True, text=True, env=env, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"modal {' '.join(args)} failed: {r.stderr[-500:]}")
    return r.stdout


def _state():
    import json
    metered = float(json.loads(_cli("billing", "summary", "--json"))["metered_cost"])
    apps = json.loads(_cli("app", "list", "--json"))
    live = [a for a in apps if a.get("state") not in ("stopped", "stopping") and a.get("description") != GUARD_APP]
    return metered, live


def _guard(dry: bool):
    import time
    cap = float(os.environ.get("GUARD_CAP", CAP))
    metered, live = _state()
    ids = [a["app_id"] for a in live]
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} metered={metered:.2f} cap={cap:.2f} live={' '.join(ids) or '-'}"
    if metered >= cap and ids and not dry:
        for i in ids:
            _cli("app", "stop", "-y", i)
        line += f" | CAP HIT: stopped {' '.join(ids)}"
    print(line, flush=True)
    return line


@app.function(image=image, secrets=SECRETS, schedule=modal.Period(minutes=2), cpu=0.25, memory=256, timeout=120,
              retries=0)
def guard():
    return _guard(dry=False)


@app.function(image=image, secrets=SECRETS, cpu=0.25, memory=256, timeout=120)
def check_remote():
    return _guard(dry=True)


@app.local_entrypoint()
def check():
    print(check_remote.remote())
