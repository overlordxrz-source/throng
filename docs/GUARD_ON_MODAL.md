# Spend guard on Modal: steps for M

The spend guard currently runs on the Mac, so it stops if the Mac sleeps. `scripts/modal_guard.py` moves it onto
Modal as a function that runs every 2 minutes. If the month-to-date **metered** cost reaches the cap, it stops every
running THRONG app.

It needs a Modal API token stored as a Modal secret. **You create the token and the secret. Never paste the token
into chat, a file or a command anyone else will see.** Will only refers to the secret by name.

## Steps (about 3 minutes, in the Modal dashboard of the workspace the runs use)

1. **Create a token.** modal.com → Settings → API Tokens → **New Token**. Name it `throng-guard`. The dashboard shows
   the Token ID (`ak-…`) and Token Secret (`as-…`) **once**. Keep the page open for step 2.
2. **Store it as a secret.** modal.com → Secrets → **Create new secret** → **Custom**.
   - Name: `throng-guard-token` (exactly this).
   - Key `MODAL_TOKEN_ID`, value = the Token ID.
   - Key `MODAL_TOKEN_SECRET`, value = the Token Secret.
   - Save, then close the token page.
3. **Tell Will "guard secret created"**, with no values.

Will then runs one dry check that stops nothing (`modal run scripts/modal_guard.py::check`; it prints the metered
cost, the cap and the live apps), and deploys the schedule with the cap Cam set (`GUARD_CAP=15.0 modal deploy
scripts/modal_guard.py`). Until that dry check passes, the Mac guard keeps running as before.

## Good to know

- **The token can act on the whole workspace.** That is what lets it stop apps. Delete it (Settings → API Tokens)
  when the account's work is done, or at any time; the guard then simply fails its checks.
- **Cost:** one small container every 2 minutes for about 10 seconds, roughly $0.03 a day.
- **Where to look:** `modal app logs throng-guard` shows one line per check (time, metered, cap, live apps). To stop
  it: `modal app stop throng-guard`.
- **What it watches:** the *metered* cost, the same figure as the Mac guard. **Credits are not the cap.** On the
  `rapidcrisp` account the plan credits covered $30.00 and the remaining **$8.43 of the $38.44 metered was billed**
  (Modal's billing summary, Nov 9). If the allowance is meant to come only from credits, set the cap from the credits
  left, not from the total.
