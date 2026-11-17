"""FUN F1: watch a life. usage: f1_render.py dump.npz mode(learn|dead) out_stem
learn: a life with a FIRST hearing of a word (parent-led) and a LATER hearing of the same word (alone), both eaten.
dead: one bout in a diagonal sector where the head's argmax is right and the young barely moves (timeout)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
f, mode, stem = sys.argv[1:4]
z = dict(np.load(f)); Wn, T, N = z["hear"].shape; L = 96
U = np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], -1)
tor = lambda o: (o + L // 2) % L - L // 2  # noqa: E731
def bouts(w, n):
    hear = z["hear"][w, :, n]; close = z["cl_item"][w, :, n] | z["cl_tmax"][w, :, n] | z["cl_death"][w, :, n]; out = []
    for s0 in np.flatnonzero(hear):
        e = np.flatnonzero(close[s0:])
        if len(e): out.append((int(s0), int(s0 + e[0])))
    return out
pick = None
for w in range(Wn):
    for n in range(N):
        B = bouts(w, n); dead = z["dead"][w, :, n]
        for i, (a0, a1) in enumerate(B):
            sec = int(z["sec"][w, a0, n])
            if mode == "learn":
                if not (z["first"][w, a0, n] and z["dem"][w, a0, n] and z["cl_item"][w, a1, n] and z["d"][w, a0, n] >= 5): continue
                for (b0, b1) in B[i + 1:]:
                    if dead[a1:b0].any(): break
                    if int(z["sec"][w, b0, n]) == sec and not z["dem"][w, b0, n] and z["cl_item"][w, b1, n] and z["d"][w, b0, n] >= 5 and int(z["hl"][w, b0, n]) == sec:
                        pick = (w, n, [(a0, a1), (b0, b1)]); break
            else:
                mv = np.isin(z["a"][w, a0:a1 + 1, n], [1, 2, 3, 4]).sum()
                if sec % 2 == 1 and z["cl_tmax"][w, a1, n] and int(z["hl"][w, a0, n]) == sec and mv <= 3 and z["d"][w, a0, n] >= 6 and not z["dem"][w, a0, n]:
                    pick = (w, n, [(a0, a1)])
            if pick: break
        if pick: break
    if pick: break
assert pick, "no suitable life found"
w, n, BB = pick
frames = []
for k, (s0, s1) in enumerate(BB):
    frames += [(s, k) for s in range(max(s0 - 2, 0), min(s1 + 4, T))]
    if k < len(BB) - 1: frames += [(None, k)] * 6
s_ref = BB[0][0]; c0 = z["start"][w, BB[-1][0], n].astype(int)
R_ = int(max(z["d"][w, s0, n] for s0, _ in BB)) + 5
fig = plt.figure(figsize=(8.4, 5.6), dpi=90); ax = fig.add_axes([0.03, 0.06, 0.6, 0.88]); axm = fig.add_axes([0.66, 0.42, 0.31, 0.5]); axe = fig.add_axes([0.66, 0.12, 0.31, 0.06])
def draw(fr):
    s, k = fr
    for a_ in (ax, axm, axe): a_.clear()
    if s is None:
        ax.text(0.5, 0.5, "... later in the same life ...", ha="center", va="center", fontsize=14, transform=ax.transAxes); ax.axis("off"); axm.axis("off"); axe.axis("off"); return
    s0, s1 = BB[k]; st = z["start"][w, s0, n].astype(int); cen = st
    rel = lambda p: tor(np.asarray(p).astype(int) - cen)  # noqa: E731
    ax.set_xlim(-R_, R_); ax.set_ylim(-R_, R_); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticks(np.arange(-R_, R_ + 1) - 0.5, minor=True); ax.set_yticks(np.arange(-R_, R_ + 1) - 0.5, minor=True); ax.grid(which="minor", lw=0.3, color="#ddd")
    inb = s0 <= s <= s1; yp = rel(z["pos"][w, s, n]); ip = rel(z["ipos"][w, s0, n]); vis = np.abs(ip - yp).max() <= 1
    ax.plot(0, 0, "x", color="#888", ms=9, mew=2); ax.text(0.4, -0.9, "where it heard", fontsize=7, color="#888")
    ax.plot(*ip, "*", ms=18, color="#2a9d3a" if vis else "#9fd3a8", mec="k", mew=0.5, alpha=1 if vis else 0.6)
    ax.text(ip[0] + 0.6, ip[1] + 0.6, "called item" + ("" if vis else " (not visible)"), fontsize=7, color="#2a7d3a")
    tr = np.array([rel(z["pos"][w, q, n]) for q in range(max(s0, frames[0][0] or 0), s + 1)]) if s >= s0 else np.zeros((0, 2))
    if len(tr): ax.plot(tr[:, 0], tr[:, 1], "-", color="#4a78c8", lw=1.2, alpha=0.6)
    if z["dem"][w, s, n] and inb:
        dp = rel(z["dem_pos"][w, s, n]); ax.plot(*dp, "o", ms=13, color="#e8902a", mec="k"); ax.text(dp[0] + 0.6, dp[1] - 0.9, "parent", fontsize=8, color="#b0601a")
    ax.plot(*yp, "o", ms=11, color="#2b5fb8", mec="k"); ax.text(yp[0] + 0.6, yp[1] + 0.4, "young", fontsize=8, color="#2b5fb8")
    hs = z["hsoft"][w, s, n].astype(float); hk = int(np.argmax(hs))
    if inb:
        ax.annotate("", xy=(yp[0] + U[hk, 0] * 2.5 * hs[hk] + 1e-3, yp[1] + U[hk, 1] * 2.5 * hs[hk]), xytext=tuple(yp), arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.2))
    sec = int(z["sec"][w, s0, n]); tok = int(np.argmax(z["x"][w, s0, n][100:108])) if z["x"][w, s0, n][100:108].max() > 0 else -1
    title = f"word #{tok} (sector {sec}, d = {int(z['d'][w, s0, n])})  " + ("FIRST hearing: parent leads" if z["first"][w, s0, n] else "LATER hearing: alone")
    if s == s0: title = "THE CALL: " + title
    if z["cl_item"][w, s1, n] and s >= s1: title += "  -> ATE"
    if z["cl_tmax"][w, s1, n] and s >= s1: title += "  -> timed out"
    ax.set_title(title, fontsize=9.5, color="#c0392b" if s == s0 else "k")
    # head rose + attended memory
    axm.set_xlim(-1.3, 1.3); axm.set_ylim(-1.3, 1.3); axm.set_aspect("equal"); axm.axis("off")
    axm.set_title("head: where is the item? (red)\nmost-attended memory record (blue)", fontsize=8)
    for j in range(8):
        axm.plot([0, U[j, 0] * hs[j]], [0, U[j, 1] * hs[j]], "-", color="#c0392b", lw=1 + 5 * hs[j], alpha=0.8 if inb else 0.25)
        axm.text(U[j, 0] * 1.18, U[j, 1] * 1.18, str(j), ha="center", va="center", fontsize=7, color="#999")
    if inb and z["att_w"][w, s, n] > 0:
        rec = z["att_rec"][w, s, n].astype(float); dv = rec[8:10] * 24; nv = np.linalg.norm(dv) + 1e-9; rt = int(np.argmax(rec[:8]))
        axm.annotate("", xy=(dv[0] / nv, dv[1] / nv), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#2b5fb8", lw=1.5, ls="--"))
        axm.text(0, -1.32, f"slot {int(z['att_j'][w, s, n])}: word #{rt}, went ({int(round(dv[0]))},{int(round(dv[1]))}), {'ate' if rec[10] > 0.5 else 'missed'}; weight {float(z['att_w'][w, s, n]):.2f}", ha="center", fontsize=7, color="#2b5fb8")
    E = float(z["energy"][w, s, n]); axe.barh([0], [E], color="#2a9d3a" if E > 0.6 else "#e0a020"); axe.set_xlim(0, 1); axe.set_yticks([]); axe.axvline(0.7, color="k", lw=0.8, ls=":")
    axe.set_title(f"energy {E:.2f} (dotted: hungry below .70)", fontsize=8)
anim = animation.FuncAnimation(fig, draw, frames=frames, interval=220)
anim.save(stem + ".gif", writer=animation.PillowWriter(fps=5))
try:
    anim.save(stem + ".mp4", writer=animation.FFMpegWriter(fps=5))
except Exception as e:
    print("mp4 failed", e)
print("WROTE", stem, "world", w, "agent", n, "bouts", BB, "frames", len(frames))
