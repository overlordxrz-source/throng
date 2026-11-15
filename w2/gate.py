"""Cam Nov 15 ~12:20: the HONEST curriculum gate for W2.
Score = hungry-bout reach (bouts whose energy stayed < H until the item was eaten or T_max; sated bouts count neither
way). The frontier advances only when TWO CONSECUTIVE, NON-OVERLAPPING windows of `win` closed hungry bouts each score
>= `bar`. Within one iteration the outcomes are laid out evenly (deterministic), so a window boundary never favours
successes or failures."""
import numpy as np


class HonestGate:  # noqa: D101
    def __init__(self, win=2000, bar=0.8):
        self.win, self.bar = win, bar; self.buf = []; self.prev = None; self.scores = []

    def add(self, n_eat, n_hungry):
        """n_eat successes out of n_hungry closed hungry bouts this iteration. Returns True when the frontier advances."""
        n_eat, n_hungry = int(n_eat), int(n_hungry)
        if n_hungry > 0:
            pos = np.floor((np.arange(n_eat) + 0.5) * n_hungry / max(n_eat, 1)).astype(int) if n_eat else np.zeros(0, int)
            o = np.zeros(n_hungry, np.int8); o[pos] = 1; self.buf.extend(o.tolist())
        up = False
        while len(self.buf) >= self.win:
            sc = float(np.mean(self.buf[: self.win])); self.buf = self.buf[self.win:]; self.scores.append(sc)
            if self.prev is not None and self.prev >= self.bar and sc >= self.bar:
                up = True; self.prev = None; self.buf = []; break
            self.prev = sc
        return up

    def state(self):
        return dict(buf=self.buf, prev=self.prev, scores=self.scores)

    def load(self, st):
        self.buf, self.prev, self.scores = list(st["buf"]), st["prev"], list(st["scores"])
