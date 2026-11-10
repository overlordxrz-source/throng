"""World presets for the W2 harnesses (Cam Nov 10 ~05:15, 4b: move the learnability and fast-learner harnesses to the
retuned W2 world). "proto" = the prototype world (P6, d0 0.008, no relocation); otherwise "d0,P,relocate"."""


def world_kw(spec: str) -> dict:
    if spec == "proto":
        return dict(P=6, d0=0.008, relocate=0.0)
    d0, P, rel = spec.split(",")
    return dict(P=int(P), d0=float(d0), relocate=float(rel))
