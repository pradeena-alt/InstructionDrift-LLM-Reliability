"""Reinforcement policies.

Assumption (documented in the paper): every reminder policy knows the CURRENT set of active
rules (the application tracks rule changes). This isolates *when/what* to restate from
the separate problem of tracking rule updates.

Names:  P0 | P1 | P2_k<K> | P3 | P4a_t<TAU> | P4b_t<TAU>
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

from .constraints import build_system_prompt
from .risk import RiskModel


@dataclass
class State:
    turn: int
    active: dict           # cid -> Constraint
    feats: dict            # cid -> feature dict (decision-time, before any reminder)


def render_reminder(constraints, style: str) -> str:
    if not constraints:
        return ""
    if style == "full":      # verbatim repetition of the full rule block
        return "\n\n[Reminder]\n" + build_system_prompt(constraints)
    return "\n\n[Rules: " + "; ".join(c.short() for c in constraints) + "]"


class Policy:
    name = "base"

    def select(self, st: State):
        """Return (list of constraints to restate, style)."""
        return [], "short"


class SystemOnly(Policy):
    name = "P0"


class FullRepeat(Policy):
    name = "P1"

    def select(self, st):
        return list(st.active.values()), "full"


class FixedInterval(Policy):
    def __init__(self, k: int):
        self.k, self.name = k, f"P2_k{k}"

    def select(self, st):
        return (list(st.active.values()) if st.turn % self.k == 0 else []), "short"


class Ledger(Policy):
    """Compact rule ledger attached to every turn (summarised constraint memory)."""
    name = "P3"

    def select(self, st):
        return list(st.active.values()), "short"


class Adaptive(Policy):
    def __init__(self, model: RiskModel, tau: float):
        self.model, self.tau = model, tau
        self.name = f"P4{'b' if model.reactive else 'a'}_t{tau:.2f}"

    def select(self, st):
        cids = list(st.active)
        if not cids:
            return [], "short"
        df = pd.DataFrame([st.feats[c] for c in cids])
        p = self.model.predict(df)
        return [st.active[c] for c, pi in zip(cids, p) if pi > self.tau], "short"


def parse_policy(name: str, models: dict | None = None) -> Policy:
    if name == "P0":
        return SystemOnly()
    if name == "P1":
        return FullRepeat()
    if name == "P3":
        return Ledger()
    if name.startswith("P2_k"):
        return FixedInterval(int(name[4:]))
    if name.startswith("P4"):
        variant, tau = name[2], float(name.split("_t")[1])
        return Adaptive(models[variant], tau)
    raise ValueError(name)
