"""Statistics, survival analysis, Pareto analysis, figures and report generation."""
from __future__ import annotations
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constraints import CLASSES

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ loading
def load_run(out_dir: str):
    c = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out_dir, "constraint_*.csv")))]
    t = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(out_dir, "turn_*.csv")))]
    return pd.concat(c, ignore_index=True), pd.concat(t, ignore_index=True)


def group_of(policy: str) -> str:
    return policy.split("_t")[0] if policy.startswith("P4") else policy


def tau_of(policy: str):
    return float(policy.split("_t")[1]) if policy.startswith("P4") else np.nan


# ------------------------------------------------------------------ per-conversation table
def conv_table(turn_df: pd.DataFrame, split: str | None = None, max_turn: int | None = None) -> pd.DataFrame:
    d = turn_df
    if split:
        d = d[d.split == split]
    if max_turn:
        d = d[d.turn <= max_turn]
    g = d.groupby(["policy", "sid"]).agg(
        strict=("strict", "mean"), loose=("loose", "mean"),
        task_acc=("task_correct", "mean"), reminder_tokens=("reminder_tokens", "sum"),
        total_tokens=("response_tokens", "sum")).reset_index()
    return g


def _boot_ci(vals: np.ndarray, n=2000, alpha=0.05):
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    m = vals[idx].mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def summarize(conv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pol, g in conv.groupby("policy"):
        row = {"policy": pol, "n_conv": len(g)}
        for m in ["strict", "loose", "task_acc", "reminder_tokens"]:
            v = g[m].dropna().values
            lo, hi = _boot_ci(v)
            row[m] = v.mean()
            row[m + "_lo"], row[m + "_hi"] = lo, hi
        rows.append(row)
    s = pd.DataFrame(rows)
    if "P1" in set(s.policy) and "P0" in set(s.policy):
        p1, p0 = s.set_index("policy").loc["P1"], s.set_index("policy").loc["P0"]
        s["gap_closed"] = (s.strict - p0.strict) / max(1e-9, (p1.strict - p0.strict))
        s["token_ratio_vs_P1"] = s.reminder_tokens / max(1e-9, p1.reminder_tokens)
    return s.sort_values("policy").reset_index(drop=True)


def paired_diff(conv: pd.DataFrame, a: str, b: str, metric="strict", n=5000):
    """Paired bootstrap over conversations for mean(a) - mean(b)."""
    pa = conv[conv.policy == a].set_index("sid")[metric]
    pb = conv[conv.policy == b].set_index("sid")[metric]
    d = (pa - pb.reindex(pa.index)).dropna().values
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return dict(a=a, b=b, metric=metric, diff=float(d.mean()),
                lo=float(np.quantile(m, 0.025)), hi=float(np.quantile(m, 0.975)),
                p_boot=float(2 * min((m <= 0).mean(), (m >= 0).mean())))


# ------------------------------------------------------------------ survival
def km_curve(durations: np.ndarray, events: np.ndarray):
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    times = np.unique(d[e == 1])
    S, s = [], 1.0
    for t in times:
        at_risk = (d >= t).sum()
        dead = ((d == t) & (e == 1)).sum()
        s *= 1 - dead / at_risk
        S.append(s)
    return times, np.array(S)


def survival_table(c_df: pd.DataFrame, policy="P0", by="cls"):
    """Time-to-first-violation of each (conversation, rule) instance, counted in turns since stated."""
    d = c_df[c_df.policy == policy].sort_values(["sid", "cid", "turn"])
    out = {}
    for key, sub in d.groupby(by):
        dur, ev = [], []
        for _, g in sub.groupby(["sid", "cid"]):
            viol = g[~g.complied.astype(bool)]
            if len(viol):
                dur.append(viol.age.iloc[0] + 1); ev.append(1)
            else:
                dur.append(g.age.iloc[-1] + 1); ev.append(0)
        out[key] = (np.array(dur), np.array(ev))
    return out


def half_life(times, S, level=0.5):
    below = np.where(S <= level)[0]
    return float(times[below[0]]) if len(below) else float("nan")


# ------------------------------------------------------------------ tau selection (on val)
def select_tau(conv_val: pd.DataFrame, target_ratio=0.95):
    """For each adaptive variant, the cheapest tau reaching >= target_ratio of P1's strict compliance on val."""
    s = summarize(conv_val).set_index("policy")
    chosen = {}
    p1 = s.loc["P1", "strict"]
    for var in ["P4a", "P4b"]:
        cand = s[[p.startswith(var + "_") for p in s.index]]
        if cand.empty:
            continue
        ok = cand[cand.strict >= target_ratio * p1]
        row = ok.sort_values("reminder_tokens").iloc[0] if len(ok) else cand.sort_values("strict").iloc[-1]
        chosen[var] = dict(policy=row.name, met_target=bool(len(ok)))
    return chosen


# ------------------------------------------------------------------ figures
def fig_decay(turn_df, policies, path, split="test"):
    d = turn_df[(turn_df.split == split) & (turn_df.policy.isin(policies))]
    g = d.groupby(["policy", "turn"]).strict.mean().reset_index()
    plt.figure(figsize=(7, 4))
    for p, gg in g.groupby("policy"):
        plt.plot(gg.turn, gg.strict, label=p, marker="o", ms=3)
    plt.xlabel("turn"); plt.ylabel("strict compliance (all active rules)"); plt.ylim(0, 1.02)
    plt.title("Instruction-compliance decay"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def fig_km(surv, path):
    plt.figure(figsize=(7, 4))
    for k in CLASSES:
        if k in surv:
            t, S = km_curve(*surv[k])
            plt.step(np.r_[0, t], np.r_[1, S], where="post", label=k)
    plt.axhline(0.5, color="gray", ls="--", lw=.8)
    plt.xlabel("turns since rule was stated"); plt.ylabel("survival (no violation yet)")
    plt.title("Instruction survival by class (P0: system prompt only)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def fig_pareto(summary, path):
    plt.figure(figsize=(7, 4.5))
    s = summary.copy()
    s["group"] = s.policy.map(group_of)
    for gname, g in s.groupby("group"):
        g = g.sort_values("reminder_tokens")
        line = gname.startswith("P4")
        plt.plot(g.reminder_tokens, g.strict, marker="o" if line else "s",
                 ls="-" if line else "", label=gname)
        if not line:
            for _, r in g.iterrows():
                plt.annotate(r.policy, (r.reminder_tokens, r.strict), fontsize=6, xytext=(3, 3),
                             textcoords="offset points")
    plt.xlabel("reminder tokens per conversation"); plt.ylabel("strict compliance")
    plt.title("Compliance vs. token overhead (Pareto view)")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


# ------------------------------------------------------------------ report
def _md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def full_report(out_dir: str, is_mock: bool, target_ratio=0.95):
    c_df, t_df = load_run(out_dir)
    figs = os.path.join(out_dir, "figures"); os.makedirs(figs, exist_ok=True)
    lines = ["# InstructionDrift results", ""]
    if is_mock:
        lines += ["> **WARNING: produced with the MOCK model. Numbers only demonstrate that the pipeline "
                  "works and must NOT be reported as research findings.**", ""]

    # 1. survival / half-life (descriptive; all splits, P0)
    surv = survival_table(c_df, "P0")
    rows = []
    for k in CLASSES:
        if k in surv:
            t, S = km_curve(*surv[k])
            rows.append({"class": k, "n_rule_instances": len(surv[k][0]),
                         "violation_rate": surv[k][1].mean(), "half_life_turns": half_life(t, S)})
    hl = pd.DataFrame(rows)
    fig_km(surv, os.path.join(figs, "km_by_class.png"))
    lines += ["## 1. Instruction half-life (P0, Kaplan-Meier)", _md(hl),
              "", "`half_life_turns = nan` means survival never fell to 50% within the conversation (censored).", ""]

    # 2. tau selection on val
    conv_val = conv_table(t_df, "val")
    chosen = select_tau(conv_val, target_ratio) if "P1" in set(conv_val.policy) else {}
    lines += ["## 2. Threshold selection (validation split)",
              f"Rule: cheapest tau with strict compliance >= {target_ratio:.0%} of P1.",
              "```", json.dumps(chosen, indent=2), "```", ""]

    # 3. test summary
    conv_test = conv_table(t_df, "test")
    keep = [p for p in conv_test.policy.unique()
            if not p.startswith("P4") or p in {v["policy"] for v in chosen.values()} or not chosen]
    summ_all = summarize(conv_test)
    summ = summ_all[summ_all.policy.isin(keep)]
    show = summ[["policy", "n_conv", "strict", "strict_lo", "strict_hi", "loose", "task_acc",
                 "reminder_tokens"] + [c for c in ["gap_closed", "token_ratio_vs_P1"] if c in summ]]
    lines += ["## 3. Test-split results (95% cluster-bootstrap CIs over conversations)", _md(show), ""]
    lines += ["`gap_closed` = (strict - P0) / (P1 - P0). `token_ratio_vs_P1` = reminder tokens relative to full repetition.", ""]

    # 4. paired comparisons
    comp = []
    for var, info in chosen.items():
        pol = info["policy"]
        comp.append(paired_diff(conv_test, pol, "P0"))
        comp.append(paired_diff(conv_test, pol, "P1"))
        p2 = summ_all[summ_all.policy.str.startswith("P2_")]
        if len(p2):
            tok = summ_all.set_index("policy").loc[pol, "reminder_tokens"]
            near = p2.iloc[(p2.reminder_tokens - tok).abs().argmin()].policy
            comp.append(paired_diff(conv_test, pol, near))
    if comp:
        lines += ["## 4. Paired bootstrap differences (strict compliance)", _md(pd.DataFrame(comp)), ""]

    # 5. by turn-horizon
    hz = []
    for h in [5, 10, 20, 30]:
        ct = conv_table(t_df, "test", max_turn=h)
        sm = summarize(ct)
        sm = sm[sm.policy.isin(keep)][["policy", "strict"]].assign(horizon=h)
        hz.append(sm)
    hz = pd.concat(hz).pivot(index="policy", columns="horizon", values="strict").reset_index()
    lines += ["## 5. Strict compliance by conversation horizon (first N turns)", _md(hz), ""]

    # figures
    fig_decay(t_df, keep, os.path.join(figs, "decay.png"))
    fig_pareto(summ_all, os.path.join(figs, "pareto.png"))
    lines += ["## Figures", "- `figures/decay.png`, `figures/km_by_class.png`, `figures/pareto.png`", ""]

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    summ_all.to_csv(os.path.join(out_dir, "summary_test.csv"), index=False)
    return path
