"""Run scripted conversations under a policy, verify every active rule on every turn, log rows."""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kw):
        return x

from .constraints import build_system_prompt, count_tokens
from .policies import State, render_reminder


def _task_correct(resp: str, turn) -> float:
    if turn.kind == "chat":
        return float("nan")
    r = resp.lower()
    ok = any(g in r for g in turn.gold) and not any(x in r for x in turn.reject)
    return float(ok)


def run_conversation(scn, policy, llm, model_name: str):
    active = {c.cid: c for c in scn.constraints}
    last_stated = {cid: 0 for cid in active}
    stated_tok = {cid: 0 for cid in active}
    start_turn = {cid: 0 for cid in active}
    prev_viol = {cid: 0 for cid in active}

    system = build_system_prompt(scn.constraints)
    messages = [{"role": "system", "content": system}]
    cum = count_tokens(system)
    families = []
    c_rows, t_rows = [], []

    for turn in scn.turns:
        if turn.override:
            old_cid, new = turn.override
            active.pop(old_cid, None)
            active[new.cid] = new
            last_stated[new.cid], stated_tok[new.cid] = turn.idx, cum
            start_turn[new.cid], prev_viol[new.cid] = turn.idx, 0

        recent = families[-3:]
        off_frac3 = sum(f == "chat" for f in recent) / 3 if recent else 0.0
        topic_shift = int(bool(families) and turn.family != families[-1])
        feats = {}
        for cid, c in active.items():
            feats[cid] = dict(cls=c.cls, turns_since=turn.idx - last_stated[cid],
                              tokens_since=cum - stated_tok[cid], n_active=len(active),
                              off_frac3=off_frac3, topic_shift=topic_shift, ctx_tokens=cum,
                              is_override=int(turn.kind == "override"), prev_viol=prev_viol[cid])

        chosen, style = policy.select(State(turn.idx, dict(active), feats))
        reminded = {c.cid for c in chosen}
        reminder = render_reminder(chosen, style)
        for cid in reminded:
            last_stated[cid], stated_tok[cid] = turn.idx, cum

        user_text = turn.user + reminder
        messages.append({"role": "user", "content": user_text})
        meta = dict(active=list(active.values()),
                    turns_since={cid: turn.idx - last_stated[cid] for cid in active},
                    off_frac3=off_frac3, topic_shift=topic_shift,
                    base_text=("Sounds fun, I would enjoy that." if turn.kind == "chat"
                               else f"The answer is {turn.gold[0]}."))
        resp = llm.chat(messages, meta)
        messages.append({"role": "assistant", "content": resp})
        cum += count_tokens(user_text) + count_tokens(resp)

        oks = {}
        for cid, c in active.items():
            ok = c.check(resp)
            oks[cid] = ok
            f = feats[cid]
            c_rows.append(dict(model=model_name, policy=policy.name, sid=scn.sid, split=scn.split,
                               density=scn.density, turn=turn.idx, kind=turn.kind, cid=cid,
                               complied=ok, reminded=cid in reminded, age=turn.idx - start_turn[cid],
                               start=start_turn[cid], **f))
            prev_viol[cid] = int(not ok)
        n_ok = sum(oks.values())
        t_rows.append(dict(model=model_name, policy=policy.name, sid=scn.sid, split=scn.split,
                           density=scn.density, turn=turn.idx, kind=turn.kind, family=turn.family,
                           n_active=len(active), strict=float(n_ok == len(active)),
                           loose=n_ok / max(1, len(active)), task_correct=_task_correct(resp, turn),
                           reminder_tokens=count_tokens(reminder) if reminder else 0,
                           response_tokens=count_tokens(resp), ctx_tokens=cum))
        families.append(turn.family)
    return c_rows, t_rows


def run_policy(scenarios, policy, llm, model_name, out_dir, workers=4, force=False, desc=None):
    cpath = os.path.join(out_dir, f"constraint_{policy.name}.csv")
    tpath = os.path.join(out_dir, f"turn_{policy.name}.csv")
    if os.path.exists(cpath) and os.path.exists(tpath) and not force:
        return pd.read_csv(cpath), pd.read_csv(tpath)
    os.makedirs(out_dir, exist_ok=True)
    c_all, t_all = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_conversation, s, policy, llm, model_name) for s in scenarios]
        for f in tqdm(futs, desc=desc or policy.name, leave=False):
            c, t = f.result()
            c_all += c
            t_all += t
    c_df, t_df = pd.DataFrame(c_all), pd.DataFrame(t_all)
    c_df.to_csv(cpath, index=False)
    t_df.to_csv(tpath, index=False)
    return c_df, t_df
