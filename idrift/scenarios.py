"""Scripted multi-turn scenarios with persistent constraints, distractors and rule overrides.

User turns are fixed in advance so every policy sees IDENTICAL user input; only the
assistant replies (and optional reminders) differ between policies.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional

from .constraints import (Constraint, KIND_CLASS, compatible, BANNED_WORDS, MAX_WORDS,
                          PREFIX_TAGS, SUFFIX_TAGS)

QA = [
    ("What is the capital of Australia?", ["canberra"]),
    ("Which planet is known as the Red Planet?", ["mars"]),
    ("What gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2"]),
    ("Who wrote the play Hamlet?", ["shakespeare"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("At sea level, at what temperature in Celsius does water boil?", ["100"]),
    ("Which organ pumps blood around the human body?", ["heart"]),
    ("What is the square root of 144?", ["12", "twelve"]),
    ("Which metal is liquid at room temperature?", ["mercury"]),
    ("What is the freezing point of water in Fahrenheit?", ["32"]),
]

SENT = [
    ("The battery died within an hour and support never replied.", "negative", "positive"),
    ("Setup took two minutes and the screen is gorgeous.", "positive", "negative"),
    ("Total waste of money, it broke on day three.", "negative", "positive"),
    ("Fast delivery and the quality exceeded my expectations.", "positive", "negative"),
    ("The app crashes constantly and loses my work.", "negative", "positive"),
    ("Friendly staff, spotless rooms, I will definitely come back.", "positive", "negative"),
    ("Rude waiter and the food arrived cold.", "negative", "positive"),
    ("Brilliant camera and the battery lasts all week.", "positive", "negative"),
    ("Instructions were confusing and half the parts were missing.", "negative", "positive"),
    ("Comfortable, sturdy and exactly as described.", "positive", "negative"),
]

CHAT = [
    "Unrelated, but what is a good name for a pet goldfish?",
    "By the way, do you prefer tea or coffee in the morning?",
    "Totally off topic: any tips for staying focused while studying?",
    "Random thought, what makes a good weekend plan?",
    "Quick aside, what is your favourite season and why?",
    "Just curious, how do people usually learn to juggle?",
    "Side question: what is a nice hobby to pick up this year?",
    "Out of nowhere, what is the best way to remember names?",
    "Change of subject, how would you describe a perfect sunset?",
    "Unrelated again, what is a fun fact about octopuses?",
    "Different topic, how do you politely end a long meeting?",
    "Just chatting: what board game is good for four players?",
]

FAMILIES = ["qa", "sentiment", "extraction", "arithmetic"]


@dataclass
class Turn:
    idx: int                       # 1-based
    kind: str                      # task | chat | override
    family: str                    # qa | sentiment | extraction | arithmetic | chat
    user: str
    gold: list = field(default_factory=list)
    reject: list = field(default_factory=list)
    override: Optional[tuple] = None   # (old_cid, new Constraint)


@dataclass
class Scenario:
    sid: str
    split: str
    density: float
    constraints: list
    turns: list


def _task_item(family: str, rng: random.Random, used: set):
    if family == "qa":
        pool = [q for q in QA if q[0] not in used] or QA
        q, gold = rng.choice(pool)
        used.add(q)
        return q, gold, []
    if family == "sentiment":
        pool = [s for s in SENT if s[0] not in used] or SENT
        t, g, r = rng.choice(pool)
        used.add(t)
        return f'Is the sentiment of this review positive or negative? Review: "{t}"', [g], [r]
    if family == "extraction":
        code = f"INV-{rng.randint(1000, 9999)}"
        amt, day = rng.randint(50, 900), rng.randint(1, 28)
        return (f'Extract the invoice number from this text: "Invoice {code} for ${amt} '
                f'was paid on {day} March."'), [code.lower()], []
    a, b = rng.randint(11, 99), rng.randint(11, 99)
    if rng.random() < 0.5:
        return f"What is {a} + {b}?", [str(a + b)], []
    a, b = max(a, b), min(a, b)
    return f"What is {a} - {b}?", [str(a - b)], []


def _pick_initial(rng: random.Random):
    kinds = list(KIND_CLASS)
    rng.shuffle(kinds)
    target = rng.choice([2, 3, 4, 5])
    chosen: set = set()
    for k in kinds:
        if len(chosen) >= target:
            break
        if compatible(chosen, k):
            chosen.add(k)
    out = []
    for k in sorted(chosen):
        p = {"max_words": str(rng.choice(MAX_WORDS)), "banned_word": rng.choice(BANNED_WORDS),
             "prefix_tag": rng.choice(PREFIX_TAGS), "suffix_tag": rng.choice(SUFFIX_TAGS)}.get(k, "")
        out.append(Constraint(k, p))
    return out


def _override_for(c: Constraint, active_kinds: set, rng: random.Random):
    if c.kind == "lowercase" and "json_answer" not in active_kinds:
        return Constraint("uppercase")
    if c.kind == "uppercase":
        return Constraint("lowercase")
    if c.kind == "max_words":
        return Constraint("max_words", str(rng.choice([n for n in MAX_WORDS if str(n) != c.param])))
    if c.kind == "banned_word":
        return Constraint("banned_word", rng.choice([w for w in BANNED_WORDS if w != c.param]))
    if c.kind == "prefix_tag":
        return Constraint("prefix_tag", rng.choice([t for t in PREFIX_TAGS if t != c.param]))
    if c.kind == "suffix_tag":
        return Constraint("suffix_tag", rng.choice([t for t in SUFFIX_TAGS if t != c.param]))
    return None


def make_scenario(i: int, n_turns: int, density: float, seed: int, split: str) -> Scenario:
    rng = random.Random(f"{seed}-{i}")
    init = _pick_initial(rng)
    active = {c.cid: c for c in init}
    # Optional mid-conversation rule override
    ov_turn, ov = None, None
    if rng.random() < 0.5 and n_turns >= 8:
        cand = [c for c in init if _override_for(c, {a.kind for a in init}, rng) is not None]
        if cand:
            old = rng.choice(cand)
            ov_turn = rng.randint(4, n_turns - 3)
            ov = (old, _override_for(old, {a.kind for a in init}, rng))
    turns, used, prev_family = [], set(), None
    for t in range(1, n_turns + 1):
        is_chat = t > 1 and t != ov_turn and rng.random() < density
        if is_chat:
            turns.append(Turn(t, "chat", "chat", rng.choice(CHAT)))
            prev_family = "chat"
            continue
        fam = rng.choice(FAMILIES)
        q, gold, rej = _task_item(fam, rng, used)
        if t == ov_turn:
            old, new = ov
            user = (f"Change of rules: from now on, {new.text()} This replaces your earlier rule "
                    f"({old.short()}). Now: {q}")
            turns.append(Turn(t, "override", fam, user, gold, rej, (old.cid, new)))
        else:
            turns.append(Turn(t, "task", fam, q, gold, rej))
        prev_family = fam
    return Scenario(f"s{i:04d}", split, density, init, turns)


def make_scenarios(n: int, n_turns: int, seed: int, densities, split_fracs=(0.5, 0.2, 0.3)):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_tr, n_va = int(split_fracs[0] * n), int(split_fracs[1] * n)
    split_of = {}
    for j, i in enumerate(idx):
        split_of[i] = "train" if j < n_tr else "val" if j < n_tr + n_va else "test"
    return [make_scenario(i, n_turns, densities[i % len(densities)], seed, split_of[i]) for i in range(n)]
