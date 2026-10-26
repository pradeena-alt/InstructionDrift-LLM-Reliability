"""Verifiable constraints.

Each constraint has:
  * text()/short()  -- verbose instruction and compact reminder wording
  * check(resp)     -- programmatic verifier (no LLM judge needed)
  * apply(text, ok) -- used ONLY by the offline mock model to emit a compliant or violating reply
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass

KIND_CLASS = {
    "lowercase": "format",
    "uppercase": "format",
    "json_answer": "format",
    "max_words": "length",
    "banned_word": "negative",
    "term_rule": "terminology",
    "prefix_tag": "persistence",
    "suffix_tag": "persistence",
}
CLASSES = ["format", "length", "negative", "terminology", "persistence"]

# Application order used when rendering a mock reply.
_STAGE = {"max_words": 0, "term_rule": 1, "banned_word": 2, "json_answer": 3,
          "lowercase": 4, "uppercase": 4, "prefix_tag": 5, "suffix_tag": 5}

# Kinds that cannot be active together in one conversation.
EXCLUSIVE = [
    {"lowercase", "uppercase"},
    {"json_answer", "prefix_tag"},
    {"json_answer", "suffix_tag"},
    {"json_answer", "uppercase"},
]

BANNED_WORDS = ["however", "basically", "very", "actually", "simply"]
MAX_WORDS = [25, 30, 40]
PREFIX_TAGS = [">>>", "###", "@@@"]
SUFFIX_TAGS = ["<<<", "~~~", "$$$"]

_FILLER = ("this extra sentence adds more words to the reply without changing the "
           "answer at all ") * 5


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


@dataclass(frozen=True)
class Constraint:
    kind: str
    param: str = ""

    @property
    def cid(self) -> str:
        return f"{self.kind}:{self.param}" if self.param else self.kind

    @property
    def cls(self) -> str:
        return KIND_CLASS[self.kind]

    # ---- wording ---------------------------------------------------------
    def text(self) -> str:
        k, p = self.kind, self.param
        return {
            "lowercase": "Write your entire reply in lowercase letters only.",
            "uppercase": "Write your entire reply in UPPERCASE letters only.",
            "json_answer": 'Reply with a single JSON object of the form {"answer": "<your answer>"} and nothing else.',
            "max_words": f"Use at most {p} words in your reply.",
            "banned_word": f'Never use the word "{p}".',
            "term_rule": 'Always say "learner"; never use the word "student".',
            "prefix_tag": f"Begin every reply with the tag {p}",
            "suffix_tag": f"End every reply with the tag {p}",
        }[k]

    def short(self) -> str:
        k, p = self.kind, self.param
        return {
            "lowercase": "all lowercase",
            "uppercase": "ALL UPPERCASE",
            "json_answer": 'JSON only {"answer": ...}',
            "max_words": f"max {p} words",
            "banned_word": f'never say "{p}"',
            "term_rule": 'say "learner" not "student"',
            "prefix_tag": f"start with {p}",
            "suffix_tag": f"end with {p}",
        }[k]

    # ---- verifier --------------------------------------------------------
    def check(self, resp: str) -> bool:
        k, p = self.kind, self.param
        if k == "lowercase":
            return resp == resp.lower()
        if k == "uppercase":
            return resp == resp.upper()
        if k == "json_answer":
            try:
                d = json.loads(_strip_fences(resp))
            except Exception:
                return False
            return isinstance(d, dict) and isinstance(d.get("answer"), str)
        if k == "max_words":
            return len(resp.split()) <= int(p)
        if k == "banned_word":
            return re.search(rf"\b{re.escape(p)}\b", resp, re.I) is None
        if k == "term_rule":
            return re.search(r"\bstudents?\b", resp, re.I) is None
        if k == "prefix_tag":
            return resp.lstrip().startswith(p)
        if k == "suffix_tag":
            return resp.rstrip().endswith(p)
        raise ValueError(k)

    # ---- mock-only transformation ---------------------------------------
    def apply(self, text: str, ok: bool) -> str:
        k, p = self.kind, self.param
        if k == "max_words":
            return " ".join(text.split()[: int(p)]) if ok else text + " " + _FILLER
        if k == "term_rule":
            return text if ok else text + " Every student can do this."
        if k == "banned_word":
            return text if ok else text + f" This is {p} fine."
        if k == "json_answer":
            return json.dumps({"answer": text}) if ok else text
        if k == "lowercase":
            return text.lower() if ok else text          # base text starts with a capital
        if k == "uppercase":
            return text.upper() if ok else text          # base text contains lowercase
        if k == "prefix_tag":
            return f"{p} {text}" if ok else text
        if k == "suffix_tag":
            return f"{text} {p}" if ok else text
        raise ValueError(k)


def render_response(base: str, flags: dict) -> str:
    """Apply constraint transformations in a fixed order (mock model only)."""
    out = base
    for c in sorted(flags, key=lambda c: _STAGE[c.kind]):
        out = c.apply(out, flags[c])
    return out


def compatible(kinds: set, new: str) -> bool:
    if new in kinds:
        return False
    return not any(new in grp and (grp - {new}) & kinds for grp in EXCLUSIVE)


def build_system_prompt(constraints) -> str:
    lines = ["You are a helpful assistant. Follow ALL of these rules in EVERY reply for the "
             "whole conversation, unless the user explicitly changes them:"]
    for i, c in enumerate(constraints, 1):
        lines.append(f"{i}. {c.text()}")
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token). Replace with a real tokenizer if desired."""
    return max(1, round(len(text) / 4))
