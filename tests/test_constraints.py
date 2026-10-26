import itertools
import random

from idrift.constraints import Constraint, render_response, compatible, KIND_CLASS
from idrift.scenarios import make_scenarios

ALL = [Constraint("lowercase"), Constraint("uppercase"), Constraint("json_answer"),
       Constraint("max_words", "25"), Constraint("banned_word", "however"), Constraint("term_rule"),
       Constraint("prefix_tag", ">>>"), Constraint("suffix_tag", "<<<")]
BASES = ["The answer is canberra.", "The answer is carbon dioxide.", "Sounds fun, I would enjoy that."]


def _valid_sets():
    for r in range(1, 6):
        for combo in itertools.combinations(ALL, r):
            kinds = set()
            ok = True
            for c in combo:
                if not compatible(kinds, c.kind):
                    ok = False
                    break
                kinds.add(c.kind)
            if ok:
                yield combo


def test_every_constraint_alone_complies_and_violates():
    for c in ALL:
        for b in BASES:
            assert c.check(render_response(b, {c: True})), (c.cid, b)
            assert not c.check(render_response(b, {c: False})), (c.cid, b)


def test_all_compatible_combinations_with_random_flags():
    rng = random.Random(1)
    for combo in _valid_sets():
        for b in BASES:
            for _ in range(6):
                flags = {c: rng.random() < 0.5 for c in combo}
                resp = render_response(b, flags)
                for c in combo:
                    assert c.check(resp) == flags[c], (c.cid, flags, resp)


def test_scenarios_are_deterministic_and_consistent():
    a = make_scenarios(20, 30, 7, [0.0, 0.3, 0.6])
    b = make_scenarios(20, 30, 7, [0.0, 0.3, 0.6])
    assert [t.user for s in a for t in s.turns] == [t.user for s in b for t in s.turns]
    assert {s.split for s in a} == {"train", "val", "test"}
    for s in a:
        kinds = {c.kind for c in s.constraints}
        assert not any(len(g & kinds) > 1 for g in
                       [{"lowercase", "uppercase"}, {"json_answer", "prefix_tag"},
                        {"json_answer", "suffix_tag"}, {"json_answer", "uppercase"}])
        assert s.turns[0].kind != "chat"
        for t in s.turns:
            if t.kind == "override":
                assert {c.kind for c in s.constraints} >= {t.override[0].split(":")[0]}
