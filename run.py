"""InstructionDrift command line.

  python run.py baselines   --config config.yaml     # P0,P1,P2,P3 on all splits
  python run.py train-risk  --config config.yaml     # fit risk models on TRAIN split (no-reminder rows)
  python run.py adaptive    --config config.yaml     # P4a/P4b over the tau grid on val+test splits
  python run.py analyze     --config config.yaml     # tables, stats, figures, report.md
  python run.py all         --config config.yaml
"""
import argparse
import json
import os

import pandas as pd
import yaml

from idrift.analysis import full_report, load_run
from idrift.llm import make_llm
from idrift.policies import parse_policy
from idrift.risk import RiskModel
from idrift.runner import run_policy
from idrift.scenarios import make_scenarios


def setup(cfg):
    out = os.path.join("results", cfg["run_name"])
    os.makedirs(out, exist_ok=True)
    scns = make_scenarios(cfg["n_scenarios"], cfg["n_turns"], cfg["seed"], cfg["densities"],
                          tuple(cfg["split_fracs"]))
    return out, scns, make_llm(cfg)


def cmd_baselines(cfg):
    out, scns, llm = setup(cfg)
    for p in cfg["baseline_policies"]:
        run_policy(scns, parse_policy(p), llm, cfg["model"], out, cfg["workers"])
        print("done", p)


def cmd_train_risk(cfg):
    out, _, _ = setup(cfg)
    c_df, _ = load_run(out)
    base = c_df[(c_df.split == "train") & (~c_df.reminded)]
    val = c_df[(c_df.split == "val") & (~c_df.reminded)]
    metrics = {}
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    for name, reactive in [("A", False), ("B", True)]:
        m = RiskModel(reactive=reactive, kind=cfg["risk_model"]).fit(base)
        metrics[f"risk_{name}"] = {"train": m.evaluate(base), "val": m.evaluate(val)}
        m.save(os.path.join(out, "models", f"risk_{name}.joblib"))
    json.dump(metrics, open(os.path.join(out, "risk_metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))


def cmd_adaptive(cfg):
    out, scns, llm = setup(cfg)
    models = {"a": RiskModel.load(os.path.join(out, "models", "risk_A.joblib")),
              "b": RiskModel.load(os.path.join(out, "models", "risk_B.joblib"))}
    held = [s for s in scns if s.split in ("val", "test")]     # never run adaptive on the training split
    for v in ["a", "b"]:
        for tau in cfg["taus"]:
            pol = parse_policy(f"P4{v}_t{tau:.2f}", models)
            run_policy(held, pol, llm, cfg["model"], out, cfg["workers"])
            print("done", pol.name)


def cmd_analyze(cfg):
    out, _, _ = setup(cfg)
    print("report:", full_report(out, is_mock=cfg["backend"] == "mock", target_ratio=cfg["target_ratio"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "train-risk", "adaptive", "analyze", "all"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    steps = {"baselines": [cmd_baselines], "train-risk": [cmd_train_risk], "adaptive": [cmd_adaptive],
             "analyze": [cmd_analyze],
             "all": [cmd_baselines, cmd_train_risk, cmd_adaptive, cmd_analyze]}[a.cmd]
    for s in steps:
        s(cfg)
