"""Instruction-loss risk model: P(constraint violated this turn | NOT reminded, features).

Trained only on rows where no reminder was injected, so it estimates the natural hazard of
losing a rule. Two variants:
  predictive (A): no verifier feedback -> uses only observable context features
  reactive   (B): additionally uses whether the rule was violated on the previous turn
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constraints import CLASSES

NUM_A = ["turns_since", "log_tokens_since", "n_active", "off_frac3", "topic_shift", "log_ctx", "is_override"]
NUM_B = NUM_A + ["prev_viol"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_tokens_since"] = np.log1p(df["tokens_since"])
    df["log_ctx"] = np.log1p(df["ctx_tokens"])
    return df


class RiskModel:
    def __init__(self, reactive: bool = False, kind: str = "logreg"):
        self.reactive, self.kind = reactive, kind
        self.num = NUM_B if reactive else NUM_A
        clf = (LogisticRegression(max_iter=2000, C=1.0) if kind == "logreg"
               else GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0))
        pre = ColumnTransformer([
            ("num", StandardScaler(), self.num),
            ("cls", OneHotEncoder(categories=[CLASSES], handle_unknown="ignore"), ["cls"]),
        ])
        self.pipe = Pipeline([("pre", pre), ("clf", clf)])

    def _X(self, df):
        return add_derived(df)[self.num + ["cls"]]

    def fit(self, df: pd.DataFrame):
        y = (~df["complied"].astype(bool)).astype(int).values
        if y.min() == y.max():
            raise ValueError("Training labels contain a single class (no violations or all violations). "
                             "Increase turns/densities or use a model that drifts.")
        self.pipe.fit(self._X(df), y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(self._X(df))[:, 1]

    def evaluate(self, df: pd.DataFrame) -> dict:
        y = (~df["complied"].astype(bool)).astype(int).values
        p = self.predict(df)
        if y.min() == y.max():
            return {"n": int(len(y))}
        return {"n": int(len(y)), "violation_rate": float(y.mean()),
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p))}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RiskModel":
        return joblib.load(path)
