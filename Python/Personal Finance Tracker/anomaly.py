"""
Detects anomalous transactions using two complementary methods:

1. IsolationForest  — catches statistically unusual amounts in each category
2. Duplicate detector — flags same amount + same description on same day
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two columns to df:
      - is_anomaly  (bool)  : True if IsolationForest flags it
      - anomaly_reason (str): human-readable explanation
    """
    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_reason"] = ""

    # ── 1. IsolationForest per category ──────────────────────────────────────
    # Only run on debits (negative amounts) to avoid flagging large salary credits.
    debits_mask = df["amount"] < 0

    for category, group in df[debits_mask].groupby("category"):
        if len(group) < 3:
            # Too few samples to train a meaningful model
            continue

        amounts = group["amount"].values.reshape(-1, 1)
        clf = IsolationForest(contamination=0.05, random_state=42)
        preds = clf.fit_predict(amounts)  # -1 = anomaly, 1 = normal

        anomaly_idx = group.index[preds == -1]
        df.loc[anomaly_idx, "is_anomaly"] = True

        # Describe how far the amount is from the category median
        median_abs = abs(group["amount"].median())
        for idx in anomaly_idx:
            amt = abs(df.loc[idx, "amount"])
            multiple = amt / median_abs if median_abs > 0 else 0
            df.loc[idx, "anomaly_reason"] = (
                f"Unusually large {category} spend "
                f"(₹{amt:,.0f} vs median ₹{median_abs:,.0f}, "
                f"{multiple:.1f}× typical)"
            )

    # ── 2. Duplicate transactions ─────────────────────────────────────────────
    dup_cols = ["date", "description", "amount"]
    dupes = df.duplicated(subset=dup_cols, keep=False)
    dup_idx = df[dupes & ~df["is_anomaly"]].index  # don't overwrite existing flag
    df.loc[dup_idx, "is_anomaly"] = True
    df.loc[dup_idx, "anomaly_reason"] = "Possible duplicate transaction"

    return df


def anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the anomalous rows, sorted by absolute amount descending."""
    flagged = df[df["is_anomaly"]].copy()
    flagged = flagged.sort_values("amount").reset_index(drop=True)
    return flagged[["date", "description", "category", "amount", "anomaly_reason"]]
