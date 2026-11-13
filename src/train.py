"""Retail sales forecasting — training & evaluation.

Dataset: Sample Superstore (Tableau sample, mirrored on GitHub),
data/superstore.csv — 9,994 orders, 2019-01-03 to 2022-12-30.
Aggregated to weekly sales per category (Furniture, Office Supplies,
Technology). This is a fictional/demo retail dataset, not real store data.

Split is strictly by TIME: the last 12 weeks are the test set, so the
models are judged on genuinely future weeks, never on a random split.

Run:  python src/train.py        (from the repo root)
Outputs: metrics.json, plots/*.svg
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast import (  # noqa: E402
    FEATURES,
    TEST_WEEKS,
    add_features,
    load_weekly,
    mape,
    rmse,
    train_lightgbm,
    wmape,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "metrics.json"
PLOTS = ROOT / "plots"


def main():
    weekly = load_weekly()
    raw_rows = sum(1 for _ in open(ROOT / "data" / "superstore.csv", encoding="unicode_escape")) - 1
    feat = add_features(weekly)
    feat = feat.dropna(subset=FEATURES).reset_index(drop=True)

    weeks = sorted(feat["week"].unique())
    test_weeks = weeks[-TEST_WEEKS:]
    train = feat[feat["week"] < test_weeks[0]].copy()
    test = feat[feat["week"] >= test_weeks[0]].copy()

    # Baseline: seasonal naive — same category, 52 weeks ago.
    test = test.copy()
    test["pred_baseline"] = test["lag_52"]

    import numpy as np

    model = train_lightgbm(train)
    test["pred_lgbm"] = np.maximum(model.predict(test[FEATURES]), 0)

    results = {}
    for name, col in [("Seasonal naive (lag 52)", "pred_baseline"), ("LightGBM", "pred_lgbm")]:
        results[name] = {
            "rmse": rmse(test["sales"], test[col]),
            "mape_pct": mape(test["sales"], test[col]),
            "wmape_pct": wmape(test["sales"], test[col]),
        }

    per_category = {}
    for cat, sub in test.groupby("category"):
        per_category[cat] = {
            "n_test_weeks": int(len(sub)),
            "baseline_rmse": rmse(sub["sales"], sub["pred_baseline"]),
            "baseline_mape_pct": mape(sub["sales"], sub["pred_baseline"]),
            "baseline_wmape_pct": wmape(sub["sales"], sub["pred_baseline"]),
            "lgbm_rmse": rmse(sub["sales"], sub["pred_lgbm"]),
            "lgbm_mape_pct": mape(sub["sales"], sub["pred_lgbm"]),
            "lgbm_wmape_pct": wmape(sub["sales"], sub["pred_lgbm"]),
        }

    importance = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
        .to_dict()
    )
    importance = {k: int(v) for k, v in importance.items()}

    metrics = {
        "dataset": {
            "name": "Sample Superstore (Tableau sample, GitHub mirror)",
            "file": "data/superstore.csv",
            "source_url": "https://raw.githubusercontent.com/WuCandice/Superstore-Sales-Analysis/main/dataset/Superstore%20Dataset.csv",
            "raw_orders": int(raw_rows),
            "columns": 20,
            "date_min": str(weekly["week"].min().date()),
            "date_max": str(weekly["week"].max().date()),
            "n_weeks": int(weekly["week"].nunique()),
            "categories": ["Furniture", "Office Supplies", "Technology"],
            "weekly_rows": int(len(weekly)),
            "total_sales": float(weekly["sales"].sum()),
        },
        "split": {
            "type": "time-based, no random split",
            "test_weeks": TEST_WEEKS,
            "test_start": str(pd.Timestamp(test_weeks[0]).date()),
            "test_end": str(pd.Timestamp(test_weeks[-1]).date()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        },
        "results": results,
        "per_category": per_category,
        "feature_importance_lightgbm": importance,
    }
    OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    # ---- Plots ----
    PLOTS.mkdir(exist_ok=True)
    total = weekly.groupby("week")["sales"].sum()
    test_total = test.groupby("week").agg(actual=("sales", "sum"), lgbm=("pred_lgbm", "sum"), baseline=("pred_baseline", "sum"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(total.index, total.values, label="Actual (all categories)", color="#1f3a5f")
    ax.plot(test_total.index, test_total["lgbm"].values, label="LightGBM forecast (test)", color="#c0392b", marker="o", ms=3)
    ax.plot(test_total.index, test_total["baseline"].values, label="Seasonal-naive baseline (test)", color="#7f8c8d", linestyle="--", marker="x", ms=3)
    ax.axvline(test_total.index[0], color="grey", linestyle=":", label="Test starts")
    ax.set_title("Weekly retail sales — test period (last 12 weeks)")
    ax.set_ylabel("Sales ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "forecast_vs_actual.svg")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (cat, sub) in zip(axes, test.groupby("category")):
        sub = sub.sort_values("week")
        ax.plot(sub["week"], sub["sales"], label="Actual", color="#1f3a5f", marker="o", ms=3)
        ax.plot(sub["week"], sub["pred_lgbm"], label="LightGBM", color="#c0392b", marker="s", ms=3)
        ax.plot(sub["week"], sub["pred_baseline"], label="Seasonal naive", color="#7f8c8d", linestyle="--", marker="x", ms=3)
        ax.set_title(cat)
        ax.set_ylabel("Sales ($)")
        ax.legend(fontsize=8)
    fig.suptitle("Test period by category — actual vs forecast")
    fig.tight_layout()
    fig.savefig(PLOTS / "forecast_by_category.svg")
    plt.close(fig)

    imp = pd.Series(importance).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp.plot.barh(ax=ax, color="#1f3a5f")
    ax.set_title("LightGBM feature importance (split count)")
    ax.set_xlabel("Splits")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.svg")
    plt.close(fig)
    print("Plots written to", PLOTS)


if __name__ == "__main__":
    main()
