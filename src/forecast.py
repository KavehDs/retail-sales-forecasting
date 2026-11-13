"""Shared data + feature logic for the retail sales forecast.

The raw file is the classic Sample Superstore dataset (Tableau sample,
mirrored on GitHub): order-level retail transactions, 2019-01-03 to
2022-12-30. We aggregate to *weekly sales per category* and forecast that
series. All lag / rolling features are computed with ``shift`` only, so a
row never sees future sales.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

DATA = Path(__file__).resolve().parent.parent / "data" / "superstore.csv"
CATEGORIES = ["Furniture", "Office Supplies", "Technology"]
TEST_WEEKS = 12
FORECAST_HORIZON = 8


def load_weekly(csv_path=DATA):
    """Return weekly sales per category.

    Columns: week (Sunday-ending), category, sales. The week x category
    grid is completed so a week with no orders for a category is 0 sales
    rather than a missing row.
    """
    df = pd.read_csv(csv_path, encoding="unicode_escape")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])
    df["week"] = df["Order Date"].dt.to_period("W-SUN").dt.to_timestamp("W-SUN")
    weekly = (
        df.groupby(["week", "Category"], as_index=False)["Sales"]
        .sum()
        .rename(columns={"Category": "category", "Sales": "sales"})
    )
    all_weeks = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="W-SUN")
    grid = pd.MultiIndex.from_product(
        [all_weeks, CATEGORIES], names=["week", "category"]
    ).to_frame(index=False)
    weekly = grid.merge(weekly, on=["week", "category"], how="left")
    weekly["sales"] = weekly["sales"].fillna(0.0)
    return weekly.sort_values(["category", "week"]).reset_index(drop=True)


def add_features(weekly):
    """Add lag, rolling, calendar and trend features (past only)."""
    df = weekly.copy()
    df = df.sort_values(["category", "week"]).reset_index(drop=True)
    g = df.groupby("category")["sales"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_4"] = g.shift(4)
    df["lag_52"] = g.shift(52)
    df["roll_mean_4"] = g.shift(1).rolling(4).mean()
    df["roll_mean_8"] = g.shift(1).rolling(8).mean()
    df["roll_std_4"] = g.shift(1).rolling(4).std()
    df["trend"] = df.groupby("category").cumcount()
    df["month"] = df["week"].dt.month
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["week"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    for cat in CATEGORIES:
        df[f"is_{cat.replace(' ', '_')}"] = (df["category"] == cat).astype(int)
    return df


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_52",
    "roll_mean_4",
    "roll_mean_8",
    "roll_std_4",
    "trend",
    "month",
    "weekofyear",
    "quarter",
    "month_sin",
    "month_cos",
    "is_Furniture",
    "is_Office_Supplies",
    "is_Technology",
]


def train_lightgbm(train_df):
    """Fit LightGBM on rows with complete features."""
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(train_df[FEATURES], train_df["sales"])
    return model


def recursive_forecast(model, history, category, horizon=FORECAST_HORIZON):
    """Forecast `horizon` weeks ahead for one category, recursively.

    `history` is the full weekly panel (with sales). Each step appends the
    previous prediction as if it were observed sales, so lag features for
    the next week use the model's own forecasts.
    """
    hist = history[history["category"] == category].copy()
    last_week = hist["week"].max()
    rows = hist.to_dict("records")
    preds = []
    for step in range(1, horizon + 1):
        week = last_week + pd.Timedelta(weeks=step)
        # Append a placeholder row for the new week first: add_features()
        # then computes its lags / rolling means from the rows before it
        # (real sales for step 1, the model's own forecasts afterwards).
        rows.append({"week": week, "category": category, "sales": np.nan})
        tmp = pd.DataFrame(rows)
        tmp = add_features(tmp)
        last = tmp.iloc[[-1]]
        pred = float(model.predict(last[FEATURES])[0])
        pred = max(pred, 0.0)
        preds.append({"week": week, "category": category, "forecast": pred})
        rows[-1]["sales"] = pred
    return pd.DataFrame(preds)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def wmape(y_true, y_pred):
    """Weighted MAPE: sum|error| / sum(actual). Robust when some weeks
    are near-zero, where plain MAPE explodes on a single quiet week."""
    import numpy as np

    return float(
        np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)).sum()
        / np.asarray(y_true, dtype=float).sum()
        * 100
    )


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
