# Retail Sales Forecasting

Forecast next weeks' sales for a retail store, by product category —
so stock and staffing can be planned before the week happens, not after.

## Results (test set: last 12 weeks, 2022-10-16 → 2023-01-01)

Time-based split only — the test weeks are genuinely in the future
relative to training, never a random split. 36 test rows (12 weeks ×
3 categories).

| Model | RMSE | MAPE | WMAPE |
|---|---|---|---|
| Seasonal naive (same week, 52 weeks ago) | $5,529 | 50.0% | 54.1% |
| **LightGBM** | **$4,101** | 79.6% | **43.9%** |

LightGBM cuts RMSE by 26% and weighted MAPE by 10 points. It does
*not* win plain MAPE — and that is worth saying plainly. Weekly
category sales are spiky (a single big order can triple a week), and
plain MAPE explodes on quiet weeks: Technology had a $573 week and a
$1,036 week in the test period, where a $3,000–$5,000 forecast is a
200–400% error on its own. WMAPE (total absolute error ÷ total sales)
is the fairer summary for this series, and on WMAPE LightGBM wins in
every category.

### By category

| Category | Baseline RMSE / MAPE | LightGBM RMSE / MAPE |
|---|---|---|
| Furniture | $4,052 / 46.8% | $3,809 / 44.6% |
| Office Supplies | $4,797 / 53.7% | **$2,262 / 34.9%** |
| Technology | $7,230 / 49.4% | $5,552 / 159.2% |

Reproduce: `python src/train.py` — it prints these numbers and writes
`metrics.json` plus the plots in `plots/`.

## What drives the forecast? (LightGBM feature importance)

Top features by split count: rolling std (4w), lag 4, lag 2, rolling
mean (8w), rolling mean (4w), lag 52, lag 1, trend. In other words:
recent level and volatility matter most, with the same week last year
as a seasonal anchor.

## Method

1. **Data** — Sample Superstore, 9,994 orders, 2019-01-03 → 2022-12-30,
   aggregated to weekly sales per category: 209 weeks × 3 categories
   = 627 rows, total sales $2,297,201.
2. **Features** — lag 1 / 2 / 4 / 52 weeks, rolling mean (4w, 8w) and
   rolling std (4w), trend, month / week-of-year / quarter, month
   sine–cosine, category one-hots. Every feature uses `shift`, so no
   row ever sees future sales.
3. **Models** — seasonal-naive baseline (lag 52) vs LightGBM
   (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`,
   `random_state=42`). Prophet was not compared: it is not a light
   dependency for this repo and the baseline + LightGBM comparison
   already answers the question "does the model beat last year?".
4. **Demo** — `app.py` (Streamlit) trains in memory, then forecasts
   the next 4–12 weeks recursively for a selected category.

## Limitations (read before reusing this)

- The dataset is the Tableau **Sample Superstore**: a fictional/demo
  retail sample, not a real store's sales. It has no promotions,
  prices, holidays, or stock-outs as features — a real deployment
  would need those.
- Only 4 years of history (209 weeks), so the seasonal lag is estimated
  from at most 3 prior years.
- The final test week (ending 2023-01-01) is short: the data ends
  Friday 2022-12-30, so that week has only 5 days of orders.
- Recursive forecasts compound error: the 8-week-ahead numbers are
  indicative, not a commitment.

## Dataset source

Sample Superstore (Tableau sample, GitHub mirror):
<https://raw.githubusercontent.com/WuCandice/Superstore-Sales-Analysis/main/dataset/Superstore%20Dataset.csv>
A copy is bundled at `data/superstore.csv` so the project runs offline.

## Structure

```text
retail-sales-forecasting/
├── README.md
├── requirements.txt
├── app.py                  # Streamlit demo: forecast next weeks per category
├── metrics.json            # measured test-set results
├── data/
│   └── superstore.csv      # 9,994 orders, 2019–2022
├── src/
│   ├── forecast.py         # shared loading / features / recursive forecast
│   └── train.py            # trains, evaluates, writes metrics.json + plots
└── plots/
    ├── forecast_vs_actual.svg
    ├── forecast_by_category.svg
    └── feature_importance.svg
```
