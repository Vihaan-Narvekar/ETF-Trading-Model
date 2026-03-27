# Simple Quant Trading Project

A compact, two-step quant workflow:

1. Run regression models on macro + market features to predict forward ETF returns.
2. Convert those predictions into a simple, rules-based strategy backtest.

This project is intentionally minimal and centered around two folders:

- `regression_analysis/`: data collection, feature engineering, regression modeling
- `model_development/`: strategy rules and backtest

## Project structure

- `regression_analysis/run_regression_analysis.py`
- `model_development/run_strategy_development.py`
- `outputs/` (generated datasets, model results, predictions, strategy metrics)
- `requirements.txt`

## Data sources

- ETF prices: `yfinance` (SPY, QQQ, IWM)
- Macro factors: FRED CSV endpoints
- CAPE source (reference): `http://www.econ.yale.edu/~shiller/data/ie_data.xls`

## Models included

- `ols` (linear regression)
- `ridge`
- `lasso`
- `elasticnet`
- `huber`
- `quantile`

All models are trained in a consistent pipeline:

- median imputation
- standard scaling
- regression model fit

## Target and features

- Target: forward 3-month log return for each ETF (`y_SPY`, `y_QQQ`, `y_IWM`)
- Feature examples:
  - yield curve slope (`10Y - 2Y`)
  - CPI YoY
  - unemployment
  - rate level (10Y yield)
  - simple cross-ETF technical aggregates (momentum, volatility, mean reversion)
- Features are lagged by 1 month to reduce lookahead risk.

## Train/test and prediction flow

- Uses `train_test_split(..., train_size=60, shuffle=False)` to set the initial train window.
- Then builds expanding-window out-of-sample predictions for the remaining periods.
- This keeps the code simple while preserving realistic time ordering.

## Strategy logic (simple and consistent)

For each month:

1. Use each ETF's best model from `outputs/best_models.csv`.
2. Signal = sign of predicted return.
3. Normalize to gross exposure = 1.
4. Shift weights by 1 period (no lookahead).
5. Subtract simple transaction costs from turnover:
   - `net_return = gross_return - turnover * 5 bps`

## How to run

From project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python regression_analysis/run_regression_analysis.py
python model_development/run_strategy_development.py
```

## Main outputs

Regression:

- `outputs/regression_dataset.csv`
- `outputs/mlr_results.csv`
- `outputs/best_models.csv`
- `outputs/predictions_all_models.csv`

Strategy:

- `outputs/strategy_weights.csv`
- `outputs/strategy_backtest.csv`
- `outputs/strategy_performance_summary.csv`
- `outputs/strategy_summary.txt`

## Notes

- Data is pulled live, so results can vary by run date.
- If internet/data provider access fails, rerun once connectivity is available.
