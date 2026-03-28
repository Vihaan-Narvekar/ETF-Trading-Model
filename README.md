# Systematic and Macroeconomic Equity Long/Short Strategy Using Regression Models

1. Run regression models on macro + market features to predict forward ETF returns.
2. Convert those predictions into a simple, rules-based strategy backtest.

This project is centered around two folders:

- `regression_analysis/`: data collection, feature engineering, regression modeling
- `model_development/`: strategy rules and backtest
- `visualization/`: chart generation from saved outputs

## Project structure

- `regression_analysis/run_regression_analysis.py`
- `model_development/run_strategy_development.py`
- `visualization/create_visualizations.py`
- `outputs/`
- `requirements.txt`

## Data sources

- ETF prices: `yfinance` (SPY, QQQ, IWM)
- Macro factors: FRED CSV endpoints

## Models included

- `ols` (linear regression)
- `ridge`
- `lasso`
- `elasticnet`
- `huber`
- `quantile`
- Model selection metric: highest `dir_acc` (directional accuracy), tie-breaker = lower `rmse`

## Target and features

- Target: forward 3-month log return for each ETF (`y_SPY`, `y_QQQ`, `y_IWM`)
- Feature examples:
  - yield curve slope (`10Y - 2Y`)
  - CPI YoY
  - unemployment
  - rate level (10Y yield)

## Strategy logic

1. Use each ETF's best model from `outputs/best_models.csv`.
2. Signal = sign of predicted return.
3. Normalize to gross exposure = 1.
4. Shift weights by 1 period (no lookahead).
5. Subtract simple transaction costs from turnover:


## Main outputs

Regression:

- `outputs/regression_dataset.csv`
- `outputs/mlr_results.csv`
- `outputs/best_models.csv`
- `outputs/feature_importance.csv`
- `outputs/feature_importance_best_models.csv`
- `outputs/predictions_all_models.csv`

Strategy:
- `outputs/strategy_weights.csv`
- `outputs/strategy_backtest.csv`
- `outputs/strategy_performance_summary.csv`
- `outputs/strategy_summary.txt`

Visualizations:
- `outputs/charts/model_directional_accuracy.png`
- `outputs/charts/best_model_realized_vs_predicted.png`
- `outputs/charts/model_quintile_return_spread.png`
- `outputs/charts/strategy_equity_drawdown_vs_benchmark.png`
- `outputs/charts/strategy_return_histograms_gross_vs_net.png`

## Run order

```bash
python regression_analysis/run_regression_analysis.py
python model_development/run_strategy_development.py
python visualization/create_visualizations.py
```


Performance (3/27/2026)
 - ann_return: 0.4009. 
 - ann_vol: 0.2482. 
 - sharpe: 1.4416. 
 - max_drawdown: -0.5727. 
 - hit_rate: 0.7288. 
 - n_periods: 177. 
