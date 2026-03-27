from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")
TRANSACTION_COST_BPS = 5


def strategy_stats(net_returns: pd.Series, annualization: int = 12) -> dict[str, float]:
    r = net_returns.dropna()
    if r.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "n_periods": 0,
        }

    ann_return = (1 + r).prod() ** (annualization / len(r)) - 1
    ann_vol = r.std() * np.sqrt(annualization)
    sharpe = (ann_return - 0.043) / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((r > 0).mean()),
        "n_periods": int(len(r)),
    }


def main() -> None:
    best_models = pd.read_csv(OUTPUT_DIR / "best_models.csv")
    all_predictions = pd.read_csv(OUTPUT_DIR / "predictions_all_models.csv", index_col=0, parse_dates=True)
    regression_dataset = pd.read_csv(OUTPUT_DIR / "regression_dataset.csv", index_col=0, parse_dates=True)

    # Changing to consistent naming
    chosen_predictions = {}
    for _, row in best_models.iterrows():
        target = row["target"]
        model = row["best_model"]
        pred_col = f"pred__{target}__{model}"

        if pred_col in all_predictions.columns and target in regression_dataset.columns:
            etf = target.replace("y_", "")
            chosen_predictions[etf] = all_predictions[pred_col]

    pred = pd.DataFrame(chosen_predictions).dropna(how="all")
    if pred.empty:
        raise ValueError("No matching prediction columns found. Run regression first.")

    realized_cols = [f"y_{etf}" for etf in pred.columns]
    realized = regression_dataset[realized_cols].copy()
    realized.columns = [c.replace("y_", "") for c in realized.columns]

    idx = pred.index.intersection(realized.index)
    pred = pred.loc[idx]
    realized = realized.loc[idx]

    # Strategy: Direction from sign(prediction), Normalize to gross exposure = 1
    # Use magnitude instead of just sign
    raw_signal = pred.copy()
    # Normalize to get weights
    gross_exposure = raw_signal.abs().sum(axis=1).replace(0, np.nan)
    weights = raw_signal.div(gross_exposure, axis=0).fillna(0.0)

    # Preventing Lookahead Bias
    weights = weights.shift(1).fillna(0.0)

    gross_return = (weights * realized).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    transaction_cost = turnover * (TRANSACTION_COST_BPS / 10000.0)
    net_return = gross_return - transaction_cost

    active = weights.abs().sum(axis=1) > 0

    backtest = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "active": active.astype(int),
        }
    )

    summary = strategy_stats(backtest.loc[backtest["active"] == 1, "net_return"])
    summary_df = pd.DataFrame([summary]).round(4)

    weights.round(4).to_csv(OUTPUT_DIR / "strategy_weights.csv")
    backtest.round(6).to_csv(OUTPUT_DIR / "strategy_backtest.csv")
    summary_df.to_csv(OUTPUT_DIR / "strategy_performance_summary.csv", index=False)

    report = [
        "Strategy Development Summary",
        "=" * 28,
        "",
        "Approach:",
        "1. For each ETF, use the best model from best_models.csv",
        "2. Signal = sign(predicted return)",
        "3. Normalize to gross exposure = 1",
        "4. Shift weights by 1 period (no lookahead)",
        f"5. Net return = gross return - turnover * {TRANSACTION_COST_BPS} bps",
        "",
        "Performance:",
        summary_df.to_string(index=False),
    ]
    (OUTPUT_DIR / "strategy_summary.txt").write_text("\n".join(report), encoding="utf-8")

    print("Strategy development complete")
    print("Saved: outputs/strategy_weights.csv")
    print("Saved: outputs/strategy_backtest.csv")
    print("Saved: outputs/strategy_performance_summary.csv")
    print("Saved: outputs/strategy_summary.txt")


if __name__ == "__main__":
    main()
