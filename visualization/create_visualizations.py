from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs")
CHART_DIR = OUTPUT_DIR / "charts"


def load_inputs() -> dict[str, pd.DataFrame]:
    data = {
        "results": pd.read_csv(OUTPUT_DIR / "mlr_results.csv"),
        "best_models": pd.read_csv(OUTPUT_DIR / "best_models.csv"),
        "predictions": pd.read_csv(OUTPUT_DIR / "predictions_all_models.csv", index_col=0, parse_dates=True),
        "dataset": pd.read_csv(OUTPUT_DIR / "regression_dataset.csv", index_col=0, parse_dates=True),
        "backtest": pd.read_csv(OUTPUT_DIR / "strategy_backtest.csv", index_col=0, parse_dates=True),
    }
    return data


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.size"] = 10


def chart_model_directional_accuracy(results: pd.DataFrame) -> None:
    df = results.copy()
    df["ticker"] = df["target"].str.replace("y_", "", regex=False)
    pivot = df.pivot(index="model", columns="ticker", values="dir_acc")
    tickers = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(tickers))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, ticker in enumerate(tickers):
        ax.bar(x + offsets[i], pivot[ticker], width=width, label=ticker, color=colors[i % len(colors)])

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylim(0.35, 0.9)
    ax.set_title("Directional Accuracy by Model and ETF")
    ax.set_ylabel("Directional Accuracy")
    ax.set_xlabel("Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "model_directional_accuracy.png")
    plt.close()


def chart_best_model_pred_vs_realized(
    best_models: pd.DataFrame, predictions: pd.DataFrame, dataset: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    if len(best_models) < 3:
        axes = np.atleast_1d(axes)

    for ax, (_, row) in zip(axes, best_models.iterrows()):
        target = row["target"]
        model = row["best_model"]
        pred_col = f"pred__{target}__{model}"

        y_true = dataset[target]
        y_pred = predictions[pred_col]
        mask = y_pred.notna()

        ax.plot(y_true.loc[mask].index, y_true.loc[mask].values, label="Realized", color="#1f77b4", linewidth=1.2)
        ax.plot(y_pred.loc[mask].index, y_pred.loc[mask].values, label="Predicted", color="#d62728", linewidth=1.2)
        ax.set_title(f"{target.replace('y_', '')}: Best Model ({model})")
        ax.set_ylabel("3M Log Return")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "best_model_realized_vs_predicted.png")
    plt.close()


def chart_prediction_quintile_spread(
    best_models: pd.DataFrame, predictions: pd.DataFrame, dataset: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, (_, row) in zip(axes, best_models.iterrows()):
        target = row["target"]
        model = row["best_model"]
        pred_col = f"pred__{target}__{model}"

        tmp = pd.DataFrame({"pred": predictions[pred_col], "realized": dataset[target]}).dropna()
        if len(tmp) < 20:
            continue

        tmp["bucket"] = pd.qcut(tmp["pred"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        bucket_ret = tmp.groupby("bucket", observed=True)["realized"].mean()

        colors = ["#b2182b", "#ef8a62", "#fddbc7", "#67a9cf", "#2166ac"]
        ax.bar(bucket_ret.index.astype(str), bucket_ret.values, color=colors)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{target.replace('y_', '')}: Realized Return by Predicted Quintile")
        ax.set_xlabel("Predicted Return Quintile")
        ax.set_ylabel("Avg Realized 3M Return")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "model_quintile_return_spread.png")
    plt.close()


def chart_strategy_equity_and_drawdown(backtest: pd.DataFrame, dataset: pd.DataFrame) -> None:
    bt = backtest.copy()
    bt = bt.loc[bt["active"] == 1].copy()

    y_cols = [c for c in dataset.columns if c.startswith("y_")]
    benchmark = dataset[y_cols].mean(axis=1).rename("benchmark_return")
    idx = bt.index.intersection(benchmark.index)

    bt = bt.loc[idx]
    benchmark = benchmark.loc[idx]

    strategy_equity = (1 + bt["net_return"]).cumprod()
    benchmark_equity = (1 + benchmark).cumprod()

    strategy_dd = strategy_equity / strategy_equity.cummax() - 1
    benchmark_dd = benchmark_equity / benchmark_equity.cummax() - 1

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(strategy_equity.index, strategy_equity.values, label="Strategy (Net)", color="#1f77b4")
    axes[0].plot(benchmark_equity.index, benchmark_equity.values, label="Equal-Weight ETF Benchmark", color="#ff7f0e")
    axes[0].set_title("Equity Curve: Strategy vs Benchmark")
    axes[0].set_ylabel("Equity (Start = 1)")
    axes[0].legend(loc="upper left")

    axes[1].plot(strategy_dd.index, strategy_dd.values, label="Strategy Drawdown", color="#d62728")
    axes[1].plot(benchmark_dd.index, benchmark_dd.values, label="Benchmark Drawdown", color="#9467bd")
    axes[1].set_title("Drawdown Comparison")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "strategy_equity_drawdown_vs_benchmark.png")
    plt.close()


def chart_strategy_return_histograms(backtest: pd.DataFrame) -> None:
    bt = backtest.loc[backtest["active"] == 1].copy()
    gross = bt["gross_return"].dropna()
    net = bt["net_return"].dropna()

    bins = np.linspace(min(gross.min(), net.min()), max(gross.max(), net.max()), 25)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].hist(gross, bins=bins, color="#17becf", alpha=0.85, edgecolor="white")
    axes[0].axvline(gross.mean(), color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Gross Return Distribution")
    axes[0].set_xlabel("Monthly Gross Return")
    axes[0].set_ylabel("Count")

    axes[1].hist(net, bins=bins, color="#2ca02c", alpha=0.85, edgecolor="white")
    axes[1].axvline(net.mean(), color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Net Return Distribution (After Costs)")
    axes[1].set_xlabel("Monthly Net Return")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "strategy_return_histograms_gross_vs_net.png")
    plt.close()


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    style()

    data = load_inputs()

    chart_model_directional_accuracy(data["results"])
    chart_best_model_pred_vs_realized(data["best_models"], data["predictions"], data["dataset"])
    chart_prediction_quintile_spread(data["best_models"], data["predictions"], data["dataset"])
    chart_strategy_equity_and_drawdown(data["backtest"], data["dataset"])
    chart_strategy_return_histograms(data["backtest"])

    print("Saved charts to outputs/charts/")
    print("- model_directional_accuracy.png")
    print("- best_model_realized_vs_predicted.png")
    print("- model_quintile_return_spread.png")
    print("- strategy_equity_drawdown_vs_benchmark.png")
    print("- strategy_return_histograms_gross_vs_net.png")


if __name__ == "__main__":
    main()
