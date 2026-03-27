from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

START_DATE = "2005-01-01"
ETF_TICKERS = ["SPY", "QQQ", "IWM"]
TARGET_HORIZON = 3  # months
MIN_TRAIN_PERIODS = 60 # To train our models
OUTPUT_DIR = Path("outputs")

FRED_SERIES = {
    "DGS10": "10y",
    "DGS2": "2y",
    "UNRATE": "unemployment",
    "CPIAUCSL": "cpi",
}

MODELS = {
    "ols": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=0.01, max_iter=5000),
    "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
    "huber": HuberRegressor(epsilon=1.5, max_iter=300),
    "quantile": QuantileRegressor(quantile=0.5, alpha=0.1, solver="highs"), #looking at the median
}


def fetch_prices() -> pd.DataFrame:
    prices = yf.download(ETF_TICKERS, start=START_DATE, auto_adjust=True, progress=False)["Close"]
    return prices.resample("ME").last()


def fetch_macro() -> pd.DataFrame:
    dfs = []
    for fred_id, name in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        df = pd.read_csv(url)
        df.columns = ["date", name]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        dfs.append(df)
    return pd.concat(dfs, axis=1, sort=False).resample("ME").last()


def engineered_features(macro: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    df = macro.copy()

    df["yield_curve"] = df["10y"] - df["2y"]
    df["cpi_yoy"] = df["cpi"].pct_change(12)
    df["rate_level"] = df["10y"]
    macro_cols = ["yield_curve", "cpi_yoy", "unemployment", "rate_level"]

    signal_parts = []
    for ticker in prices.columns:
        p = prices[ticker]
        s = pd.DataFrame(index=p.index)
        s[f"{ticker}_mom_1m"] = p.pct_change(1)
        s[f"{ticker}_mom_3m"] = p.pct_change(3)
        s[f"{ticker}_mom_12m"] = p.pct_change(12)
        s[f"{ticker}_vol_6m"] = p.pct_change(1).rolling(6).std()
        s[f"{ticker}_mean_rev_12m"] = (p / p.rolling(12).mean()) - 1
        signal_parts.append(s)

    price_df = pd.concat(signal_parts, axis=1)
    avg_signals = pd.DataFrame(
        {
            "mom_1m": price_df.filter(like="_mom_1m").mean(axis=1),
            "mom_3m": price_df.filter(like="_mom_3m").mean(axis=1),
            "mom_12m": price_df.filter(like="_mom_12m").mean(axis=1),
            "vol_6m": price_df.filter(like="_vol_6m").mean(axis=1),
            "mean_rev_12m": price_df.filter(like="_mean_rev_12m").mean(axis=1),
        }
    )

    combined = df[macro_cols].join(avg_signals)
    return combined.shift(1)


def model_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    prices = fetch_prices()
    macro = fetch_macro()

    X = engineered_features(macro, prices)

    # Forward 3-month log return targets
    Y = (np.log(prices.shift(-TARGET_HORIZON)) - np.log(prices)).add_prefix("y_")

    dataset = X.join(Y, how="inner").dropna()
    dataset.to_csv(OUTPUT_DIR / "regression_dataset.csv", index_label="date")

    target_cols = [f"y_{t}" for t in ETF_TICKERS]
    feature_cols = list(X.columns)

    all_predictions = pd.DataFrame(index=dataset.index)
    results = []

    for target in target_cols:
        Y = dataset[target]
        X = dataset[feature_cols]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=MIN_TRAIN_PERIODS, shuffle=False)
        split_idx = len(X_train)

        for model_name, model in MODELS.items():
            pred_full = pd.Series(index=dataset.index, dtype=float)

            # Simple expanding-window OOS prediction after initial train block.
            for i in range(split_idx, len(dataset)):
                X_roll_train = X.iloc[:i]
                Y_roll_train = Y.iloc[:i]
                X_now = X.iloc[[i]]

                pipe = model_pipeline(model)
                pipe.fit(X_roll_train, Y_roll_train)
                pred_full.iloc[i] = pipe.predict(X_now)[0]

            pred_col = f"pred__{target}__{model_name}"
            all_predictions[pred_col] = pred_full

            Y_pred = pred_full.iloc[split_idx:].dropna()
            Y_eval = Y.loc[Y_pred.index]

            if len(Y_eval) > 1:
                r2 = float(r2_score(Y_eval, Y_pred))
                rmse = float(np.sqrt(mean_squared_error(Y_eval, Y_pred)))
                dir_acc = float((np.sign(Y_eval) == np.sign(Y_pred)).mean())
                n_oos = int(len(Y_eval))

            results.append(
                {
                    "target": target,
                    "model": model_name,
                    "r2": r2,
                    "rmse": rmse,
                    "dir_acc": dir_acc,
                    "n_oos_periods": n_oos,
                }
            )

    results_df = pd.DataFrame(results).sort_values(["target", "r2"], ascending=[True, False])
    results_df.to_csv(OUTPUT_DIR / "mlr_results.csv", index=False)

    best_models = results_df.groupby("target", as_index=False).first()
    best_models = best_models.rename(columns={"model": "best_model"})
    best_models.to_csv(OUTPUT_DIR / "best_models.csv", index=False)

    all_predictions.to_csv(OUTPUT_DIR / "predictions_all_models.csv", index_label="date")

    print(results_df)


if __name__ == "__main__":
    main()
