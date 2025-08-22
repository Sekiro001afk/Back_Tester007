Indicator‑driven backtesting with trade logging and supervised models for pre‑entry outcome prediction.

Features

Strategy parser to precompute TA‑Lib indicators

Multi‑symbol long‑only backtest (yfinance OHLCV)

Trade log with P/L, cumulative return, drawdown, and holding period

Classifiers (Random Forest, XGBoost, Logistic Regression, SVC, Gradient Boosting) on pre‑entry features

Repository structure

indicators.py — fetches data, computes indicators, runs backtests, writes CSVs

model.py — trains/evaluates classifiers on result.csv, optional export to models/

requirements.txt — runtime dependencies

.gitignore — excludes generated artifacts and local environments

models/ (ignored) — serialized models (*.pkl)

output/ (ignored) — generated reports/plots

raw_data.csv, result.csv (ignored) — data and trade logs

Quickstart

Install
pip install -r requirements.txt

Backtest (interactive)
python indicators.py
Example prompts:

Period: 3y

Interval: 1d

Entry: rsi,close,14 < 30 and ema,close,50 > ema,close,200

Exit: rsi,close,14 > 60 or ema,close,50 < ema,close,200
Outputs: raw_data.csv, result.csv

Train models
python model.py -i result.csv -s
Prints metrics; saves models/*.pkl (ignored by git)

Data schema (result.csv)

Symbol, Entry, Entry Date, Entry Volume

Exit, Exit Date, Holding Period (days)

P/L (%), Abs Profit, Abs Cumulative Profit

Return (cumulative %), Drawdown (cumulative %)

Indicator snapshot columns (if present) used for modeling

Notes

TA‑Lib is provided via talib-binary on Windows. If a system TA‑Lib is present, TA-Lib also works.

Serialized models are Python/library‑specific. For portability, rebuild via model.py on the target environment.

Roadmap

Position sizing and fees

Slippage modeling

Walk‑forward evaluation and cross‑validation by time splits

Contributing

Keep generated data/models out of version control.

Small, focused pull requests are preferred.
