import re
import yfinance as yf
import talib
import pandas as pd
import numpy as np
import copy
from datetime import datetime

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# TA-Lib mapping
talib_functions = {
    'rsi':    {'func': talib.RSI,       'inputs': ['close']},
    'ema':    {'func': talib.EMA,       'inputs': ['close']},
    'sma':    {'func': talib.SMA,       'inputs': ['close']},
    'macd':   {'func': talib.MACD,      'inputs': ['close']},
    'signal': {'func': lambda x: talib.MACD(x)[1], 'inputs': ['close']},
    'adx':    {'func': talib.ADX,       'inputs': ['high','low','close']},
    'linreg': {'func': talib.LINEARREG, 'inputs': ['close']},
}

class StrategyParser:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.indicators_used = []  # To store indicators used in strategy

    def _parse_indicator(self, token: str) -> str:
        parts = [p.strip() for p in token.lower().split(',') if p.strip()]
        ind = parts[0]
        if ind not in talib_functions:
            raise ValueError(f"Unknown indicator: {ind}")
        cfg = talib_functions[ind]
        
        expected_inputs = len(cfg['inputs'])
        if len(parts) < 1 + expected_inputs:
            raise ValueError(f"Insufficient arguments for {ind}. Got: {parts}")

        # Build columns
        cols = [f'self.df["{parts[i+1].capitalize()}"]' for i in range(expected_inputs)]

        # Get any extra args
        extra = parts[1 + expected_inputs:]
        args = cols + extra

        arg_str = ', '.join(args)
        fn = cfg['func']
        
        if hasattr(fn, '__name__') and fn.__name__.isupper():
            return f'talib.{fn.__name__}({arg_str})'
        else:
            return f'talib_functions["{ind}"]["func"]({arg_str})'

    def parse_logic(self, line: str) -> str:
        toks = line.replace('(', ' ( ').replace(')', ' ) ').split()
        out = []
        for t in toks:
            if ',' in t and not t.isdigit():
                # Extract indicator and add it to the list
                self.indicators_used.append(t.split(',')[0])
                out.append(self._parse_indicator(t))
            elif t.lower() in ('and','or'):
                out.append('&' if t.lower()=='and' else '|')
            else:
                out.append(t)
        return ' '.join(out)

    def evaluate(self, strategy: str) -> pd.Series:
        logic = self.parse_logic(strategy)
        parts = re.split(r'([&|])', logic)
        grouped = []
        for p in parts:
            if p in ('&','|'):
                grouped.append(f' {p} ')
            else:
                grouped.append(f'({p.strip()})')
        return eval(''.join(grouped))

    def extract_indicator_tokens(self, line: str) -> list:
        return [t for t in line.replace('(', ' ').replace(')', ' ').split() if ',' in t and not t.isdigit()]


def precompute_indicators(df: pd.DataFrame, indicator_tokens: list, parser: StrategyParser):
    indicator_columns = []
    for token in indicator_tokens:
        parts = token.lower().split(',')
        ind = parts[0]
        if ind not in talib_functions:
            continue
        cfg = talib_functions[ind]
        input_cols = [df[col.capitalize()] for col in parts[1:1+len(cfg['inputs'])]]
        extra = [int(x) for x in parts[1+len(cfg['inputs']):] if x.strip()]

        args = input_cols + extra
        try:
            values = cfg['func'](*args)
        except Exception as e:
            print(f"Error computing {token}: {e}")
            continue
        col_name = f"{ind.upper()}_" + "_".join(parts[1:])
        df[col_name] = values
        indicator_columns.append(col_name)

    # After processing indicators, assign them a unique name
    indicator_mapping = {col: f"indicator_{i+1}" for i, col in enumerate(indicator_columns)}
    df.rename(columns=indicator_mapping, inplace=True)
    parser.indicators_used = [indicator_mapping.get(col, col) for col in parser.indicators_used]

    return df


# The backtesting function remains unchanged
def symbols_backtesting_long_only(symbol_list, period, interval, entry_str, exit_str):
    all_trades = []
    all_raw_data = []  # New: to collect raw data
    buffer = max(50 for _ in talib_functions.values())

    for sym in symbol_list:
        df = (
            yf.Ticker(f"{sym}.NS")
              .history(period=period, interval=interval, auto_adjust=True)
              .round({"Open":2,"High":2,"Low":2,"Close":2})
              .assign(Volume=lambda x: x.Volume.astype(int))
        )
        if df.empty:
            continue

        df["Symbol"] = sym  # Add stock identifier
        all_raw_data.append(df.copy())  # Save raw version before modifying

        parser = StrategyParser(df)
        entry_tokens = parser.extract_indicator_tokens(entry_str)
        df = precompute_indicators(df, entry_tokens, parser)

        entry_sig = parser.evaluate(entry_str)
        exit_sig  = parser.evaluate(exit_str)

        position = False
        trade = {}

        for i in range(buffer, len(df)):
            price = df.Close.iloc[i]
            date  = df.index[i]

            # Exit
            if position and exit_sig.iloc[i]:
                trade.update({
                    "Exit": price,
                    "Exit Date": date,
                })
                all_trades.append(trade.copy())
                position = False
                trade = {}

            # Entry
            if not position and entry_sig.iloc[i]:
                position = True
                trade = {
                    "Symbol": sym,
                    "Entry": price,
                    "Entry Date": date,
                    "Entry Volume": df.Volume.iloc[i]
                }
                for token in entry_tokens:
                    col = f"{token.split(',')[0].upper()}_" + "_".join(token.split(',')[1:])
                    if col in df.columns:
                        trade[col] = df[col].iloc[i]

        if position:
            trade.update({
                "Exit": df.Close.iloc[-1],
                "Exit Date": df.index[-1]
            })
            all_trades.append(trade.copy())

    # New: return raw data also
    return all_trades, all_raw_data


# Call the backtest function and show the summary as before



def display_possible_combinations():
    periods  = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    intervals= ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"]
    print("Periods :", ", ".join(periods))
    print("Intervals:", ", ".join(intervals))

if __name__ == "__main__":
    display_possible_combinations()
    p     = input("Period (e.g. 6mo, 1y): ")
    intrv = input("Interval (e.g. 1d, 1h): ")
    entry = input("Long entry rule:\n")
    exit_ = input("Long exit rule:\n")

    symbols = [
      "HDFCBANK","RELIANCE", "SBIN","TCS" ,"WIPRO",
  "ADANIPORTS",
  "APOLLOHOSP",
  "ASIANPAINT",
  "AXISBANK",
  "BAJAJ-AUTO",
  "BAJAJFINSV",
  "BAJFINANCE",
  "BHARTIARTL",
  "BPCL",
  "BRITANNIA",
  "CIPLA",
  "COALINDIA",
  "DIVISLAB",
  "DRREDDY",
  "EICHERMOT"]

    print("\nRunning long‐only backtest…")
    trades, raw_data_list = symbols_backtesting_long_only(symbols, p, intrv, entry, exit_)

    if not trades:
        print("No trades generated.")
        exit()

        
    # Save raw data
    raw_df = pd.concat(raw_data_list).reset_index()
    raw_df.to_csv("raw_data.csv", index=False)
    print("Raw historical data saved to 'raw_data.csv'.")

    df = pd.DataFrame(trades)
    df["P/L"] = (df.Exit - df.Entry) / df.Entry * 100
    df["Abs Profit"] = df["Exit"] - df["Entry"]
    df["Abs Cumulative Profit"] = df["Abs Profit"].cumsum()
    df["Return"]   = df["P/L"].cumsum()
    df["Drawdown"] = df["Return"] - df["Return"].cummax()
    df["Holding Period"] = (df["Exit Date"] - df["Entry Date"]).dt.days

    # Performance metrics
    total_trades = len(df)
    total_pnl    = df["P/L"].sum()
    total_abs_profit = df["Abs Profit"].sum()
    avg_pnl      = df["P/L"].mean()
    wins         = df[df["P/L"]>0]
    losses       = df[df["P/L"]<=0]
    win_rate     = len(wins)/total_trades*100
    largest_win  = df["P/L"].max()
    largest_loss = df["P/L"].min()
    avg_hold     = df["Holding Period"].mean()
    med_hold     = df["Holding Period"].median()

    gross_profit = wins["P/L"].sum()
    gross_loss   = -losses["P/L"].sum()
    profit_factor= gross_profit/gross_loss if gross_loss>0 else np.nan

    w = win_rate/100
    avg_win = wins["P/L"].mean()   if len(wins)   else 0
    avg_loss= -losses["P/L"].mean() if len(losses) else 0
    expectancy = w*avg_win - (1-w)*avg_loss

    final_ret   = df["Return"].iloc[-1]
    start_dt    = df["Entry Date"].min()
    end_dt      = df["Exit Date"].max()
    years       = (end_dt - start_dt).days / 365.25
    cagr        = (1 + final_ret/100)**(1/years) - 1 if years>0 else np.nan
    vol         = df["P/L"].std() * np.sqrt(total_trades/years) if years>0 else np.nan
    sharpe      = cagr/vol if vol>0 else np.nan
    recovery    = cagr/abs(df["Drawdown"].min()/100) if df["Drawdown"].min()<0 else np.nan

    max_consec_wins  = max((df["P/L"]>0).groupby((df["P/L"]<=0).cumsum()).sum())
    max_consec_losses= max((df["P/L"]<=0).groupby((df["P/L"]>0).cumsum()).sum())

    gross_profit     = wins["P/L"].sum()
    gross_loss       = -losses["P/L"].sum()
    total_abs_pnl    = gross_profit + gross_loss
    profit_share_pct = gross_profit / total_abs_pnl * 100 if total_abs_pnl else 0
    loss_share_pct   = gross_loss   / total_abs_pnl * 100 if total_abs_pnl else 0

    print(f"Gross Profit      : {gross_profit:.2f}% ({profit_share_pct:.2f}% of total P/L magnitude)")
    print(f"Gross Loss        : -{gross_loss:.2f}% ({loss_share_pct:.2f}% of total P/L magnitude)")
    print(f"\n--- BACKTEST SUMMARY ---")
    print(f"Total Trades           : {total_trades}")
    print(f"Total P/L (%)          : {total_pnl:.2f}")
    print(f"Avg P/L per Trade (%)  : {avg_pnl:.2f}")
    print(f"Win Rate (%)           : {win_rate:.2f}")
    print(f"Largest Win (%)        : {largest_win:.2f}")
    print(f"Largest Loss (%)       : {largest_loss:.2f}")
    print(f"Avg Hold (days)        : {avg_hold:.1f}")
    print(f"Median Hold (days)     : {med_hold}")
    print(f"Profit Factor          : {profit_factor:.2f}")
    print(f"Expectancy (%)         : {expectancy:.2f}")
    print(f"CAGR (%)               : {cagr*100:.2f}")
    print(f"Sharpe Ratio           : {sharpe:.2f}")
    print(f"Recovery Factor        : {recovery:.2f}")
    print(f"Max Drawdown (%)       : {df['Drawdown'].min():.2f}")
    print(f"Max Consecutive Wins   : {max_consec_wins}")
    print(f"Max Consecutive Losses : {max_consec_losses}")
    print(f"Total Absolute Profit  : ₹{total_abs_profit:,.2f}")

    round_cols = ["Entry", "Exit", "P/L", "Abs Profit", "Abs Cumulative Profit", "Return", "Drawdown"]
    df[round_cols] = df[round_cols].round(2)

    df.to_csv("result.csv", index=False)
    print("\nDetailed trades saved to 'result.csv'.")
