"""Historical and hypothetical stress testing of the current book."""
import numpy as np
import pandas as pd

from . import config as C


def historical_stress(prices: pd.DataFrame, w: pd.Series, horizon=C.HORIZON_DAYS) -> pd.DataFrame:
    """Apply the worst 10-day move inside each stress window to today's positions."""
    rows = []
    for name, (s, e) in C.STRESS_WINDOWS.items():
        px = prices.loc[s:e]
        moves = px.pct_change(horizon).dropna()
        pnl = moves @ w
        worst = pnl.idxmin()
        rows.append({
            "Scenario": name, "Type": "Historical",
            "Worst 10d window ends": worst.strftime("%Y-%m-%d"),
            "P&L (INR cr)": pnl.min(),
            "Biggest driver": (moves.loc[worst] * w).idxmin(),
        })
    return pd.DataFrame(rows)


def hypothetical_stress(w: pd.Series) -> pd.DataFrame:
    rows = []
    for name, shocks in C.HYPOTHETICAL_SHOCKS.items():
        s = pd.Series(shocks).reindex(w.index).fillna(0.0)
        contrib = s * w
        rows.append({
            "Scenario": name, "Type": "Hypothetical",
            "Worst 10d window ends": "-",
            "P&L (INR cr)": contrib.sum(),
            "Biggest driver": contrib.idxmin() if contrib.min() < 0 else contrib.idxmax(),
        })
    return pd.DataFrame(rows)
