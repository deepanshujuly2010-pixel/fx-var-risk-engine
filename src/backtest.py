"""Out-of-sample VaR backtesting: Kupiec POF, Christoffersen independence, Basel traffic light."""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from . import config as C
from . import var_models as M


def rolling_forecasts(rets: pd.DataFrame, w: np.ndarray, days=C.BACKTEST_DAYS, window=C.WINDOW):
    """For each day t, estimate VaR on [t-window, t) and compare with realised P&L on day t."""
    rows = []
    start = len(rets) - days
    for t in range(start, len(rets)):
        hist = rets.iloc[t - window:t]
        realised = float(rets.iloc[t].values @ w)
        rows.append({
            "Date": rets.index[t],
            "PnL": realised,
            "Historical": M.historical(hist, w)[0],
            "Parametric": M.parametric(hist, w)[0],
            "EWMA": M.ewma(hist, w)[0],
        })
    return pd.DataFrame(rows).set_index("Date")


def kupiec_pof(n_obs, n_exc, p=1 - C.VAR_CONF):
    """Proportion-of-failures likelihood ratio test. H0: exception rate == p."""
    x, n = n_exc, n_obs
    phat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return lr, 1 - chi2.cdf(lr, 1)


def christoffersen_ind(hits: np.ndarray):
    """Independence test: are exceptions clustered? H0: exceptions independent."""
    h = hits.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(h[:-1], h[1:]):
        n00 += (a == 0) & (b == 0); n01 += (a == 0) & (b == 1)
        n10 += (a == 1) & (b == 0); n11 += (a == 1) & (b == 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b):
        return (a * np.log(1 - p) if a and p < 1 else 0) + (b * np.log(p) if b and p > 0 else 0)
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return lr, 1 - chi2.cdf(lr, 1)


def traffic_light(exceptions_250: int) -> str:
    """Basel Committee zones for 250 observations at 99%."""
    if exceptions_250 <= 4:
        return "GREEN"
    if exceptions_250 <= 9:
        return "AMBER"
    return "RED"


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["Historical", "Parametric", "EWMA"]:
        hits = (bt["PnL"] < -bt[model]).values
        n, x = len(hits), int(hits.sum())
        lr_pof, p_pof = kupiec_pof(n, x)
        lr_ind, p_ind = christoffersen_ind(hits)
        last250 = int(hits[-250:].sum())
        rows.append({
            "Model": model, "Observations": n, "Exceptions": x,
            "Expected": round(n * (1 - C.VAR_CONF), 1),
            "Exception Rate": x / n,
            "Kupiec LR": lr_pof, "Kupiec p-value": p_pof,
            "Kupiec Result": "Accept" if p_pof > 0.05 else "Reject",
            "Christoffersen LR": lr_ind, "Christoffersen p-value": p_ind,
            "Independence Result": "Accept" if p_ind > 0.05 else "Reject",
            "Exceptions (last 250d)": last250,
            "Basel Zone": traffic_light(last250),
        })
    return pd.DataFrame(rows).set_index("Model")
