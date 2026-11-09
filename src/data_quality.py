"""Market data loading, quality checks and remediation.

Mirrors a market-data control process: every check writes to an exception log,
and every remediation (proxy / fill) is recorded so the audit trail is complete.
"""
import numpy as np
import pandas as pd

from . import config as C


def load_raw() -> pd.DataFrame:
    """Load the long-format source file and pivot to one column per currency (units per USD)."""
    raw = pd.read_csv(C.RAW_FILE, parse_dates=["Date"])
    wide = raw.pivot(index="Date", columns="Country", values="Exchange rate")
    inv = {v: k for k, v in C.SOURCE_COLUMNS.items()}
    wide = wide.rename(columns=inv)[list(C.SOURCE_COLUMNS)]
    return wide.loc[C.START_DATE:].sort_index()


def run_checks(px: pd.DataFrame) -> pd.DataFrame:
    """Return an exception log with one row per data issue found."""
    log = []

    # 1. Missing values. Dates where every currency is blank are market holidays;
    #    isolated blanks are genuine missing fixings that need a proxy.
    all_blank = px.isna().all(axis=1)
    for d in px.index[all_blank]:
        log.append((d, "ALL", "HOLIDAY", "No fixings for any currency (non-trading day)"))
    for ccy in px:
        for d in px.index[px[ccy].isna() & ~all_blank]:
            log.append((d, ccy, "MISSING", "Fixing missing while other currencies published"))

    # 2. Non-positive prices
    for ccy in px:
        for d in px.index[px[ccy] <= 0]:
            log.append((d, ccy, "INVALID", f"Non-positive price {px.at[d, ccy]}"))

    # 3. Stale prices: same value repeated STALE_RUN_DAYS+ days
    for ccy in px:
        s = px[ccy].dropna()
        run_id = (s != s.shift()).cumsum()
        lengths = s.groupby(run_id).transform("size")
        stale = s[(lengths >= C.STALE_RUN_DAYS) & (run_id != run_id.shift())]
        for d in stale.index:
            log.append((d, ccy, "STALE", f"Price unchanged for {lengths[d]} days"))

    # 4. Outliers: robust z-score of each daily return vs the trailing 120-day
    #    median / MAD, so the check adapts to each currency's volatility regime.
    #    Flagged moves are sent for review, not deleted: genuine market shocks
    #    must stay in the VaR data set.
    rets = np.log(px.dropna(how="all").ffill()).diff()
    for ccy in rets:
        r = rets[ccy]
        med = r.rolling(120, min_periods=60).median().shift()
        mad = (r - med).abs().rolling(120, min_periods=60).median().shift() * 1.4826
        z = ((r - med) / mad.replace(0, np.nan)).dropna()
        for d in z.index[z.abs() > C.OUTLIER_Z]:
            log.append((d, ccy, "OUTLIER", f"Return {r[d]:.2%} (robust z={z[d]:.1f}) - review vs news"))

    # 5. Calendar gaps > 5 calendar days (possible feed outage)
    gaps = px.index.to_series().diff().dt.days
    for d in gaps.index[gaps > 5]:
        log.append((d, "ALL", "GAP", f"{int(gaps[d])} calendar days since last record"))

    out = pd.DataFrame(log, columns=["Date", "Currency", "Check", "Detail"])
    return out.sort_values(["Date", "Currency"]).reset_index(drop=True)


def remediate(px: pd.DataFrame):
    """Fill short gaps with the last valid price (proxy) and drop dates still incomplete.

    Returns cleaned prices and a remediation log.
    """
    # Rows where every currency is blank are non-trading days: drop them rather than
    # proxy, so no artificial zero-return days enter the risk model.
    all_blank = px.index[px.isna().all(axis=1)]
    rem = [(d, "ALL", "DROPPED", "Non-trading day (no fixings)") for d in all_blank]
    px = px.drop(index=all_blank)

    # Isolated missing fixings: proxy with last valid price, up to MAX_FFILL_DAYS.
    filled = px.ffill(limit=C.MAX_FFILL_DAYS)
    proxied = px.isna() & filled.notna()
    rem += [(d, c, "PROXY_FFILL", "Last valid price carried forward")
            for c in px for d in px.index[proxied[c]]]
    dropped = filled.index[filled.isna().any(axis=1)]
    rem += [(d, "ALL", "DROPPED", "Incomplete after remediation") for d in dropped]
    clean = filled.drop(index=dropped)
    rem_log = pd.DataFrame(rem, columns=["Date", "Currency", "Action", "Detail"])
    return clean, rem_log.sort_values("Date").reset_index(drop=True)


def to_inr_crosses(px_usd: pd.DataFrame) -> pd.DataFrame:
    """Convert 'units per USD' quotes into INR per 1 unit of each foreign currency."""
    inr = px_usd["INR"]
    out = pd.DataFrame(index=px_usd.index)
    out["USD"] = inr
    for ccy in px_usd.columns.drop("INR"):
        out[ccy] = inr / px_usd[ccy]
    return out[list(C.POSITIONS_CR)]
