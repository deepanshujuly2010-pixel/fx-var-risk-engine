"""VaR and Expected Shortfall models.

All functions take a returns matrix (T x N, simple daily returns) and an exposure
vector (INR crore) and return a positive loss number in INR crore.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C


def pnl_series(rets: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Linear P&L of the book for each historical return vector."""
    return rets @ w


def historical(rets, w, conf=C.VAR_CONF, es_conf=C.ES_CONF):
    pnl = pnl_series(rets, w).values
    var = -np.quantile(pnl, 1 - conf)
    cut = np.quantile(pnl, 1 - es_conf)
    es = -pnl[pnl <= cut].mean()
    return var, es


def parametric(rets, w, conf=C.VAR_CONF, es_conf=C.ES_CONF, cov=None):
    """Variance-covariance (delta-normal) VaR, zero-mean assumption."""
    cov = rets.cov().values if cov is None else cov
    sigma = float(np.sqrt(w @ cov @ w))
    var = norm.ppf(conf) * sigma
    es = sigma * norm.pdf(norm.ppf(es_conf)) / (1 - es_conf)
    return var, es, sigma


def ewma_cov(rets: pd.DataFrame, lam=C.EWMA_LAMBDA) -> np.ndarray:
    """RiskMetrics exponentially-weighted covariance (more weight on recent data)."""
    x = rets.values - 0.0
    n = len(x)
    weights = (1 - lam) * lam ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()
    return (x * weights[:, None]).T @ x


def ewma(rets, w, conf=C.VAR_CONF, es_conf=C.ES_CONF):
    return parametric(rets, w, conf, es_conf, cov=ewma_cov(rets))[:2]


def monte_carlo(rets, w, conf=C.VAR_CONF, es_conf=C.ES_CONF, n=C.MC_SIMS, seed=C.SEED, dof=None):
    """Correlated simulation via Cholesky. dof=None -> normal, else Student-t (fat tails)."""
    rng = np.random.default_rng(seed)
    cov = rets.cov().values
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, len(w)))
    if dof:
        # scale t-draws to unit variance so the covariance is preserved
        chi = rng.chisquare(dof, size=(n, 1))
        z = z / np.sqrt(chi / dof) * np.sqrt((dof - 2) / dof)
    sims = z @ L.T @ w
    var = -np.quantile(sims, 1 - conf)
    es = -sims[sims <= np.quantile(sims, 1 - es_conf)].mean()
    return var, es


def component_var(rets, w, conf=C.VAR_CONF) -> pd.DataFrame:
    """Euler allocation of parametric VaR: contribution of each currency (sums to total)."""
    cov = rets.cov().values
    sigma = np.sqrt(w @ cov @ w)
    marginal = norm.ppf(conf) * (cov @ w) / sigma
    comp = w * marginal
    standalone = norm.ppf(conf) * np.abs(w) * np.sqrt(np.diag(cov))
    return pd.DataFrame({
        "Exposure (INR cr)": w,
        "Standalone VaR": standalone,
        "Marginal VaR": marginal,
        "Component VaR": comp,
        "% of Total": comp / comp.sum(),
    }, index=rets.columns)


def scale_horizon(x, days=C.HORIZON_DAYS):
    return x * np.sqrt(days)
