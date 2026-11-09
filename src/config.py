"""Central configuration: portfolio, model parameters and scenario definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "fx_rates_per_usd.csv"
OUTPUT = ROOT / "output"
CHARTS = OUTPUT / "charts"

# Source quotes every currency as units per 1 USD (FRED H.10 convention).
# Map portfolio currency code -> source column.
SOURCE_COLUMNS = {
    "INR": "India", "EUR": "Euro", "GBP": "United Kingdom", "JPY": "Japan",
    "AUD": "Australia", "CHF": "Switzerland", "CNY": "China", "SGD": "Singapore",
}
BASE_CCY = "INR"

# Hypothetical FX book of an Indian bank: INR crore exposure per currency.
# Positive = long the foreign currency vs INR, negative = short.
POSITIONS_CR = {
    "USD": 40.0, "EUR": 25.0, "GBP": 15.0, "JPY": -10.0,
    "CHF": 8.0, "AUD": 7.0, "CNY": 5.0, "SGD": 5.0,
}

START_DATE = "2012-01-01"

# Risk model parameters (Basel-style)
VAR_CONF = 0.99          # 1-day 99% VaR
ES_CONF = 0.975          # 97.5% Expected Shortfall (FRTB)
WINDOW = 250             # look-back window in trading days
HORIZON_DAYS = 10        # regulatory holding period (sqrt-of-time scaling)
EWMA_LAMBDA = 0.94       # RiskMetrics decay factor
MC_SIMS = 20_000
SEED = 42
BACKTEST_DAYS = 1000     # length of out-of-sample backtest

# Data-quality thresholds
MAX_FFILL_DAYS = 3       # longest gap we proxy with last valid price
STALE_RUN_DAYS = 5       # identical prices for this many days => stale flag
OUTLIER_Z = 6.0          # robust z-score on daily returns

# Historical stress windows (start, end) — worst cumulative 10-day move is applied
STRESS_WINDOWS = {
    "Taper Tantrum (2013)": ("2013-05-20", "2013-09-30"),
    "Brexit Referendum (2016)": ("2016-06-20", "2016-07-15"),
    "COVID-19 Shock (2020)": ("2020-02-20", "2020-04-30"),
    "Russia-Ukraine (2022)": ("2022-02-15", "2022-03-31"),
    "Global Rate Hikes (2022)": ("2022-09-01", "2022-10-31"),
}
# Hypothetical shocks: % change in INR per unit of foreign currency
HYPOTHETICAL_SHOCKS = {
    "INR rallies 5% vs all": {c: -0.05 for c in POSITIONS_CR},
    "INR falls 5% vs all": {c: 0.05 for c in POSITIONS_CR},
    "JPY +10% safe-haven bid": {"JPY": 0.10, "CHF": 0.06},
    "EUR & GBP -8% (Europe stress)": {"EUR": -0.08, "GBP": -0.08},
}
