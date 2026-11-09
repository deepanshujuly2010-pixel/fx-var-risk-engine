"""End-to-end run: data quality -> VaR/ES -> backtest -> stress test -> Excel report + charts.

Usage:  python run.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import config as C
from src import data_quality as DQ
from src import var_models as M
from src import backtest as BT
from src import stress as ST

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
BLUE, ORANGE, GREEN, RED = "#1F4E8C", "#E07B39", "#2E8B57", "#C0392B"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "figure.dpi": 150, "axes.axisbelow": True})


def main():
    C.CHARTS.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. Market data & quality ----------------
    raw = DQ.load_raw()
    exceptions = DQ.run_checks(raw)
    clean_usd, remediation = DQ.remediate(raw)
    prices = DQ.to_inr_crosses(clean_usd)
    rets = prices.pct_change().dropna()
    w_s = pd.Series(C.POSITIONS_CR)[rets.columns]
    w = w_s.values

    dq_summary = (exceptions.groupby(["Check", "Currency"]).size().unstack(fill_value=0))
    print(f"Raw rows: {len(raw)} | Clean rows: {len(prices)} | DQ exceptions: {len(exceptions)}")

    # ---------------- 2. VaR / ES on latest window ----------------
    win = rets.iloc[-C.WINDOW:]
    asof = win.index[-1].date()
    hv, hes = M.historical(win, w)
    pv, pes, sigma = M.parametric(win, w)
    ev, ees = M.ewma(win, w)
    mv, mes = M.monte_carlo(win, w)
    tv, tes = M.monte_carlo(win, w, dof=4)
    var_tbl = pd.DataFrame({
        "1d VaR 99%": [hv, pv, ev, mv, tv],
        "1d ES 97.5%": [hes, pes, ees, mes, tes],
    }, index=["Historical Simulation", "Parametric (Delta-Normal)", "EWMA (RiskMetrics, λ=0.94)",
              "Monte Carlo (Normal)", "Monte Carlo (Student-t, ν=4)"])
    var_tbl["10d VaR 99%"] = M.scale_horizon(var_tbl["1d VaR 99%"])
    var_tbl["1d VaR % of Gross"] = var_tbl["1d VaR 99%"] / np.abs(w).sum()
    comp = M.component_var(win, w)
    corr = win.corr()
    print(f"\nAs of {asof} — gross exposure INR {np.abs(w).sum():.0f} cr\n", var_tbl.round(3))

    # ---------------- 3. Backtest ----------------
    bt = BT.rolling_forecasts(rets, w)
    bt_sum = BT.summarise(bt)
    print("\nBacktest\n", bt_sum[["Exceptions", "Expected", "Kupiec Result", "Independence Result", "Basel Zone"]])

    # ---------------- 4. Stress ----------------
    stress = pd.concat([ST.historical_stress(prices, w_s), ST.hypothetical_stress(w_s)], ignore_index=True)
    stress["Multiple of 10d VaR"] = -stress["P&L (INR cr)"] / var_tbl.loc["Historical Simulation", "10d VaR 99%"]
    print("\nStress\n", stress.round(2))

    # ---------------- 5. Charts ----------------
    charts(prices, rets, w, bt, comp, corr, stress, var_tbl, win)

    # ---------------- 6. Excel report ----------------
    out = C.OUTPUT / "FX_VaR_Risk_Report.xlsx"
    summary = pd.DataFrame({
        "Item": ["As-of date", "Base currency", "Gross exposure (INR cr)", "Net exposure (INR cr)",
                 "Look-back window (days)", "Backtest length (days)", "Raw data rows", "Clean data rows",
                 "DQ exceptions logged", "Remediation actions"],
        "Value": [str(asof), C.BASE_CCY, np.abs(w).sum(), w.sum(), C.WINDOW, C.BACKTEST_DAYS,
                  len(raw), len(prices), len(exceptions), len(remediation)],
    })
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        summary.to_excel(xw, "Summary", index=False)
        var_tbl.to_excel(xw, "VaR_ES", index_label="Model")
        comp.to_excel(xw, "Component_VaR", index_label="Currency")
        corr.to_excel(xw, "Correlation", index_label="Currency")
        bt_sum.to_excel(xw, "Backtest_Summary")
        bt.assign(**{f"{m} Exception": bt["PnL"] < -bt[m] for m in ["Historical", "Parametric", "EWMA"]}) \
          .to_excel(xw, "Backtest_Daily")
        stress.to_excel(xw, "Stress_Tests", index=False)
        dq_summary.to_excel(xw, "DQ_Summary")
        exceptions.to_excel(xw, "DQ_Exception_Log", index=False)
        remediation.to_excel(xw, "Remediation_Log", index=False)
        prices.to_excel(xw, "Clean_Prices_INR")
    format_workbook(out)
    print(f"\nReport written to {out}")


def charts(prices, rets, w, bt, comp, corr, stress, var_tbl, win):
    # 1. Backtest: P&L vs VaR
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(bt.index, bt["PnL"], color=np.where(bt["PnL"] < -bt["Historical"], RED, "#9ca3af"), width=1.5, label="Daily P&L")
    ax.plot(bt.index, -bt["Historical"], color=BLUE, lw=1.4, label="Historical VaR 99%")
    ax.plot(bt.index, -bt["EWMA"], color=ORANGE, lw=1.1, label="EWMA VaR 99%")
    ax.set_title("VaR backtest — daily P&L vs 1-day 99% VaR (red = exception)", loc="left", color=INK)
    ax.set_ylabel("INR crore"); ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.tight_layout(); fig.savefig(C.CHARTS / "01_backtest.png"); plt.close(fig)

    # 2. P&L distribution with VaR / ES
    pnl = M.pnl_series(win, w)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pnl, bins=50, color="#9ca3af", edgecolor="white")
    hv, hes = var_tbl.loc["Historical Simulation", ["1d VaR 99%", "1d ES 97.5%"]]
    ax.axvline(-hv, color=BLUE, lw=2, label=f"VaR 99%: {hv:.2f} cr")
    ax.axvline(-hes, color=RED, lw=2, ls="--", label=f"ES 97.5%: {hes:.2f} cr")
    ax.set_title("Historical 1-day P&L distribution (last 250 days)", loc="left", color=INK)
    ax.set_xlabel("INR crore"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.CHARTS / "02_pnl_distribution.png"); plt.close(fig)

    # 3. Component VaR
    c = comp["Component VaR"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(c.index, c.values, color=[GREEN if v < 0 else BLUE for v in c.values])
    for i, v in enumerate(c.values):
        ax.text(v, i, f" {v:.2f}", va="center", ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    ax.set_title("Component VaR by currency (negative = diversifying)", loc="left", color=INK)
    ax.set_xlabel("INR crore"); ax.axvline(0, color=MUTED, lw=0.8)
    fig.tight_layout(); fig.savefig(C.CHARTS / "03_component_var.png"); plt.close(fig)

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index); ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.6 else INK)
    fig.colorbar(im, fraction=0.046); ax.set_title("Return correlation (INR crosses, 250d)", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(C.CHARTS / "04_correlation.png"); plt.close(fig)

    # 5. Stress results
    s = stress.sort_values("P&L (INR cr)")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(s["Scenario"], s["P&L (INR cr)"], color=[RED if v < 0 else GREEN for v in s["P&L (INR cr)"]])
    ax.axvline(-var_tbl.loc["Historical Simulation", "10d VaR 99%"], color=BLUE, ls="--", lw=1.4, label="10-day VaR 99%")
    ax.set_title("Stress test P&L on current book", loc="left", color=INK)
    ax.set_xlabel("INR crore"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(C.CHARTS / "05_stress_tests.png"); plt.close(fig)

    # 6. Rolling VaR through time
    roll = M.pnl_series(rets, w)
    hist_var = -roll.rolling(C.WINDOW).quantile(1 - C.VAR_CONF)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist_var.index, hist_var, color=BLUE, lw=1.3)
    ax.set_title("Rolling 250-day historical VaR 99% — how risk changed over time", loc="left", color=INK)
    ax.set_ylabel("INR crore")
    fig.tight_layout(); fig.savefig(C.CHARTS / "06_rolling_var.png"); plt.close(fig)


def format_workbook(path):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = load_workbook(path)
    hdr_fill = PatternFill("solid", fgColor="1F4E8C")
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF"); cell.fill = hdr_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.freeze_panes = "B2"
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                if isinstance(c.value, float):
                    hdr = str(ws.cell(1, c.column).value or "")
                    pct = hdr in {"1d VaR % of Gross", "% of Total", "Exception Rate"} or ws.title == "Correlation"
                    c.number_format = "0.00" if ws.title == "Correlation" else ("0.00%" if pct else "#,##0.000")
    wb.save(path)


if __name__ == "__main__":
    main()
