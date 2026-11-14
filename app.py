"""app.py - the Streamlit web app (user interface only).

Run it with:   streamlit run app.py
All calculations live in the `analysis` package, data downloads in `data`, and charts in `ui/charts.py`.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock analysis and portfolio builder", layout="wide")  # must be the first st call

import config  # noqa: E402
from analysis import pipeline as pl  # noqa: E402
from analysis import portfolio as pf  # noqa: E402
from analysis import sentiment as sm  # noqa: E402
from analysis import technicals as tech_mod  # noqa: E402
from analysis.ratios import ratio_tables  # noqa: E402
from analysis.risk_metrics import METRIC_LABELS, drawdown_series  # noqa: E402
from data.market_data import parse_tickers  # noqa: E402
from data.universes import UNIVERSE_NOTE, UNIVERSES  # noqa: E402
from ui import charts as ch  # noqa: E402
from ui import components as ui  # noqa: E402

config.setup_logging()
log = logging.getLogger("stock_app.app")
ui.inject_css()

# (key, label, min, max, step, is_percent, help) for the DCF assumption sliders
DCF_INPUTS = [
    ("growth1", "Revenue growth in year 1 (%)", -20.0, 50.0, 0.5, True, "Growth then fades towards the terminal rate over 5 years."),
    ("ebit_margin", "EBIT (operating) margin (%)", -30.0, 70.0, 0.5, True, "Operating profit as a share of sales."),
    ("tax_rate", "Tax rate (%)", 0.0, 45.0, 0.5, True, "Tax on operating profit."),
    ("capex_pct", "Capital spending, % of sales", 0.0, 40.0, 0.5, True, "Money spent on equipment and buildings."),
    ("da_pct", "Depreciation and amortisation, % of sales", 0.0, 25.0, 0.5, True, "Non-cash cost added back."),
    ("nwc_pct", "Working capital investment, % of extra sales", 0.0, 50.0, 1.0, True, "Cash tied up in stock and customer credit as sales grow."),
    ("terminal_growth", "Terminal growth (%)", 0.0, 6.0, 0.1, True, "Long-run growth after year 5. It is capped below the risk-free rate."),
    ("exit_multiple", "Exit EV/EBITDA multiple (x)", 3.0, 30.0, 0.5, False, "Default: the peer median."),
]
WACC_INPUTS = [
    ("rf", "Risk-free rate (%)", 0.0, 12.0, 0.1, True, "Return on very safe government bonds."),
    ("erp", "Equity risk premium (%)", 1.0, 10.0, 0.1, True, "Extra return investors demand for owning shares."),
    ("beta", "Beta", 0.2, 2.5, 0.05, False, "Sensitivity to the market. Default is bounded between 0.5 and 2.0."),
    ("cost_of_debt", "Cost of debt, before tax (%)", 0.0, 15.0, 0.1, True, "Interest expense divided by total debt (or risk-free rate + 2%)."),
    ("debt_weight", "Debt share of capital, D/(D+E) (%)", 0.0, 90.0, 1.0, True, "Uses today's market value of equity and book value of debt."),
]
REBALANCE_WORDS = {"M": "monthly", "Q": "quarterly", "Y": "annually"}
TABS = ["Overview", "Price & Risk", "News & Sentiment", "Ratios", "Valuation", "Recommendations",
        "Portfolio Optimiser", "Stock Screener", "Help & Glossary"]


# =============================================================================
# Small helpers
# =============================================================================
def num_config(df: pd.DataFrame, decimals: int = 2, skip: tuple = ()) -> dict:
    """Column formats so numbers stay numeric (sortable) but read cleanly."""
    return {c: st.column_config.NumberColumn(format=f"%.{decimals}f") for c in df.columns
            if c not in skip and pd.api.types.is_float_dtype(df[c])}


def pct_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Turn fractions into percentages and mark the column title with (%)."""
    out = df.copy()
    for c in cols:
        if c in out:
            out[c + " (%)"] = out[c] * 100
            out = out.drop(columns=[c])
    return out


def safe(label: str, fn, *args) -> None:
    """Run one tab; if it fails show a friendly message instead of a raw error page."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        log.exception("Tab '%s' failed", label)
        st.error(f"Sorry, the '{label}' tab ran into a problem and could not be drawn. The details were saved to "
                 f"stock_app.log. Try pressing 'Run analysis' again.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")


def add_ticker(symbols: list[str]) -> None:
    """Button callback: add tickers to the sidebar list (the user then presses Run analysis)."""
    current, _ = parse_tickers(st.session_state.get("tickers_input", ""))
    new = [s for s in dict.fromkeys(symbols) if s not in current]
    st.session_state["tickers_input"] = ", ".join(current + new)
    st.session_state["added_note"] = ", ".join(new) if new else "nothing new (already in your list)"


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> None:
    ss = st.session_state
    ss.setdefault("news_nonce", 0)
    ss.setdefault("run_id", 0)
    st.sidebar.title("Controls")
    with st.sidebar:
        with st.form("controls"):
            raw = st.text_area("Stock tickers", value=config.DEFAULT_TICKERS, key="tickers_input", height=90,
                               help="Separate with commas or spaces. US: AAPL, MSFT. London: VOD.L. Amsterdam: ASML.AS. "
                                    "Frankfurt: SAP.DE. Tokyo: 7203.T.")
            hist_label = st.select_slider("History", options=list(config.HISTORY_OPTIONS), value=config.DEFAULT_HISTORY)
            c1, c2 = st.columns(2)
            rf = c1.number_input("Risk-free rate (%)", 0.0, 20.0, config.DEFAULT_RF * 100, 0.1,
                                 help="Annual return of very safe bonds. Used for Sharpe, CAPM and the optimiser.")
            base_ccy = c2.selectbox("Base currency", config.BASE_CURRENCIES,
                                    help="Prices and portfolio maths are converted into this currency.")
            bench_label = st.selectbox("Benchmark", list(config.BENCHMARKS))
            custom_bench = st.text_input("Custom benchmark ticker", placeholder="only used if 'Custom ticker' is chosen")
            st.markdown("**Portfolio constraints**")
            c3, c4 = st.columns(2)
            min_w = c3.number_input("Min weight (%)", 0.0, 100.0, config.DEFAULT_MIN_WEIGHT * 100, 1.0)
            max_w = c4.number_input("Max weight (%)", 1.0, 100.0, config.DEFAULT_MAX_WEIGHT * 100, 1.0)
            long_only = st.toggle("Long only (no short selling)", value=True)
            objective = st.selectbox("Optimisation objective", config.OBJECTIVES)
            target_vol = st.number_input("Target risk (% volatility)", 1.0, 100.0, config.DEFAULT_TARGET_VOL * 100, 1.0,
                                         help="Only used for 'Maximum Return for a Target Risk'.")
            amount = st.number_input("Investment amount (base currency)", 0.0, 1e9, float(config.DEFAULT_INVESTMENT), 500.0)
            with st.expander("Advanced settings"):
                ret_method = st.radio("Expected returns", ["Historical mean", "CAPM-implied"], horizontal=True)
                cov_method = st.radio("Covariance matrix", ["Ledoit-Wolf shrinkage", "Sample"], horizontal=True)
                rebalance = st.selectbox("Backtest rebalancing", list(config.REBALANCE_OPTIONS), index=1)
                sent_model = st.radio("Sentiment model", ["VADER (fast)", "FinBERT (needs transformers + torch)"])
                erp = st.number_input("Equity risk premium (%)", 1.0, 12.0, config.DEFAULT_ERP * 100, 0.1)
                tg = st.number_input("Terminal growth (%)", 0.0, 6.0, config.DEFAULT_TERMINAL_GROWTH * 100, 0.1)
                st.markdown("**Recommendation pillar weights** (rescaled to 100%)")
                pw = {p: st.slider(p, 0, 100, int(round(config.DEFAULT_PILLAR_WEIGHTS[p] * 100)), key=f"pw_{p}")
                      for p in config.PILLARS}
                st.markdown("**Blended valuation weights** (rescaled to 100%)")
                vw = {k: st.slider(k, 0, 100, int(round(v * 100)), key=f"vw_{k}")
                      for k, v in config.DEFAULT_VALUATION_WEIGHTS.items()}
            submitted = ui.stretch_button(st.form_submit_button, "Run analysis", type="primary")
        st.divider()
        st.markdown("**Live news**")
        if ui.stretch_button(st.button, "Refresh live news", key="refresh_news"):
            ss["news_nonce"] += 1
        auto = st.selectbox("Auto-refresh", ["Off", "Every 5 minutes", "Every 15 minutes", "Every 30 minutes"])
        minutes = {"Every 5 minutes": 5, "Every 15 minutes": 15, "Every 30 minutes": 30}.get(auto)
        ss["news_minutes"] = minutes
        if minutes:
            try:
                from streamlit_autorefresh import st_autorefresh

                st_autorefresh(interval=minutes * 60 * 1000, key="news_autorefresh")
            except ImportError:
                st.caption("Install `streamlit-autorefresh` to enable auto-refresh.")
        found = [k for k, v in config.keys_available().items() if v]
        st.caption("Optional data keys found: " + (", ".join(found) if found else "none. The app works without them."))
        st.caption("Free data can be delayed by about 15 minutes.")

    if submitted:
        good, bad = parse_tickers(raw)
        if not good:
            st.sidebar.error("Enter at least one ticker, for example AAPL.")
            return
        symbol, bench_ccy = config.BENCHMARKS[bench_label]
        if bench_label == "Custom ticker":
            symbol, bench_ccy = (custom_bench.strip().upper() or "^GSPC"), None
        pw_total, vw_total = sum(pw.values()) or 1, sum(vw.values()) or 1
        ss["bad_format"] = bad
        ss["run_id"] += 1
        ss["committed"] = {
            "tickers": good, "years": config.HISTORY_OPTIONS[hist_label], "rf": rf / 100, "erp": erp / 100,
            "terminal_growth": tg / 100, "min_w": min_w / 100, "max_w": max_w / 100, "long_only": long_only,
            "objective": objective, "target_vol": target_vol / 100, "invest_amount": amount,
            "benchmark_symbol": symbol, "benchmark_ccy": bench_ccy, "benchmark_label": bench_label,
            "base_currency": base_ccy, "pillar_weights": {p: v / pw_total for p, v in pw.items()},
            "valuation_weights": {k: v / vw_total for k, v in vw.items()},
            "sentiment_model": "FinBERT" if sent_model.startswith("FinBERT") else "VADER",
            "returns_method": "capm" if ret_method.startswith("CAPM") else "historical",
            "cov_method": "ledoit_wolf" if cov_method.startswith("Ledoit") else "sample",
            "rebalance": config.REBALANCE_OPTIONS[rebalance], "run_id": ss["run_id"],
        }
        if bench_label == "Custom ticker":
            ss["committed"]["benchmark_label"] = f"Benchmark ({symbol})"
        st.cache_data.clear()  # a fresh run downloads fresh data


def current_settings() -> dict | None:
    """Committed settings plus the news-refresh key (changes on manual refresh or auto-refresh tick)."""
    ss = st.session_state
    base = ss.get("committed")
    if not base:
        return None
    minutes = ss.get("news_minutes")
    tick = int(time.time() // (minutes * 60)) % 1_000_000 if minutes else 0
    return {**base, "news_key": ss["news_nonce"] * 1_000_000 + tick}


def collect_overrides(run_id: int, tickers: list[str]) -> dict:
    """Read the per-stock valuation controls (DCF sliders, haircuts, peers...) back from session state."""
    ss, out = st.session_state, {}
    for t in tickers:
        reset, entry = ss.get(f"reset_{t}", 0), {}
        dcf = {}
        for spec in DCF_INPUTS + WACC_INPUTS:
            v = ss.get(f"dcf|{run_id}|{reset}|{t}|{spec[0]}")
            if v is not None:
                dcf[spec[0]] = v / 100 if spec[5] else v
        if dcf:
            entry["dcf"] = dcf
        hair = {c: ss.get(f"hc|{run_id}|{reset}|{t}|{c}") for c in config.DEFAULT_HAIRCUTS}
        hair = {c: v / 100 for c, v in hair.items() if v is not None}
        if hair:
            entry["haircuts"] = hair
        probs = {s: ss.get(f"pr|{run_id}|{reset}|{t}|{s}") for s in config.SCENARIOS}
        probs = {s: v / 100 for s, v in probs.items() if v is not None}
        if probs:
            entry["probs"] = probs
        peers = ss.get(f"peers|{run_id}|{t}")
        if peers and peers.strip():
            entry["peers"] = parse_tickers(peers)[0]
        if ss.get(f"force|{run_id}|{t}"):
            entry["force_dcf"] = True
        if entry:
            out[t] = entry
    return out


# =============================================================================
# Tab 1: Overview
# =============================================================================
def show_warnings(A: dict) -> None:
    m = A["market"]
    if m["invalid"]:
        st.warning("No price data was found for: **" + ", ".join(m["invalid"]) + "**. Check the spelling and add the "
                   "exchange suffix for non-US stocks (VOD.L, ASML.AS, SAP.DE). The other stocks were analysed.")
    bad = st.session_state.get("bad_format") or []
    if bad:
        st.warning("These entries do not look like tickers and were ignored: " + ", ".join(bad))
    for msg in m["warnings"] + A["prices"]["warnings"]:
        st.warning(msg)
    fallback = {t: s for t, s in m["price_src"].items() if s != "Yahoo Finance" and t in m["tickers"]}
    if fallback:
        st.info("Yahoo Finance had no data for " + ", ".join(fallback) + ", so a fallback source was used: "
                + "; ".join(f"{t} via {s}" for t, s in fallback.items()) + ". Fallback prices may not be dividend-adjusted.")


def tab_overview(A: dict | None) -> None:
    if A is None:
        st.subheader("Welcome")
        st.markdown("1. Type your tickers in the sidebar (the example list is already filled in).\n"
                    "2. Press **Run analysis**.\n3. Explore the tabs: prices and risk, news sentiment, ratios, "
                    "three valuations, recommendations, and the portfolio optimiser.\n\nNo API keys are needed. "
                    "The first run takes a minute because it downloads prices, statements and news.")
        st.info(ui.DISCLAIMER)
        return
    if not A["ok"]:
        st.error("No valid tickers were found, so there is nothing to analyse. Check the spelling and exchange suffix "
                 "(for example VOD.L for London or ASML.AS for Amsterdam) and run again.")
        show_warnings({"market": A["market"], "prices": {"warnings": []}})
        return
    m = A["market"]
    st.caption(f"Last updated {A['asof']:%d %b %Y %H:%M} UTC  ·  base currency {m['base_ccy']}  ·  benchmark "
               f"{m['bench_label']}  ·  prices adjusted for splits and dividends")
    show_warnings(A)
    st.subheader("Portfolio headline")
    pfo = A["portfolio"]
    if pfo["available"]:
        c = pfo["chosen"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected return", ui.fmt_pct(c["expected_return"]), help="Annual, from the selected return method.")
        k2.metric("Volatility", ui.fmt_pct(c["volatility"]), help="Annual standard deviation of returns.")
        k3.metric("Sharpe ratio", ui.fmt_num(c["sharpe"]))
        k4.metric("Objective", c["objective"])
        w = c["weights"].sort_values(ascending=False)
        st.markdown("**Largest weights:** " + ", ".join(f"{t} {v:.0%}" for t, v in w.head(5).items())
                    + ". See the Portfolio Optimiser tab for the full picture.")
    else:
        st.info(pfo["reason"])
    st.subheader("Your stocks")
    cols = st.columns(3)
    for i, t in enumerate(m["tickers"]):
        px = m["prices_major"][t].dropna()
        change = float(px.iloc[-1] / px.iloc[-2] - 1) if len(px) > 1 else float("nan")
        blend = A["fund"][t]["valuation"]["blend"]
        s = A["scores"][t]
        with cols[i % 3]:
            ui.stock_card(t, m["names"][t], float(px.iloc[-1]), m["ccy"][t], change, s["rating"], s["composite"],
                          blend["fair_value"], blend["upside"], A["sentiment"][t]["label"])
    st.caption(ui.DISCLAIMER)


# =============================================================================
# Tab 2: Price and risk
# =============================================================================
def tab_price_risk(A: dict | None) -> None:
    if not A or not A["ok"]:
        st.info("Run an analysis first.")
        return
    m, P = A["market"], A["prices"]
    names = m["names"]
    t = st.selectbox("Stock", m["tickers"], format_func=lambda x: f"{x} · {names[x]}", key="pr_stock")
    start = P["window_start"]
    frame = P["tech_frames"][t].loc[start:]
    vol = m["volume"][t].loc[start:] if t in m["volume"].columns else None
    crosses = tech_mod.cross_events(m["prices_major"][t].dropna())
    crosses = crosses[crosses.index >= start]
    ui.show_plot(ch.price_chart(t, frame, vol, m["ccy"][t], crosses), f"price_{t}")
    s = P["tech"][t]
    if s:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)", ui.fmt_num(s["rsi"], 1), s["rsi_state"], delta_color="off")
        c2.metric("MACD", "Bullish" if s["macd"] > s["macd_signal"] else "Bearish", help=s["macd_state"])
        c3.metric("Trend", "Up" if "uptrend" in s["trend"] else "Down" if "downtrend" in s["trend"] else "N/A", help=s["trend"])
        c4.metric("50/200-day cross", (s["last_cross"] or "None"), help=s["cross_flag"])
        st.caption(s["cross_flag"] + ". " + s["trend"] + ".")
    t1, t2, t3 = st.tabs(["RSI", "MACD", "Bollinger Bands"])
    with t1:
        ui.show_plot(ch.rsi_chart(t, frame), f"rsi_{t}")
    with t2:
        ui.show_plot(ch.macd_chart(t, frame), f"macd_{t}")
    with t3:
        ui.show_plot(ch.bollinger_chart(t, frame), f"bb_{t}")
    st.subheader("Risk and return metrics")
    st.caption(f"Calculated in {m['base_ccy']} over the common period, with a {A['settings']['rf']:.1%} risk-free rate. "
               "Hover a column title for sorting; click it to sort.")
    met = P["metrics"].copy()
    table = pd.DataFrame({(f"{lbl} (%)" if kind == "pct" else lbl): (met[k] * 100 if kind == "pct" else met[k])
                          for k, (lbl, kind) in METRIC_LABELS.items()})
    table.insert(0, "Ticker", table.index)
    ui.show_table(table.reset_index(drop=True), column_config=num_config(table))
    st.markdown("**How to read this:** Sharpe and Sortino above 1 are good; a beta above 1 means the stock amplifies "
                "market moves; VaR and CVaR are one-day losses on bad days (95% of days are better than VaR).")
    left, right = st.columns(2)
    with left:
        win = P["window_prices"][P["eligible"] + P["short"]]
        ui.show_plot(ch.drawdown_chart(drawdown_series(win.dropna(how="all"))), "drawdowns")
    with right:
        if len(P["corr"]):
            ui.show_plot(ch.correlation_heatmap(P["corr"]), "corr")
        else:
            st.info("The correlation matrix needs at least two stocks with a year or more of price history.")
    ui.show_plot(ch.cumulative_returns_chart(P["cum"], m["bench_label"]), "cumulative")


# =============================================================================
# Tab 3: News and sentiment
# =============================================================================
def tab_news(A: dict | None) -> None:
    if not A or not A["ok"]:
        st.info("Run an analysis first.")
        return
    m, S = A["market"], A["sentiment"]
    stamp = max(s["fetched_at"] for s in S.values())
    st.caption(f"News last downloaded {stamp:%d %b %Y %H:%M:%S} UTC. Press 'Refresh live news' in the sidebar for the "
               f"latest, or turn on auto-refresh. Model used: {next(iter(S.values()))['model']}.")
    st.info("Social sentiment is noisy and easy to manipulate. Treat it as context, never as the only input to a decision.")
    with st.expander("How the sentiment score is weighted"):
        st.markdown(sm.weighting_explanation())
    summary = pd.DataFrame([{"Ticker": t, "Sentiment": S[t]["label"], "Score": S[t]["overall"], "News items": S[t]["n_news"],
                             "Social posts": S[t]["n_social"],
                             "Mentions spike": "Yes" if S[t]["spike"].get("flag") else "No"} for t in m["tickers"]])
    ui.show_table(summary, column_config=num_config(summary))
    t = st.selectbox("Stock", m["tickers"], format_func=lambda x: f"{x} · {m['names'][x]}", key="news_stock")
    r = S[t]
    for note in r["notices"]:
        st.caption(note)
    if r["n"] == 0:
        st.warning("No news or posts could be downloaded for this stock right now. Try the refresh button later.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall sentiment", r["label"], f"{r['overall']:+.2f}", delta_color="off")
    c2.metric("News score", ui.fmt_num(r["news_score"]), help="Weighted average of news headlines.")
    c3.metric("Social score", ui.fmt_num(r["social_score"]), help="Weighted average of social posts (blank if none).")
    c4.metric("Items analysed", f"{r['n']}", f"{r['n_news']} news, {r['n_social']} social", delta_color="off")
    if r["spike"].get("flag"):
        st.warning("Unusual activity: " + r["spike"]["note"])
    else:
        st.caption("Mention volume: " + r["spike"]["note"])
    st.markdown("**Key risk and event headlines**")
    ui.event_items(r["events"])
    left, right = st.columns(2)
    with left:
        if len(r["timeline"]):
            ui.show_plot(ch.sentiment_timeline_chart(r["timeline"], t), f"sent_time_{t}")
    with right:
        if len(r["counts_by_type"]):
            ui.show_plot(ch.sentiment_counts_chart(r["counts_by_type"], t), f"sent_count_{t}")
    pos, neg = st.columns(2)
    with pos:
        st.markdown("**Top 5 most positive**")
        ui.news_items(r["top_pos"], 5, show_summary=False)
    with neg:
        st.markdown("**Top 5 most negative**")
        ui.news_items(r["top_neg"], 5, show_summary=False)
    items = r["items"]
    with st.expander(f"Latest news ({r['n_news']})", expanded=True):
        ui.news_items(items[items["type"] == "news"], 20)
    with st.expander(f"Latest social posts ({r['n_social']})"):
        ui.news_items(items[items["type"] == "social"], 30, show_summary=False)


# =============================================================================
# Tab 4: Ratios
# =============================================================================
def tab_ratios(A: dict | None) -> None:
    if not A or not A["ok"]:
        st.info("Run an analysis first.")
        return
    m = A["market"]
    ratios = {t: A["fund"][t]["ratios"] for t in m["tickers"]}
    mode = st.radio("Colour the table by", ["General benchmarks", "Median of the other stocks I entered"], horizontal=True)
    if mode.startswith("Median") and len(ratios) < 2:
        st.info("Enter at least two stocks to compare with each other. Showing general benchmarks instead.")
    display, colours, raw = ratio_tables(ratios, "peers" if mode.startswith("Median") and len(ratios) > 1 else "benchmark")
    st.caption("Green = favourable, amber = middling, red = unfavourable. Ratios using trailing twelve months (TTM) of "
               "results and the latest balance sheet. N/A means the data is not available (common for banks).")
    ui.show_table(ui.styled_ratio_table(display, colours), height=min(40 + 35 * len(display), 1250))
    c1, c2 = st.columns(2)
    csv = raw.to_csv(index=False).encode("utf-8")
    c1.download_button("Download ratios as CSV", csv, "ratios.csv", "text/csv", key="dl_ratios_csv")
    xlsx = pl.workbook_bytes({"Ratios": raw})
    c2.download_button("Download ratios as Excel", xlsx, "ratios.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_ratios_xlsx")
    t = st.selectbox("Explore one stock in detail (hover the (?) icons for explanations)", m["tickers"], key="ratio_stock")
    group_tabs = st.tabs(["Valuation", "Profitability", "Liquidity", "Leverage", "Efficiency", "Growth", "Dividend"])
    for tab, group in zip(group_tabs, ["Valuation", "Profitability", "Liquidity", "Leverage", "Efficiency", "Growth", "Dividend"]):
        with tab:
            ui.ratio_grid(ratios[t], group)
    if not A["fund"][t]["base"]["has_statements"]:
        st.warning(f"No financial statements were found for {t}, so most ratios are N/A. Yahoo Finance may not cover this stock.")


# =============================================================================
# Tab 5: Valuation
# =============================================================================
def dcf_controls(t: str, run_id: int, v: dict, forced: bool) -> None:
    """Sliders that let you override every DCF and WACC assumption for one stock."""
    reset = st.session_state.get(f"reset_{t}", 0)
    defaults = v["defaults"]
    with st.expander("Adjust the DCF and WACC assumptions", expanded=False):
        st.caption("Defaults come from the company's history and analyst estimates. Move a slider to override it; "
                   "all valuations, scores and ratings update.")
        cols = st.columns(2)
        for group, container in ((DCF_INPUTS, cols[0]), (WACC_INPUTS, cols[1])):
            with container:
                st.markdown("**Cash flow forecast**" if group is DCF_INPUTS else "**Discount rate (WACC) inputs**")
                for key, label, lo, hi, step, pct, help_ in group:
                    default = defaults.get(key, float("nan"))
                    if not np.isfinite(default):
                        default = (lo + hi) / 2 / (100 if pct else 1)
                    value = float(np.clip(default * 100 if pct else default, lo, hi))
                    st.slider(label, lo, hi, value, step, help=help_, key=f"dcf|{run_id}|{reset}|{t}|{key}")
        if st.button("Reset this stock's assumptions", key=f"reset_btn_{t}"):
            st.session_state[f"reset_{t}"] = reset + 1
            st.rerun()


def tab_valuation(A: dict | None) -> None:
    if not A or not A["ok"]:
        st.info("Run an analysis first.")
        return
    m, run_id = A["market"], A["settings"]["run_id"]
    summary = pl.valuation_summary(A)
    st.subheader("Valuation summary")
    w = A["settings"]["valuation_weights"]
    st.caption("Blended fair value = " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()) + " (change the weights in the "
               "sidebar; a method with no result is dropped and the weights are rescaled). Upside = fair value / price - 1. "
               "Values are per share in each stock's own currency.")
    show = pct_columns(summary, ["Upside / downside", "Margin of safety"])
    ui.show_table(show, column_config=num_config(show))
    t = st.selectbox("Stock to examine", m["tickers"], format_func=lambda x: f"{x} · {m['names'][x]}", key="val_stock")
    v, ccy = A["fund"][t]["valuation"], m["ccy"][t]
    base, price = A["fund"][t]["base"], v["price"]
    if not base["has_statements"]:
        st.warning("No financial statements were found for this stock, so the valuation relies on very little data.")
    comp, dcf, asset, ddm = v["comparative"], v["dcf"], v["asset"], v["ddm"]
    ranges = [{"label": f"{r['Method']} (peer multiples)", "low": r["Low"], "mid": r["Implied value"], "high": r["High"]}
              for _, r in comp["table"].iterrows()] if comp["available"] else []
    if dcf["applicable"]:
        sc = dcf["scenarios"]["table"]["Value per share"].dropna()
        ranges.append({"label": "DCF (bear to bull)", "low": float(sc.min()), "mid": dcf["value"]["value_per_share"],
                       "high": float(sc.max())})
    elif ddm["applicable"]:
        ranges.append({"label": "Dividend discount model", "low": ddm["value"], "mid": ddm["value"], "high": ddm["value"]})
    if asset["available"]:
        ranges.append({"label": "Asset-based (liquidation to book)", "low": asset["liquidation_ps"],
                       "mid": asset["adjusted_nav_ps"], "high": max(asset["book_value_ps"], asset["adjusted_nav_ps"])})
    fair = v["blend"]["fair_value"]
    if np.isfinite(fair):
        ranges.append({"label": "Blended fair value", "low": fair, "mid": fair, "high": fair})
    ranges = [r for r in ranges if all(np.isfinite(r[k]) for k in ("low", "mid", "high"))]
    if ranges:
        ui.show_plot(ch.football_field(ranges, price, ccy, f"{t}: valuation football field"), f"football_{t}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"{price:,.2f} {ccy}")
        c2.metric("Blended fair value", f"{fair:,.2f} {ccy}" if np.isfinite(fair) else "N/A")
        c3.metric("Upside / downside", ui.fmt_pct(v["blend"]["upside"], 0, signed=True))
    else:
        st.warning("None of the three methods could produce a value for this stock.")
    sub = st.tabs(["Comparative", "DCF", "Asset-based", "How it blends"])
    with sub[0]:
        comp_tab(t, A, v, run_id)
    with sub[1]:
        dcf_tab(t, A, v, run_id, ccy, price)
    with sub[2]:
        asset_tab(t, v, run_id, ccy, price)
    with sub[3]:
        used = v["blend"]["weights_used"]
        rows = [{"Method": k, "Value per share": val, "Weight used": used.get(k, 0.0) * 100}
                for k, val in v["blend"]["values"].items()]
        tbl = pd.DataFrame(rows)
        ui.show_table(tbl.rename(columns={"Weight used": "Weight used (%)"}), column_config=num_config(tbl.rename(columns={"Weight used": "Weight used (%)"})))
        st.markdown("The DCF value is the **probability-weighted** value across bear, base and bull scenarios (or the "
                    "dividend discount model for banks and similar companies). Treat every valuation as a range: small "
                    "changes in growth or the discount rate move it a lot.")


def comp_tab(t: str, A: dict, v: dict, run_id: int) -> None:
    comp = v["comparative"]
    st.text_input("Peer tickers (optional, comma separated, up to 8)", key=f"peers|{run_id}|{t}",
                  placeholder="leave blank to use peers from the same Yahoo Finance industry",
                  help="Type your own peers, for example: MSFT, GOOGL, META. Press Enter to apply.")
    peers = A["fund"][t]["peers"]
    st.caption("Peers used: " + (", ".join(peers) if peers else "none found"))
    if not comp["available"]:
        st.warning(comp["message"])
        return
    stats = comp["stats"].reset_index().rename(columns={"index": "Multiple"})
    ui.show_table(stats, column_config=num_config(stats))
    tbl = comp["table"]
    ui.show_table(tbl, column_config=num_config(tbl))
    st.markdown(f"**Average implied value: {comp['mid']:,.2f}** (range {comp['low']:,.2f} to {comp['high']:,.2f}, using the "
                "25th to 75th percentile of peer multiples). Peers' multiples come from Yahoo Finance and can be noisy.")
    peer_df = A["fund"][t]["peer_table"]
    if len(peer_df):
        with st.expander("Peer data"):
            ui.show_table(peer_df, column_config=num_config(peer_df))


def dcf_tab(t: str, A: dict, v: dict, run_id: int, ccy: str, price: float) -> None:
    dcf, ddm = v["dcf"], v["ddm"]
    for w in dcf["warnings"]:
        st.warning(w)
    if dcf.get("blocked"):
        st.checkbox("Run the DCF anyway", key=f"force|{run_id}|{t}")
        if ddm["applicable"]:
            st.success(f"Dividend discount model value: {ddm['value']:,.2f} {ccy} (dividend {ddm['dividend_per_share']:.2f}, "
                       f"cost of equity {ddm['cost_of_equity']:.1%}, long-run growth {ddm['terminal_growth']:.1%}).")
        else:
            st.info(ddm["reason"] + " Use the asset-based and comparative methods, or a residual income model.")
        return
    if not dcf["wacc"]:
        return
    dcf_controls(t, run_id, v, False)
    val, wacc = dcf["value"], dcf["wacc"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gordon growth value", ui.fmt_num(val["value_gordon"]), help="Terminal value from steady growth forever.")
    c2.metric("Exit multiple value", ui.fmt_num(val["value_exit"]), help="Terminal value from selling at a multiple of EBITDA.")
    c3.metric("DCF value (average)", ui.fmt_num(val["value_per_share"]))
    mos = (val["value_per_share"] - price) / val["value_per_share"] if val["value_per_share"] > 0 else float("nan")
    c4.metric("Margin of safety", ui.fmt_pct(mos, 0, signed=True), help="(Intrinsic value - price) / intrinsic value.")
    st.markdown("**WACC build-up**")
    w = pd.DataFrame([{"Cost of equity (CAPM)": wacc["cost_of_equity"] * 100, "Beta": wacc["beta"], "Risk-free (%)": wacc["rf"] * 100,
                       "Equity risk premium (%)": wacc["erp"] * 100, "Cost of debt after tax (%)": wacc["after_tax_cost_of_debt"] * 100,
                       "Equity weight (%)": wacc["equity_weight"] * 100, "Debt weight (%)": wacc["debt_weight"] * 100,
                       "WACC (%)": wacc["wacc"] * 100, "Terminal growth (%)": val["terminal_growth"] * 100}])
    ui.show_table(w, column_config=num_config(w))
    proj = val["projection"].copy()
    ui.show_plot(ch.dcf_projection_chart(proj, ccy), f"proj_{t}")
    show = proj[["growth", "revenue", "ebit", "nopat", "da", "capex", "change_nwc", "fcff", "pv_fcff"]].copy()
    show["growth"] *= 100
    show = show.rename(columns={"growth": "Growth (%)", "revenue": "Revenue", "ebit": "EBIT", "nopat": "NOPAT", "da": "D&A",
                                "capex": "Capex", "change_nwc": "Change in NWC", "fcff": "FCFF", "pv_fcff": "PV of FCFF"})
    show.insert(0, "Year", show.index)
    ui.show_table(show.reset_index(drop=True), column_config=num_config(show))
    tv = pd.DataFrame([{"Method": "Gordon growth", "Terminal value": val["tv_gordon"], "Enterprise value": val["ev_gordon"],
                        "Value per share": val["value_gordon"]},
                       {"Method": "Exit multiple", "Terminal value": val["tv_exit"], "Enterprise value": val["ev_exit"],
                        "Value per share": val["value_exit"]},
                       {"Method": "Average", "Terminal value": np.nan, "Enterprise value": val["ev_avg"],
                        "Value per share": val["value_per_share"]}])
    ui.show_table(tv, column_config=num_config(tv))
    st.caption(f"Equity value = enterprise value - net debt ({dcf['assumptions'].net_debt:,.0f}); divided by "
               f"{dcf['assumptions'].shares:,.0f} diluted shares.")
    left, right = st.columns(2)
    with left:
        ui.show_plot(ch.sensitivity_heatmap(dcf["sensitivity"], price, ccy), f"sens_{t}")
        st.caption("Sensitivity uses the Gordon growth terminal value so both inputs show their effect.")
    with right:
        sc = dcf["scenarios"]
        ui.show_plot(ch.scenario_chart(sc["table"], price, ccy), f"scen_{t}")
        st.markdown(f"**Probability-weighted DCF value: {sc['weighted']:,.2f} {ccy}**")
    reset = st.session_state.get(f"reset_{t}", 0)
    with st.expander("Scenario probabilities"):
        cols = st.columns(3)
        for col, (name, p) in zip(cols, config.DEFAULT_SCENARIO_PROBS.items()):
            col.slider(f"{name} (%)", 0, 100, int(p * 100), key=f"pr|{run_id}|{reset}|{t}|{name}")
        st.caption("Bear: slower growth, lower margin, higher discount rate. Bull: the opposite. Weights are rescaled to 100%.")


def asset_tab(t: str, v: dict, run_id: int, ccy: str, price: float) -> None:
    asset = v["asset"]
    if not asset["available"]:
        st.warning(asset.get("message", "Asset-based valuation is unavailable."))
        return
    (st.success if asset["suitable"] else st.info)(asset["note"])
    reset = st.session_state.get(f"reset_{t}", 0)
    defaults = {r["Asset"]: r["Recoverable %"] for _, r in asset["table"].iterrows()}
    with st.expander("Adjust the asset haircuts (share of book value you think is recoverable)"):
        cols = st.columns(2)
        for i, (cat, default) in enumerate(defaults.items()):
            cols[i % 2].slider(f"{cat} (%)", 0, 100, int(round(default * 100)), key=f"hc|{run_id}|{reset}|{t}|{cat}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Book value per share", ui.fmt_num(asset["book_value_ps"]), help="Total assets minus total liabilities, per share.")
    c2.metric("Tangible book value", ui.fmt_num(asset["tangible_bv_ps"]), help="Book value minus goodwill and intangibles.")
    c3.metric("Adjusted NAV", ui.fmt_num(asset["adjusted_nav_ps"]), help="Assets after haircuts, minus liabilities.")
    c4.metric("Liquidation value", ui.fmt_num(asset["liquidation_ps"]), help="Adjusted assets less 5% selling costs, minus liabilities.")
    st.caption(f"Price {price:,.2f} {ccy} is {price / asset['book_value_ps']:.1f}x book value per share." if asset["book_value_ps"] > 0 else "")
    tbl = asset["table"].copy()
    tbl["Recoverable %"] *= 100
    ui.show_table(tbl, column_config=num_config(tbl))
    st.caption(asset["replacement_note"])


# =============================================================================
# Tab 6: Recommendations
# =============================================================================
def tab_recommendations(A: dict | None) -> None:
    if not A or not A["ok"]:
        st.info("Run an analysis first.")
        return
    m = A["market"]
    rank = A["ranking"].copy()
    st.subheader("Ranking")
    w = A["settings"]["pillar_weights"]
    st.caption("Composite score (0 to 100) = " + ", ".join(f"{p} {x:.0%}" for p, x in w.items()) + ". Rating: Strong Buy 80+, "
               "Buy 65-79, Hold 45-64, Reduce 30-44, Sell below 30 (at least three pillars must have data). Change the "
               "weights in the sidebar under Advanced settings.")
    rank["Upside"] = rank["Upside"] * 100
    rank = rank.rename(columns={"Upside": "Upside (%)"})
    cfg = {p: st.column_config.ProgressColumn(p, min_value=0, max_value=100, format="%.0f") for p in config.PILLARS}
    cfg["Score"] = st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")
    cfg.update({c: st.column_config.NumberColumn(format="%.2f") for c in ("Fair value", "Price", "Upside (%)")})
    ui.show_table(rank, column_config=cfg)
    pillars = pd.DataFrame({t: A["scores"][t]["pillars"] for t in m["tickers"]}).T
    left, right = st.columns([1, 1])
    with left:
        ui.show_plot(ch.radar_chart(pillars), "radar")
    with right:
        st.markdown("**What each pillar measures**")
        st.markdown("- **Valuation:** upside to the blended fair value.\n- **Quality:** profitability and balance sheet strength.\n"
                    "- **Growth:** sales, profit and cash flow growth.\n- **Momentum:** trend, 12-month return, RSI, MACD.\n"
                    "- **Risk:** volatility, drawdown, beta (higher score = lower risk).\n- **Sentiment:** news and social tone.")
    st.subheader("Why each rating")
    for t in rank["Ticker"]:
        s = A["scores"][t]
        with st.expander(f"{t} · {m['names'][t]}: {s['rating']} ({s['composite']:.0f}/100)" if np.isfinite(s["composite"]) else f"{t}: not enough data"):
            st.markdown(ui.rating_badge(s["rating"]), unsafe_allow_html=True)
            for bullet in s["explanation"]:
                st.markdown("- " + bullet)
    st.warning(ui.DISCLAIMER)


# =============================================================================
# Tab 7: Portfolio optimiser
# =============================================================================
def tab_optimiser(A: dict | None) -> None:
    if not A or not A["ok"]:
        st.info("Run an analysis first.")
        return
    pfo, S = A["portfolio"], A["settings"]
    if not pfo["available"]:
        st.info(pfo["reason"] + " Add more stocks in the sidebar, or use the Stock Screener tab to find candidates.")
        return
    for msg in pfo["messages"]:
        st.warning(msg)
    chosen, tickers = pfo["chosen"], pfo["tickers"]
    st.caption(f"Objective: **{chosen['objective']}** · expected returns: {'CAPM-implied' if S['returns_method'] == 'capm' else 'historical mean'} · "
               f"covariance: {'Ledoit-Wolf shrinkage' if S['cov_method'] == 'ledoit_wolf' else 'sample'} · solver: {chosen['engine']} · "
               f"weights {pfo['bounds'][0]:.0%} to {pfo['bounds'][1]:.0%} per stock · {len(pfo['returns'])} days of common history")
    names = {"Maximum Sharpe": "Maximum Sharpe", "Minimum volatility": "Minimum volatility", "Equal weight": "Equal weight",
             "Risk parity": "Risk parity"}
    markers = {n: (pfo["results"][n]["volatility"], pfo["results"][n]["expected_return"]) for n in names}
    markers["Selected portfolio"] = (chosen["volatility"], chosen["expected_return"])
    ui.show_plot(ch.frontier_chart(pfo["random"], pfo["frontier"], markers, pfo["stocks"]), "frontier")
    st.caption(f"{len(pfo['random']):,} random portfolios (coloured by Sharpe ratio) sit behind the efficient frontier. "
               "The frontier is the best possible return for each level of risk.")
    st.subheader("Optimal weights")
    w = chosen["weights"]
    stats = pfo["compare"].iloc[0]
    k = st.columns(5)
    k[0].metric("Expected return", ui.fmt_pct(stats["Expected return"]))
    k[1].metric("Volatility", ui.fmt_pct(stats["Volatility"]))
    k[2].metric("Sharpe ratio", ui.fmt_num(stats["Sharpe"]))
    k[3].metric("Max drawdown (history)", ui.fmt_pct(stats["Max drawdown (history)"]))
    k[4].metric("Diversification ratio", ui.fmt_num(stats["Diversification ratio"]), help="Above 1 means diversification is lowering risk.")
    wt = pf.weights_table(w, pfo["last_prices"], S["invest_amount"]).rename_axis("Ticker").reset_index()
    wt["Weight"] *= 100
    wt = wt.rename(columns={"Weight": "Weight (%)", "Price": f"Price ({A['market']['base_ccy']})", "Amount": f"Amount ({A['market']['base_ccy']})"})
    left, mid, right = st.columns([1.1, 1, 1])
    with left:
        ui.show_table(wt, column_config=num_config(wt))
    with mid:
        ui.show_plot(ch.weights_donut(w), "donut")
    with right:
        ui.show_plot(ch.weights_bar(w), "wbar")
    st.subheader("Compared with other weightings")
    cmp_df = pfo["compare"].copy()
    cmp_df = pct_columns(cmp_df, ["Expected return", "Volatility", "Max drawdown (history)"]).reset_index().rename(columns={"index": "Portfolio"})
    ui.show_table(cmp_df, column_config=num_config(cmp_df))
    st.subheader("Risk contributions")
    l2, r2 = st.columns(2)
    with l2:
        ui.show_plot(ch.risk_contribution_chart(pfo["contrib"], "Selected portfolio: weight vs share of risk"), "rc_sel")
    with r2:
        ui.show_plot(ch.risk_contribution_chart(pfo["contrib_parity"], "Risk parity: every stock carries equal risk"), "rc_rp")
    st.subheader("Backtest")
    bt = pfo["backtest"]
    if bt is None:
        st.info("A backtest needs a benchmark and at least 200 days of common history.")
    else:
        st.info("**Look-ahead bias warning.** Testing weights on the same data used to find them flatters the result. "
                f"Here the weights were fitted using only the first {bt['train_days']} days ({config.TRAIN_FRACTION:.0%}) and "
                f"tested out-of-sample on the remaining {bt['test_days']} days, rebalancing {REBALANCE_WORDS[S['rebalance']]}. "
                "Past results do not guarantee future results, and an optimiser does not always beat a simple equal-weight mix.")
        ui.show_plot(ch.backtest_chart(bt["curves"], bt["in_sample"], bt["split_date"]), "backtest")
        met = bt["metrics"].copy()
        met = pct_columns(met, ["Total return", "CAGR", "Volatility", "Max drawdown"]).reset_index().rename(columns={"index": "Out-of-sample"})
        ui.show_table(met, column_config=num_config(met))
    st.subheader("Monte Carlo simulation")
    horizon = st.radio("Horizon", [1, 3, 5], index=2, horizontal=True, format_func=lambda y: f"{y} year{'s' if y > 1 else ''}", key="mc_h")
    mc = pf.monte_carlo(chosen["expected_return"], chosen["volatility"], years=5)
    ui.show_plot(ch.monte_carlo_chart(mc["bands"], horizon), f"mc_{horizon}")
    h = mc["horizons"][horizon]
    c = st.columns(4)
    c[0].metric("Median outcome", f"{h['median']:.2f}x", help="Value of 1 invested, after the chosen horizon.")
    c[1].metric("Bad case (5th percentile)", f"{h['p5']:.2f}x")
    c[2].metric("Good case (95th percentile)", f"{h['p95']:.2f}x")
    c[3].metric("Chance of losing money", ui.fmt_pct(h["prob_loss"], 0))
    st.caption(f"{config.MONTE_CARLO_PATHS:,} simulated paths using the portfolio's expected return and volatility. "
               "Real markets have fatter tails than the simulation assumes.")
    st.subheader("Try your own weights")
    st.caption("Move the sliders (they are rescaled to add up to 100%) to see the effect on risk and return.")
    cols = st.columns(min(len(tickers), 5))
    manual = {}
    low = 0.0 if S["long_only"] else -100.0     # short positions are negative weights
    for i, t in enumerate(tickers):
        start_value = float(np.clip(round(float(w[t]) * 100, 1), low, 100.0))
        manual[t] = cols[i % len(cols)].slider(t, low, 100.0, start_value, 0.5, key=f"mw|{S['run_id']}|{t}")
    total = sum(manual.values())
    if total <= 0:
        st.warning("Give at least one stock a weight above zero.")
    else:
        mw = pd.Series({t: v / total for t, v in manual.items()})
        comp = pf.compare_weights({"Optimal": w, "Your weights": mw}, pfo["mu"], pfo["cov"], S["rf"], pfo["returns"])
        comp = pct_columns(comp, ["Expected return", "Volatility", "Max drawdown (history)"]).reset_index().rename(columns={"index": "Portfolio"})
        st.caption(f"Slider total: {total:.1f}% (rescaled to 100%).")
        ui.show_table(comp, column_config=num_config(comp))
    st.subheader("Export")
    tables = pl.export_tables(A)
    e1, e2 = st.columns(2)
    e1.download_button("Download Excel workbook (one sheet per table)", pl.workbook_bytes(tables), "portfolio_analysis.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
    e2.download_button("Download weights as CSV", tables["Portfolio weights"].to_csv(index=False).encode("utf-8"),
                       "portfolio_weights.csv", "text/csv", key="dl_weights_csv")


# =============================================================================
# Tab 8: Stock screener
# =============================================================================
def tab_screener(settings: dict | None) -> None:
    st.markdown("Screen a built-in universe and shortlist the best-scoring stocks. **Stage 1** ranks every stock using prices "
                "only (momentum and risk). **Stage 2** downloads full financial statements for the best 20 and ranks them on "
                "the composite score. This screen has no news sentiment and uses DCF and asset-based valuation only, so run the "
                "full analysis on any candidate you like.")
    st.caption(UNIVERSE_NOTE)
    s = settings or {}
    universe = st.selectbox("Universe", list(UNIVERSES), key="universe")
    if st.button("Suggest candidates", type="primary", key="suggest"):
        bar = st.progress(0.0, text="Starting...")
        try:
            with st.spinner("Screening the universe (about a minute the first time)..."):
                result = pl.run_screener(universe, s.get("rf", config.DEFAULT_RF), s.get("erp", config.DEFAULT_ERP),
                                         s.get("terminal_growth", config.DEFAULT_TERMINAL_GROWTH),
                                         s.get("valuation_weights", config.DEFAULT_VALUATION_WEIGHTS),
                                         s.get("pillar_weights", config.DEFAULT_PILLAR_WEIGHTS), 10,
                                         _progress=lambda f, t: bar.progress(f, text=t))
            st.session_state["screen_result"] = result
        except Exception as exc:  # noqa: BLE001
            log.exception("Screener failed")
            st.error("The screener could not finish (usually a data or network problem). Please try again in a minute.")
            st.caption(f"{type(exc).__name__}: {exc}")
        bar.empty()
    result = st.session_state.get("screen_result")
    if st.session_state.get("added_note"):
        st.success(f"Added to your list: {st.session_state['added_note']}. Press **Run analysis** in the sidebar to analyse them.")
    if not result:
        return
    if result["message"] or result["top"].empty:
        st.warning(result["message"] or "No candidates could be scored.")
        return
    top = result["top"].copy()
    top["Upside"] = top["Upside"] * 100
    top = top.rename(columns={"Upside": "Upside (%)"})
    cfg = {p: st.column_config.ProgressColumn(p, min_value=0, max_value=100, format="%.0f")
           for p in ("Score", "Valuation", "Quality", "Growth", "Momentum", "Risk")}
    cfg["Upside (%)"] = st.column_config.NumberColumn(format="%.0f")
    st.subheader("Top candidates")
    ui.show_table(top, column_config=cfg)
    st.markdown("**Add to your list** (then press Run analysis in the sidebar):")
    cols = st.columns(5)
    for i, t in enumerate(top["Ticker"]):
        cols[i % 5].button(f"Add {t}", key=f"add_{t}", on_click=add_ticker, args=([t],))
    st.button("Add all of them", key="add_all", on_click=add_ticker, args=(top["Ticker"].tolist(),))
    with st.expander("Stage 1: price-only ranking of the whole universe"):
        st1 = result["stage1"].copy()
        for c in ("12M return", "Volatility"):
            st1[c + " (%)"] = st1[c] * 100
        st1 = st1.drop(columns=["12M return", "Volatility"])
        ui.show_table(st1, column_config=num_config(st1, 1))


# =============================================================================
# Tab 9: Help and glossary
# =============================================================================
def tab_help() -> None:
    st.subheader("How to use this app")
    st.markdown(
        "1. Enter tickers in the sidebar and press **Run analysis** (settings take effect when you press it).\n"
        "2. Read the tabs from left to right: price and risk, news sentiment, ratios, valuation, recommendation, portfolio.\n"
        "3. Change assumptions in the sidebar (Advanced settings) or in the Valuation tab, and everything updates.\n"
        "4. Use the **Stock Screener** to find new candidates, add them to your list, and optimise again.\n\n"
        "**Dark or light mode:** use the menu at the top right, then Settings, then Theme. Charts and tables follow it.\n\n"
        "**Ticker format:** US stocks use plain symbols (AAPL). Other markets need a suffix: .L London, .AS Amsterdam, "
        ".DE Germany, .PA Paris, .SW Switzerland, .T Tokyo, .TO Toronto, .AX Australia.")
    st.subheader("Where the data comes from")
    st.markdown("- **Prices, statements, FX, news:** Yahoo Finance via yfinance (free, unofficial), with Stooq and Alpha Vantage "
                "as price fallbacks.\n- **News:** yfinance news, Google News and Yahoo Finance RSS (plus Finnhub and NewsAPI if you add keys).\n"
                "- **Social:** StockTwits public stream, Reddit (key needed), X (paid key needed).\n- **Sentiment:** VADER, or FinBERT if installed.")
    st.subheader("Limitations")
    st.markdown("- Free data can be delayed, incomplete or wrong, and Yahoo can change without notice.\n"
                "- Social sentiment is noisy. Valuation outputs depend heavily on assumptions.\n"
                "- Optimisation uses past data; the future will differ. Optimal weights can be unstable.\n"
                "- Ratings are rule-based summaries, not predictions.")
    st.warning(ui.DISCLAIMER)
    st.subheader("Glossary")
    ui.render_glossary()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    render_sidebar()
    st.title(config.APP_TITLE)
    settings = current_settings()
    A = None
    if settings:
        bar = st.progress(0.0, text="Starting...")
        try:
            overrides = collect_overrides(settings["run_id"], settings["tickers"])
            with st.spinner("Downloading data and running the analysis..."):
                A = pl.run_analysis(settings, overrides, _progress=lambda f, text: bar.progress(f, text=text))
        except Exception as exc:  # noqa: BLE001 - never show a raw traceback
            log.exception("Analysis failed")
            st.error("The analysis could not be completed. This is usually a temporary data or network problem "
                     "(for example Yahoo Finance limiting requests). Wait a minute and press Run analysis again.")
            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")
        bar.empty()
    tabs = st.tabs(TABS)
    with tabs[0]:
        safe("Overview", tab_overview, A)
    with tabs[1]:
        safe("Price & Risk", tab_price_risk, A)
    with tabs[2]:
        safe("News & Sentiment", tab_news, A)
    with tabs[3]:
        safe("Ratios", tab_ratios, A)
    with tabs[4]:
        safe("Valuation", tab_valuation, A)
    with tabs[5]:
        safe("Recommendations", tab_recommendations, A)
    with tabs[6]:
        safe("Portfolio Optimiser", tab_optimiser, A)
    with tabs[7]:
        safe("Stock Screener", tab_screener, settings)
    with tabs[8]:
        safe("Help & Glossary", tab_help)


main()
