import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Streamlined Unicorn Edge — Reversal & Regime Swing Scanner",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦄 Streamlined Unicorn Edge: Reversal & Regime Swing System")
st.markdown(
    """
    *Stripped of the flawed 1/Price factor. Powered purely by the **63-day Drift Regime Filter**, **Short-Term Reversal**, and **Custom Exit Rules**.*
    """
)

# ==============================================================================
# DATA FETCHING & CACHING UTILITIES
# ==============================================================================
DEFAULT_TOP_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "XOM", "V", "PG", "MA", "COST", "JNJ", "HD", "MRK",
    "ABBV", "CVX", "CRM", "BAC", "NFLX", "WMT", "AMD", "KO", "PEP", "TMO",
    "WFC", "LIN", "ADBE", "ACN", "MCD", "DIS", "CSCO", "ABT", "PM", "INTU",
    "IBM", "GE", "CAT", "VZ", "AMGN", "QCOM", "TXN", "NOW", "PFE", "SPGI",
    "UNP", "UBER", "GS", "AXP", "ISRG", "COP", "LOW", "HON", "CMCSA", "T",
    "BKNG", "MS", "SYK", "RTX", "BLK", "ELV", "NEE", "TJX", "PGR", "PLD",
    "SCHW", "VRTX", "C", "ETN", "LRCX", "BA", "MDT", "ADP", "BMY", "CB",
    "DE", "MMC", "AMAT", "SBUX", "PANW", "GILD", "MU", "REGN", "CI", "FI",
    "ADI", "TMUS", "BSX", "KLAC", "LMT", "ZTS", "MDLZ", "SNPS", "SO", "CDNS"
]

@st.cache_data(ttl=86400)
def get_sp500_constituents():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        cols = {"Symbol": "Ticker", "Security": "Company", "GICS Sector": "Sector"}
        return df[["Symbol", "Security", "GICS Sector"]].rename(columns=cols)
    except Exception:
        return pd.DataFrame({
            "Ticker": DEFAULT_TOP_SP500,
            "Company": [f"{t} Corp." for t in DEFAULT_TOP_SP500],
            "Sector": ["Large Cap US Equity"] * len(DEFAULT_TOP_SP500)
        })

@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(tickers, start_date, end_date):
    raw_dl = yf.download(
        tickers, start=start_date, end=end_date,
        auto_adjust=True, progress=False, threads=True
    )
    if raw_dl.empty:
        return pd.DataFrame()
    if isinstance(raw_dl.columns, pd.MultiIndex):
        prices = raw_dl["Close"]
    else:
        t_name = tickers[0] if isinstance(tickers, list) else str(tickers)
        prices = raw_dl[["Close"]].rename(columns={"Close": t_name})
    prices = prices.dropna(how="all").ffill()
    valid_cols = prices.columns[prices.notna().sum() > 80]
    return prices[valid_cols]

@st.cache_data(show_spinner=False)
def generate_synthetic_sp500_data(tickers, start_date, end_date):
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_stocks = len(tickers)
    rng = np.random.default_rng(42)

    mkt_ret = rng.normal(0.0005, 0.010, size=n_days)
    t = np.linspace(0, 6 * np.pi, n_days)
    mkt_ret += 0.0008 * np.sin(t)

    betas = rng.uniform(0.6, 1.4, size=n_stocks)
    idio_vol = rng.uniform(0.008, 0.022, size=n_stocks)
    stock_drifts = rng.normal(0.0002, 0.0003, size=n_stocks)

    raw_noise = rng.normal(0, 1, size=(n_days, n_stocks)) * idio_vol
    rev_noise = np.zeros_like(raw_noise)
    for d in range(1, n_days):
        rev_noise[d] = raw_noise[d] - 0.15 * raw_noise[d - 1]

    rets = np.outer(mkt_ret, betas) + stock_drifts + rev_noise
    init_prices = rng.uniform(15.0, 450.0, size=n_stocks)
    price_matrix = init_prices * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(price_matrix, index=dates, columns=tickers)

# ==============================================================================
# STREAMLINED STRATEGY ENGINE WITH CUSTOM EXITS
# ==============================================================================
def run_streamlined_engine(
    prices: pd.DataFrame,
    rev_lookback: int = 10,
    regime_lookback: int = 63,
    regime_thresh: float = 0.60,
    exit_mode: str = "Regime Failure",
    exit_param: int = 20,
    portfolio_mode: str = "Long / Short (Market Neutral)",
    tc_bps: float = 0.6,
    train_ratio: float = 0.65,
    target_vol: float = 0.12,
    target_dd: float = 0.15,
    enable_scaling: bool = True,
):
    daily_rets = prices.pct_change()

    # 1. Reversal Component: Trailing N-day return negated & standardized
    trailing_n_ret = prices.pct_change(periods=rev_lookback)
    raw_rev = -trailing_n_ret
    rev_mean = raw_rev.mean(axis=1)
    rev_std = raw_rev.std(axis=1).replace(0, np.nan)
    reversal_z = raw_rev.sub(rev_mean, axis=0).div(rev_std, axis=0)

    # 2. Regime Filter (UpFraction > Threshold)
    is_up_day = (daily_rets > 0).astype(float)
    up_fraction = is_up_day.rolling(window=regime_lookback, min_periods=regime_lookback).mean()
    regime_filter = (up_fraction > regime_thresh).astype(float)

    # Base signal is purely the reversal score, gated by the regime filter
    edge_raw = reversal_z.where(regime_filter == 1.0, np.nan)

    # Cross-sectional Z-score among active stocks
    edge_mean = edge_raw.mean(axis=1)
    edge_std = edge_raw.std(axis=1).replace(0, np.nan)
    edge_z = edge_raw.sub(edge_mean, axis=0).div(edge_std, axis=0)

    warmup = max(regime_lookback, rev_lookback) + exit_param + 2
    valid_dates = prices.index[warmup:]

    # Simulation loop across days with stateful exit handling
    weights_records = []
    active_positions = {ticker: {"dir": 0, "entry_dt": None, "days_held": 0, "peak_p": 0.0} for ticker in prices.columns}

    rolling_highest = prices.rolling(window=exit_param).max()

    for dt in valid_dates:
        curr_edge_z = edge_z.loc[dt]
        curr_regime = regime_filter.loc[dt]
        curr_prices = prices.loc[dt]

        new_weights = {}

        for ticker in prices.columns:
            pos = active_positions[ticker]
            c_p = curr_prices[ticker]
            c_reg = curr_regime[ticker]
            c_z = curr_edge_z[ticker]

            if pos["dir"] == 0:
                # Look for entry
                if c_reg == 1.0 and pd.notna(c_z):
                    if c_z > 0:
                        pos["dir"] = 1
                        pos["entry_dt"] = dt
                        pos["days_held"] = 1
                        pos["peak_p"] = c_p
                    elif c_z < 0 and "Long / Short" in portfolio_mode:
                        pos["dir"] = -1
                        pos["entry_dt"] = dt
                        pos["days_held"] = 1
                        pos["peak_p"] = c_p
            else:
                # Check exit conditions
                pos["days_held"] += 1
                pos["peak_p"] = max(pos["peak_p"], c_p)

                exit_triggered = False
                if exit_mode == "Regime Failure":
                    if c_reg == 0.0:
                        exit_triggered = True
                elif exit_mode == "X-Day Highest Close":
                    h_val = rolling_highest.loc[dt, ticker]
                    if pd.notna(h_val) and c_p >= h_val * 0.999:  # hit new high
                        exit_triggered = True
                elif exit_mode == "Fixed Holding Period (Days)":
                    if pos["days_held"] >= exit_param:
                        exit_triggered = True

                # Also exit if signal flips sign strongly
                if pos["dir"] == 1 and pd.notna(c_z) and c_z < -0.5:
                    exit_triggered = True
                elif pos["dir"] == -1 and pd.notna(c_z) and c_z > 0.5:
                    exit_triggered = True

                if exit_triggered:
                    pos["dir"] = 0
                    pos["entry_dt"] = None
                    pos["days_held"] = 0

            new_weights[ticker] = pos["dir"]

        weights_records.append(new_weights)

    unscaled_weights_df = pd.DataFrame(weights_records, index=valid_dates)

    # Normalize weights to +50% Long / -50% Short (or +100% Long if Long-Only)
    pos_mask = unscaled_weights_df > 0
    neg_mask = unscaled_weights_df < 0

    pos_sums = pos_mask.sum(axis=1).replace(0, np.nan)
    neg_sums = neg_mask.sum(axis=1).replace(0, np.nan)

    long_alloc = 1.0 if "Long-Only" in portfolio_mode else 0.50
    short_alloc = 0.0 if "Long-Only" in portfolio_mode else 0.50

    long_w = unscaled_weights_df.where(pos_mask, 0.0).div(pos_sums, axis=0) * long_alloc
    short_w = unscaled_weights_df.where(neg_mask, 0.0).abs().div(neg_sums, axis=0) * short_alloc * -1.0

    final_unscaled_weights = long_w.fillna(0.0) + short_w.fillna(0.0)
    daily_rets_valid = daily_rets.loc[valid_dates]

    lagged_weights = final_unscaled_weights.shift(1).fillna(0.0)
    turnover = final_unscaled_weights.diff().abs().sum(axis=1).fillna(0.0)
    tc_rate = tc_bps / 10000.0

    raw_port_ret = (lagged_weights * daily_rets_valid).sum(axis=1) - (turnover * tc_rate)

    # Training vs OOS split for scaling factor s*
    split_idx = int(len(valid_dates) * train_ratio)
    split_date = valid_dates[split_idx]
    train_rets = raw_port_ret.iloc[:split_idx]

    train_vol = train_rets.std() * np.sqrt(252)
    train_cum = (1.0 + train_rets).cumprod()
    train_dd = (train_cum / train_cum.cummax() - 1.0).min()

    if enable_scaling and train_vol > 0 and abs(train_dd) > 0:
        s_star = float(min(target_vol / train_vol, target_dd / abs(train_dd)))
    else:
        s_star = 1.0

    scaled_ret = raw_port_ret * s_star
    scaled_weights = final_unscaled_weights * s_star

    equity_curve = (1.0 + scaled_ret).cumprod()
    drawdown_curve = (equity_curve / equity_curve.cummax()) - 1.0
    qualifying_pct = regime_filter.loc[valid_dates].mean(axis=1) * 100.0

    return {
        "prices": prices,
        "daily_rets": daily_rets,
        "reversal_z": reversal_z,
        "up_fraction": up_fraction,
        "regime_filter": regime_filter,
        "edge_z": edge_z,
        "unscaled_weights": final_unscaled_weights,
        "scaled_weights": scaled_weights,
        "final_ret": scaled_ret,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "qualifying_pct": qualifying_pct,
        "s_star": s_star,
        "split_date": split_date,
    }


def build_live_scanner(engine_out, meta_df, exit_param):
    prices = engine_out["prices"]
    weights = engine_out["scaled_weights"]
    up_fraction = engine_out["up_fraction"]
    regime_filter = engine_out["regime_filter"]
    edge_z = engine_out["edge_z"]

    latest_dt = weights.index[-1]
    prev_dt = weights.index[-2] if len(weights.index) > 1 else latest_dt

    pos_state = np.sign(weights)
    meta_map = meta_df.set_index("Ticker").to_dict(orient="index") if not meta_df.empty else {}
    records = []

    for ticker in prices.columns:
        curr_p = prices.loc[latest_dt, ticker]
        if pd.isna(curr_p):
            continue

        s_series = pos_state[ticker]
        curr_state = int(s_series.loc[latest_dt])
        prev_state = int(s_series.loc[prev_dt])

        if curr_state == 1 and prev_state <= 0:
            action = "🟢 BUY (New Long)"
        elif curr_state == 1 and prev_state == 1:
            action = "🔵 HOLD LONG"
        elif curr_state == -1 and prev_state >= 0:
            action = "🔴 SELL SHORT"
        elif curr_state == -1 and prev_state == -1:
            action = "🟠 HOLD SHORT"
        elif curr_state == 0 and prev_state != 0:
            action = "⚪ EXIT POSITION"
        else:
            action = "⚫ FLAT"

        info = meta_map.get(ticker, {"Company": ticker, "Sector": "US Equity"})

        records.append({
            "Ticker": ticker,
            "Company": info.get("Company", ticker),
            "Sector": info.get("Sector", "US Equity"),
            "Action_Signal": action,
            "Current_Price": float(curr_p),
            "Up_Fraction_Pct": float(up_fraction.loc[latest_dt, ticker] * 100.0) if pd.notna(up_fraction.loc[latest_dt, ticker]) else np.nan,
            "Regime_Active": bool(regime_filter.loc[latest_dt, ticker] == 1.0),
            "EDGE_Z_Score": float(edge_z.loc[latest_dt, ticker]) if pd.notna(edge_z.loc[latest_dt, ticker]) else np.nan,
            "Target_Weight_Pct": float(weights.loc[latest_dt, ticker] * 100.0),
        })

    df = pd.DataFrame(records).sort_values(by="EDGE_Z_Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df, latest_dt


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Strategy Parameters")

    data_mode = st.radio(
        "Market Data Feed",
        options=["⚡ Instant Demo Mode (Synthetic S&P 500)", "🌐 Live Yahoo Finance (Top 100 S&P 500)"],
        index=0
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=365 * 4))
    with col_d2:
        end_date = st.date_input("End Date", value=datetime.today())

    st.divider()
    st.subheader("1. Portfolio & Strategy Mode")
    portfolio_mode = st.selectbox(
        "Portfolio Type",
        options=["Long-Only Swing Buyer", "Long / Short (Market Neutral)"],
        index=0,
        help="Long-Only avoids shorting strong uptrending stocks (recommended for real S&P 500 data)."
    )

    st.subheader("2. Reversal & Regime Settings")
    rev_lookback = st.number_input("Reversal Lookback (Days)", min_value=2, max_value=30, value=10, step=1)
    regime_lookback = st.number_input("Regime Window (Days)", min_value=20, max_value=252, value=63, step=1)
    regime_thresh_pct = st.slider("Drift Regime Threshold (%)", min_value=40.0, max_value=80.0, value=60.0, step=1.0)

    st.subheader("3. Adjustable Exit Rule")
    exit_mode = st.selectbox(
        "Exit Trigger Mode",
        options=["X-Day Highest Close", "Regime Failure", "Fixed Holding Period (Days)"],
        index=0,
        help="Choose how open positions are exited."
    )
    exit_param = st.number_input(
        "Exit Parameter (Days / Lookback)",
        min_value=2, max_value=120, value=20, step=1,
        help="Lookback window for highest close or maximum holding days."
    )

    tc_bps = st.number_input("Transaction Costs (bps)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)

# ==============================================================================
# LOAD DATA & RUN
# ==============================================================================
meta_df = get_sp500_constituents()

with st.spinner("Running streamlined engine..."):
    if "Instant Demo" in data_mode:
        tickers = meta_df["Ticker"].iloc[:100].tolist()
        prices_df = generate_synthetic_sp500_data(tickers, start_date, end_date)
    else:
        tickers = DEFAULT_TOP_SP500
        prices_df = download_price_data(tickers, start_date, end_date)

    engine_res = run_streamlined_engine(
        prices=prices_df,
        rev_lookback=int(rev_lookback),
        regime_lookback=int(regime_lookback),
        regime_thresh=regime_thresh_pct / 100.0,
        exit_mode=exit_mode,
        exit_param=int(exit_param),
        portfolio_mode=portfolio_mode,
        tc_bps=tc_bps,
    )
    scanner_df, latest_dt = build_live_scanner(engine_res, meta_df, int(exit_param))

# ==============================================================================
# TABS DISPLAY
# ==============================================================================
tab1, tab2 = st.tabs(["📈 Backtest Performance", "🎯 Live Signal Scanner"])

with tab1:
    ret = engine_res["final_ret"]
    eq = engine_res["equity_curve"]
    dd = engine_res["drawdown_curve"]
    split_dt = engine_res["split_date"]

    tot_r = (eq.iloc[-1] - 1.0) * 100.0
    ann_r = ((eq.iloc[-1]) ** (252.0 / len(ret)) - 1.0) * 100.0
    ann_v = ret.std() * np.sqrt(252) * 100.0
    sharpe = (ann_r / ann_v) if ann_v > 0 else 0.0
    m_dd = dd.min() * 100.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cumulative Return", f"{tot_r:+.2f}%")
    c2.metric("CAGR", f"{ann_r:+.2f}%")
    c3.metric("Annual Volatility", f"{ann_v:.2f}%")
    c4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c5.metric("Max Drawdown", f"{m_dd:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy Equity ($1)", line=dict(color="#00CC96", width=2)))
    fig.add_vline(x=split_dt, line_dash="dash", line_color="orange", annotation_text="Train/Test Split")
    fig.update_layout(height=500, template="plotly_dark", title="Streamlined Strategy Equity Curve", margin=dict(l=30, r=30, t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"S&P 500 Signal Scanner ({latest_dt.strftime('%Y-%m-%d')})")
    st.dataframe(scanner_df, use_container_width=True, height=500, hide_index=True)
