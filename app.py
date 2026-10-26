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
    page_title="Unicorn Edge Strategy — Backtester & S&P 500 Scanner",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦄 The 'Unicorn Edge' Quantitative Strategy Suite")
st.markdown(
    """
    **Market-Neutral Regime-Conditioned Value & Short-Term Reversal System**  
    *Hover over any parameter `(?)` icon in the sidebar or **any column header** in the tables below for a simple explanation of how each metric works.*
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
    """Fetches current S&P 500 constituents from Wikipedia with fallback."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "Security", "GICS Sector"]].rename(
            columns={"Symbol": "Ticker", "Security": "Company", "GICS Sector": "Sector"}
        )
    except Exception:
        return pd.DataFrame({
            "Ticker": DEFAULT_TOP_SP500,
            "Company": [f"{t} Corp." for t in DEFAULT_TOP_SP500],
            "Sector": ["Large Cap US Equity"] * len(DEFAULT_TOP_SP500)
        })

@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(tickers, start_date, end_date):
    """Downloads daily adjusted close prices via yfinance."""
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all").ffill()
    # Keep columns with sufficient data
    valid_cols = prices.columns[prices.notna().sum() > 80]
    return prices[valid_cols]

@st.cache_data(show_spinner=False)
def generate_synthetic_sp500_data(tickers, start_date, end_date):
    """Generates realistic correlated stock price trajectories for instant demoing."""
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_stocks = len(tickers)
    rng = np.random.default_rng(42)

    # Market factor with regimes (bull trends & brief sharp drawdowns)
    mkt_ret = rng.normal(0.0005, 0.010, size=n_days)
    # Add periodic market drift waves
    t = np.linspace(0, 6 * np.pi, n_days)
    mkt_ret += 0.0008 * np.sin(t)

    betas = rng.uniform(0.6, 1.4, size=n_stocks)
    idio_vol = rng.uniform(0.008, 0.022, size=n_stocks)
    stock_drifts = rng.normal(0.0002, 0.0003, size=n_stocks)

    # Mean-reverting idiosyncratic noise (creates realistic short-term reversal alpha)
    raw_noise = rng.normal(0, 1, size=(n_days, n_stocks)) * idio_vol
    rev_noise = np.zeros_like(raw_noise)
    for d in range(1, n_days):
        rev_noise[d] = raw_noise[d] - 0.15 * raw_noise[d - 1]

    rets = np.outer(mkt_ret, betas) + stock_drifts + rev_noise
    init_prices = rng.uniform(15.0, 450.0, size=n_stocks)
    price_matrix = init_prices * np.exp(np.cumsum(rets, axis=0))

    return pd.DataFrame(price_matrix, index=dates, columns=tickers)

# ==============================================================================
# CORE STRATEGY ENGINE (EQUATIONS 1 TO 5)
# ==============================================================================
def run_unicorn_edge_engine(
    prices: pd.DataFrame,
    val_weight: float = 0.70,
    rev_lookback: int = 10,
    regime_lookback: int = 63,
    regime_thresh: float = 0.60,
    tc_bps: float = 0.6,
    train_Ratio: float = 0.65,
    target_vol: float = 0.12,
    target_dd: float = 0.15,
    enable_scaling: bool = True,
    enable_killswitch: bool = True,
    ks_max_dd: float = 0.30,
    ks_roll_loss: float = -0.10,
    ks_roll_window: int = 63,
):
    """
    Executes the full Unicorn Edge strategy across all dates and constituents:
    1) Value_i,t = Cross-sectional Percentile Rank of (1 / Price_i,t)
    2) Reversal_i,t = Cross-sectional Z-Score of (-1 * R_{i, t-rev_lookback -> t})
    3) BASE_i,t = val_weight * Value_i,t + (1 - val_weight) * Reversal_i,t
    4) UpFraction_i,t = Fraction of days with daily return > 0 over trailing `regime_lookback` days
    5) REGIME_i,t = 1 if UpFraction_i,t > regime_thresh else 0
    6) EDGE_i,t = BASE_i,t * REGIME_i,t
    7) Cross-sectional Z-score among active (REGIME == 1) stocks -> Long (>0) normalized to +0.50, Short (<0) normalized to -0.50.
    """
    daily_rets = prices.pct_change()

    # 1. Value Component: Inverse price -> cross-sectional percentile rank [0, 1]
    inv_price = 1.0 / prices.replace(0, np.nan)
    value_rank = inv_price.rank(axis=1, pct=True)

    # 2. Reversal Component: Negated N-day return -> cross-sectional z-score
    trailing_n_ret = prices.pct_change(periods=rev_lookback)
    raw_rev = -trailing_n_ret
    rev_mean = raw_rev.mean(axis=1)
    rev_std = raw_rev.std(axis=1).replace(0, np.nan)
    reversal_z = raw_rev.sub(rev_mean, axis=0).div(rev_std, axis=0)

    # Equation (1): BASE Factor
    rev_weight = round(1.0 - val_weight, 4)
    base_factor = (val_weight * value_rank) + (rev_weight * reversal_z)

    # Equation (2) & (3): UpFraction and Drift Regime Filter
    is_up_day = (daily_rets > 0).astype(float)
    up_fraction = is_up_day.rolling(window=regime_lookback, min_periods=regime_lookback).mean()
    regime_filter = (up_fraction > regime_thresh).astype(float)

    # Equation (4): EDGE Signal (Masked where REGIME == 0)
    edge_raw = base_factor.where(regime_filter == 1.0, np.nan)

    # Section 2.2: Cross-sectional Z-score of valid, non-zero EDGE scores
    edge_mean = edge_raw.mean(axis=1)
    edge_std = edge_raw.std(axis=1).replace(0, np.nan)
    edge_z = edge_raw.sub(edge_mean, axis=0).div(edge_std, axis=0)

    # Separate into Long (Z > 0) and Short (Z < 0) buckets and normalize to +50% / -50%
    pos_z = edge_z.where(edge_z > 0, 0.0)
    neg_z = edge_z.where(edge_z < 0, 0.0)

    pos_sum = pos_z.sum(axis=1).replace(0, np.nan)
    neg_sum = neg_z.abs().sum(axis=1).replace(0, np.nan)

    long_weights = pos_z.div(pos_sum, axis=0) * 0.50
    short_weights = neg_z.div(neg_sum, axis=0) * 0.50  # negative values summing to -0.50

    unscaled_weights = (long_weights.fillna(0.0) + short_weights.fillna(0.0))

    # Trim warm-up rows
    warmup = max(regime_lookback, rev_lookback) + 2
    valid_dates = prices.index[warmup:]
    unscaled_weights = unscaled_weights.loc[valid_dates]
    daily_rets_valid = daily_rets.loc[valid_dates]

    # Portfolio returns before scaling (weights at t-1 earn returns at t)
    lagged_weights = unscaled_weights.shift(1).fillna(0.0)
    turnover_unscaled = unscaled_weights.diff().abs().sum(axis=1).fillna(0.0)
    tc_rate = tc_bps / 10000.0

    raw_port_ret = (lagged_weights * daily_rets_valid).sum(axis=1) - (turnover_unscaled * tc_rate)

    # Split into Training and Out-of-Sample Testing periods for Equation (5) calibration
    split_idx = int(len(valid_dates) * train_Ratio)
    split_date = valid_dates[split_idx]
    train_rets = raw_port_ret.iloc[:split_idx]

    train_vol = train_rets.std() * np.sqrt(252)
    train_cum = (1.0 + train_rets).cumprod()
    train_dd = (train_cum / train_cum.cummax() - 1.0).min()

    # Equation (5): Static Scaling Factor s*
    if enable_scaling and train_vol > 0 and abs(train_dd) > 0:
        s_star = float(min(target_vol / train_vol, target_dd / abs(train_dd)))
    else:
        s_star = 1.0

    scaled_ret = raw_port_ret * s_star
    scaled_weights = unscaled_weights * s_star

    # Dynamic Kill-Switch Simulation (30% Peak DD or -10% rolling 63-day return)
    final_ret = scaled_ret.copy()
    ks_triggered_date = None
    ks_reason = None
    ks_active_series = pd.Series(False, index=valid_dates)

    if enable_killswitch:
        eq = 1.0
        peak = 1.0
        ret_list = []
        killed = False
        for dt, r in scaled_ret.items():
            if killed:
                final_ret.loc[dt] = 0.0
                ks_active_series.loc[dt] = True
                continue

            eq *= (1.0 + r)
            peak = max(peak, eq)
            curr_dd = (eq / peak) - 1.0
            ret_list.append(r)

            roll_ret = 0.0
            if len(ret_list) >= ks_roll_window:
                window_slice = ret_list[-ks_roll_window:]
                roll_ret = np.prod([1.0 + x for x in window_slice]) - 1.0

            if abs(curr_dd) >= ks_max_dd:
                killed = True
                ks_triggered_date = dt
                ks_reason = f"Max Drawdown Exceeded ({curr_dd*100:.1f}% vs -{ks_max_dd*100:.0f}% limit)"
            elif roll_ret <= ks_roll_loss:
                killed = True
                ks_triggered_date = dt
                ks_reason = f"Rolling {ks_roll_window}d Return Breached ({roll_ret*100:.1f}% vs {ks_roll_loss*100:.0f}% limit)"

    # Construct Equity Curve & Drawdown
    equity_curve = (1.0 + final_ret).cumprod()
    drawdown_curve = (equity_curve / equity_curve.cummax()) - 1.0
    qualifying_pct = regime_filter.loc[valid_dates].mean(axis=1) * 100.0

    return {
        "prices": prices,
        "daily_rets": daily_rets,
        "value_rank": value_rank,
        "trailing_n_ret": trailing_n_ret,
        "reversal_z": reversal_z,
        "base_factor": base_factor,
        "up_fraction": up_fraction,
        "regime_filter": regime_filter,
        "edge_raw": edge_raw,
        "edge_z": edge_z,
        "unscaled_weights": unscaled_weights,
        "scaled_weights": scaled_weights,
        "final_ret": final_ret,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "qualifying_pct": qualifying_pct,
        "s_star": s_star,
        "train_vol": train_vol,
        "train_dd": train_dd,
        "split_date": split_date,
        "ks_triggered_date": ks_triggered_date,
        "ks_reason": ks_reason,
        "ks_active_series": ks_active_series,
    }


def build_scanner_and_trade_ledger(engine_out: dict, meta_df: pd.DataFrame):
    """
    Builds the comprehensive S&P 500 Signal Scanner table for the latest date,
    including historical trade state tracking (Buy, Hold Long, Sell Short, Cover Short,
    Exit Long, Latest Entry Date, Latest Entry Price, and Open Trade PnL).
    """
    prices = engine_out["prices"]
    weights = engine_out["scaled_weights"]
    value_rank = engine_out["value_rank"]
    trailing_n_ret = engine_out["trailing_n_ret"]
    reversal_z = engine_out["reversal_z"]
    base_factor = engine_out["base_factor"]
    up_fraction = engine_out["up_fraction"]
    regime_filter = engine_out["regime_filter"]
    edge_raw = engine_out["edge_raw"]
    edge_z = engine_out["edge_z"]

    latest_dt = weights.index[-1]
    prev_dt = weights.index[-2] if len(weights.index) > 1 else latest_dt

    # Determine position states (+1 = Long, -1 = Short, 0 = Flat) across time
    pos_state = np.sign(weights)

    records = []
    meta_map = meta_df.set_index("Ticker").to_dict(orient="index") if not meta_df.empty else {}

    for ticker in prices.columns:
        curr_p = prices.loc[latest_dt, ticker]
        if pd.isna(curr_p):
            continue

        s_Series = pos_state[ticker]
        p_Series = prices[ticker].loc[weights.index]

        curr_state = int(s_Series.loc[latest_dt])
        prev_state = int(s_Series.loc[prev_dt])

        # Determine Action Signal
        if curr_state == 1 and prev_state <= 0:
            action = "🟢 BUY (New Long)"
            trade_status = "Open Long"
        elif curr_state == 1 and prev_state == 1:
            action = "🔵 HOLD LONG"
            trade_status = "Open Long"
        elif curr_state == -1 and prev_state >= 0:
            action = "🔴 SELL SHORT (New Short)"
            trade_status = "Open Short"
        elif curr_state == -1 and prev_state == -1:
            action = "🟠 HOLD SHORT"
            trade_status = "Open Short"
        elif curr_state == 0 and prev_state == 1:
            action = "⚪ EXIT LONG (Sell to Close)"
            trade_status = "Closed Today"
        elif curr_state == 0 and prev_state == -1:
            action = "🟣 COVER SHORT (Buy to Close)"
            trade_status = "Closed Today"
        else:
            action = "⚫ FLAT (Filtered Out)"
            trade_status = "Flat / No Trade"

        # Trace back to find Latest Entry Date, Entry Price, Days Held, and Total Trades
        state_vals = s_Series.values
        date_vals = s_Series.index
        price_vals = p_Series.values

        # Count total historical trade entries
        transitions = (state_vals[1:] != state_vals[:-1]) & (state_vals[1:] != 0)
        total_trades_count = int(np.sum(transitions)) + (1 if state_vals[0] != 0 else 0)

        entry_date_str = "—"
        entry_price_val = np.nan
        open_pnl_pct = np.nan
        days_in_trade = 0

        if curr_state != 0:
            # Walk backward to find when current continuous position started
            idx = len(state_vals) - 1
            while idx > 0 and state_vals[idx - 1] == curr_state:
                idx -= 1
            entry_dt = date_vals[idx]
            entry_date_str = pd.Timestamp(entry_dt).strftime("%Y-%m-%d")
            entry_price_val = float(price_vals[idx])
            days_in_trade = int(len(state_vals) - idx)
            if entry_price_val > 0:
                raw_move = (curr_p - entry_price_val) / entry_price_val
                open_pnl_pct = (raw_move * 100.0) if curr_state == 1 else (-raw_move * 100.0)
        else:
            # Walk backward to find the most recent historical entry even if currently closed
            non_zero_indices = np.where(state_vals != 0)[0]
            if len(non_zero_indices) > 0:
                last_active_idx = non_zero_indices[-1]
                last_dir = state_vals[last_active_idx]
                idx = last_active_idx
                while idx > 0 and state_vals[idx - 1] == last_dir:
                    idx -= 1
                entry_dt = date_vals[idx]
                entry_date_str = pd.Timestamp(entry_dt).strftime("%Y-%m-%d")
                entry_price_val = float(price_vals[idx])

        info = meta_map.get(ticker, {"Company": f"{ticker}", "Sector": "US Equity"})

        records.append({
            "Ticker": ticker,
            "Company": info.get("Company", ticker),
            "Sector": info.get("Sector", "US Equity"),
            "Action_Signal": action,
            "Open_Trade_Status": trade_status,
            "Current_Price": float(curr_p),
            "Up_Fraction_Pct": float(up_fraction.loc[latest_dt, ticker] * 100.0) if pd.notna(up_fraction.loc[latest_dt, ticker]) else np.nan,
            "Regime_Active": bool(regime_filter.loc[latest_dt, ticker] == 1.0),
            "Value_Percentile": float(value_rank.loc[latest_dt, ticker] * 100.0) if pd.notna(value_rank.loc[latest_dt, ticker]) else np.nan,
            "Ret_10d_Pct": float(trailing_n_ret.loc[latest_dt, ticker] * 100.0) if pd.notna(trailing_n_ret.loc[latest_dt, ticker]) else np.nan,
            "Reversal_Z": float(reversal_z.loc[latest_dt, ticker]) if pd.notna(reversal_z.loc[latest_dt, ticker]) else np.nan,
            "BASE_Score": float(base_factor.loc[latest_dt, ticker]) if pd.notna(base_factor.loc[latest_dt, ticker]) else np.nan,
            "EDGE_Score": float(edge_raw.loc[latest_dt, ticker]) if pd.notna(edge_raw.loc[latest_dt, ticker]) else 0.0,
            "EDGE_Z_Score": float(edge_z.loc[latest_dt, ticker]) if pd.notna(edge_z.loc[latest_dt, ticker]) else np.nan,
            "Target_Weight_Pct": float(weights.loc[latest_dt, ticker] * 100.0),
            "Latest_Entry_Date": entry_date_str,
            "Latest_Entry_Price": entry_price_val,
            "Days_Held": days_in_trade,
            "Open_Trade_PnL_Pct": open_pnl_pct,
            "Total_Historical_Trades": total_trades_count,
        })

    scanner_df = pd.DataFrame(records)
    # Sort by EDGE_Z_Score descending (highest conviction longs at top, shorts next, filtered at bottom)
    scanner_df = scanner_df.sort_values(
        by=["Regime_Active", "EDGE_Z_Score", "BASE_Score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    scanner_df.insert(0, "Rank", np.arange(1, len(scanner_df) + 1))
    return scanner_df, latest_dt


# ==============================================================================
# SIDEBAR CONTROLS (WITH DETAILED HOVER TOOLTIPS)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Strategy Parameters")

    st.subheader("1. Data Source & Universe")
    data_mode = st.radio(
        "Select Market Data Feed",
        options=[
            "⚡ Instant Demo Mode (Synthetic S&P 500)",
            "🌐 Live Yahoo Finance (Top 100 S&P 500)",
            "🌐 Live Yahoo Finance (Full S&P 500 - Slower)"
        ],
        index=0,
        help="Choose 'Instant Demo Mode' for immediate testing, or 'Live Yahoo Finance' to download real historical daily prices."
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.today() - timedelta(days=365 * 4),
            help="Beginning date for historical backtest and training calibration."
        )
    with col_d2:
        end_date = st.date_input(
            "End Date",
            value=datetime.today(),
            help="End date for the backtest and the reference date for the live S&P 500 signal scanner."
        )

    st.divider()
    st.subheader("2. Signal Construction (Sec 2.1)")

    val_weight = st.slider(
        "Value Weight (w_val)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Equation (1): Weight assigned to the inverse-price Value percentile rank. The paper uses 0.70 (70%). Reversal automatically receives (1 - w_val)."
    )
    st.caption(f"Computed Formula: `BASE = {val_weight:.2f} × Value + {1-val_weight:.2f} × Reversal`")

    rev_lookback = st.number_input(
        "Reversal Lookback (Trading Days)",
        min_value=2,
        max_value=60,
        value=10,
        step=1,
        help="Number of trailing trading days used to calculate short-term losers vs. winners. The paper uses 10 trading days (~2 weeks)."
    )

    regime_lookback = st.number_input(
        "Regime Filter Window (Trading Days)",
        min_value=20,
        max_value=252,
        value=63,
        step=1,
        help="Equation (2): Trailing window over which the fraction of positive return ('green') days is calculated. The paper uses 63 days (~3 months)."
    )

    regime_thresh_pct = st.slider(
        "Drift Regime Threshold (UpFraction %)",
        min_value=40.0,
        max_value=80.0,
        value=60.0,
        step=1.0,
        help="Equation (3): Minimum percentage of positive return days required over the Regime Window for a stock to qualify for trading. The paper uses 60%."
    )

    st.divider()
    st.subheader("3. Risk & Kill-Switch Controls (Sec 2.2)")

    enable_scaling = st.checkbox(
        "Enable Static Vol/DD Scaling (s*)",
        value=True,
        help="Equation (5): Automatically scales total portfolio exposure based on training-period volatility and max drawdown."
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        target_vol_pct = st.number_input(
            "Vol Cap (%)",
            min_value=2.0,
            max_value=40.0,
            value=12.0,
            step=1.0,
            help="Equation (5): Maximum annualized volatility target during training calibration (default 12%)."
        )
    with col_r2:
        target_dd_pct = st.number_input(
            "Max DD Cap (%)",
            min_value=3.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Equation (5): Maximum historical drawdown constraint during training calibration (default 15%)."
        )

    enable_killswitch = st.checkbox(
        "Enable Dynamic Kill-Switch",
        value=True,
        help="Liquidates the portfolio and halts trading if live drawdown or rolling 63-day losses breach emergency thresholds."
    )

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        ks_dd_pct = st.number_input(
            "Kill Max DD (%)",
            min_value=5.0,
            max_value=60.0,
            value=30.0,
            step=1.0,
            help="Dynamic Kill-Switch trigger if absolute peak-to-trough drawdown exceeds this percentage (paper default: 30%)."
        )
    with col_k2:
        ks_roll_pct = st.number_input(
            "Kill 63d Loss (%)",
            min_value=-40.0,
            max_value=-2.0,
            value=-10.0,
            step=1.0,
            help="Dynamic Kill-Switch trigger if rolling 63-day return drops below this percentage (paper default: -10%)."
        )

    tc_bps = st.number_input(
        "Transaction Costs (bps per unit traded)",
        min_value=0.0,
        max_value=50.0,
        value=0.6,
        step=0.2,
        help="Combined explicit and implicit trading costs deducted on daily weight changes (paper default: 0.6 bps)."
    )

    train_split_pct = st.slider(
        "Walk-Forward Training Split (%)",
        min_value=30,
        max_value=85,
        value=70,
        step=5,
        help="Percentage of historical window used to estimate static risk scaling factor s* before freezing parameters for out-of-sample testing."
    )

# ==============================================================================
# LOAD DATA & RUN ENGINE
# ==============================================================================
sp500_meta = get_sp500_constituents()

with st.spinner("Loading constituent price history & computing Unicorn Edge signals..."):
    if "Instant Demo" in data_mode:
        selected_tickers = sp500_meta["Ticker"].iloc[:120].tolist()
        prices_df = generate_synthetic_sp500_data(selected_tickers, start_date, end_date)
    elif "Top 100" in data_mode:
        selected_tickers = DEFAULT_TOP_SP500
        prices_df = download_price_data(selected_tickers, start_date, end_date)
    else:
        selected_tickers = sp500_meta["Ticker"].tolist()
        prices_df = download_price_data(selected_tickers, start_date, end_date)

    if prices_df.empty or len(prices_df) < (regime_lookback + rev_lookback + 10):
        st.error("Not enough historical trading days in the selected date range to compute the lookback windows. Please widen the Start/End dates.")
        st.stop()

    engine_results = run_unicorn_edge_engine(
        prices=prices_df,
        val_weight=val_weight,
        rev_lookback=int(rev_lookback),
        regime_lookback=int(regime_lookback),
        regime_thresh=regime_thresh_pct / 100.0,
        tc_bps=tc_bps,
        train_Ratio=train_split_pct / 100.0,
        target_vol=target_vol_pct / 100.0,
        target_dd=target_dd_pct / 100.0,
        enable_scaling=enable_scaling,
        enable_killswitch=enable_killswitch,
        ks_max_dd=ks_dd_pct / 100.0,
        ks_roll_loss=ks_roll_pct / 100.0,
        ks_roll_window=int(regime_lookback),
    )

    scanner_df, latest_market_date = build_scanner_and_trade_ledger(engine_results, sp500_meta)

# ==============================================================================
# NAVIGATION TABS
# ==============================================================================
tab_bt, tab_scan, tab_guide = st.tabs([
    "📈 Tab 1: Strategy Backtester & Walk-Forward Analysis",
    "🎯 Tab 2: S&P 500 Live Signal Scanner & Open Trades",
    "📘 Tab 3: Strategy Equations & Column Glossary"
])

# ==============================================================================
# TAB 1: STRATEGY BACKTESTER
# ==============================================================================
with tab_bt:
    ret_series = engine_results["final_ret"]
    eq_series = engine_results["equity_curve"]
    dd_series = engine_results["drawdown_curve"]
    split_dt = engine_results["split_date"]

    # Calculate Full, Training, and OOS Test performance metrics
    total_ret = (eq_series.iloc[-1] - 1.0) * 100.0
    ann_ret = ((eq_series.iloc[-1]) ** (252.0 / len(ret_series)) - 1.0) * 100.0
    ann_vol = ret_series.std() * np.sqrt(252) * 100.0
    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
    max_dd = dd_series.min() * 100.0
    avg_qualifying = engine_results["qualifying_pct"].mean()

    st.subheader("Performance Summary & Risk Calibration")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(
        "Cumulative Return",
        f"{total_ret:+.2f}%",
        help="Total compounded return of the market-neutral long-short strategy after transaction costs."
    )
    m2.metric(
        "Annualized Return (CAGR)",
        f"{ann_ret:+.2f}%",
        help="Compound annual growth rate assuming 252 trading days per year."
    )
    m3.metric(
        "Annualized Volatility",
        f"{ann_vol:.2f}%",
        delta=f"Target: {target_vol_pct:.0f}%",
        delta_color="off",
        help="Realized annualized standard deviation of daily returns. Target cap is 12%."
    )
    m4.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}",
        help="Annualized return divided by annualized volatility (assuming 0% risk-free rate on dollar-neutral book)."
    )
    m5.metric(
        "Maximum Drawdown",
        f"{max_dd:.2f}%",
        delta=f"Cap: -{target_dd_pct:.0f}%",
        delta_color="off",
        help="Worst peak-to-trough percentage drop in portfolio equity."
    )
    m6.metric(
        "Scaling Multiplier (s*)",
        f"{engine_results['s_star']:.2f}x",
        delta=f"Avg Active: {avg_qualifying:.1f}%",
        delta_color="off",
        help="Equation (5): Static leverage multiplier derived from Training Volatility and Training Max Drawdown."
    )

    if engine_results["ks_triggered_date"] is not None:
        st.error(
            f"🚨 **DYNAMIC KILL-SWITCH TRIGGERED** on **{engine_results['ks_triggered_date'].strftime('%Y-%m-%d')}** — "
            f"Reason: *{engine_results['ks_reason']}*. Portfolio liquidated to 100% cash for the remainder of the test."
        )

    # Interactive 3-Panel Plotly Chart
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.50, 0.25, 0.25],
        subplot_titles=(
            "Cumulative Equity Curve (Training vs. Out-of-Sample Testing)",
            "Portfolio Underwater Drawdown (%)",
            f"Breadth of Drift Regime (% of Stocks with >{regime_thresh_pct:.0f}% Up Days over {regime_lookback}d)"
        )
    )

    # Panel 1: Equity Curve
    fig.add_trace(
        go.Scatter(
            x=eq_series.index, y=eq_series.values,
            mode="lines", name="Strategy Equity ($1 Initial)",
            line=dict(color="#00CC96", width=2.5)
        ),
        row=1, col=1
    )

    # Highlight Training vs OOS Split
    fig.add_vline(
        x=split_dt, line_width=2, line_dash="dash", line_color="#FFA15A"
    )
    fig.add_annotation(
        x=split_dt, y=eq_series.max(),
        text="Train ➔ Out-of-Sample Test Freeze",
        showarrow=True, arrowhead=2, ax=-60, ay=-25,
        font=dict(color="#FFA15A", size=12),
        row=1, col=1
    )

    # Panel 2: Drawdown
    fig.add_trace(
        go.Scatter(
            x=dd_series.index, y=dd_series.values * 100.0,
            mode="lines", name="Drawdown (%)",
            fill="tozeroy", line=dict(color="#EF553B", width=1.5)
        ),
        row=2, col=1
    )
    fig.add_hline(
        y=-target_dd_pct, line_dash="dot", line_color="orange",
        annotation_text=f"Training DD Cap (-{target_dd_pct:.0f}%)",
        row=2, col=1
    )

    # Panel 3: Qualifying Regime %
    fig.add_trace(
        go.Scatter(
            x=engine_results["qualifying_pct"].index,
            y=engine_results["qualifying_pct"].values,
            mode="lines", name="% Stocks in Drift Regime",
            fill="tozeroy", line=dict(color="#636EFA", width=1.5)
        ),
        row=3, col=1
    )
    fig.add_hline(
        y=35.0, line_dash="dot", line_color="gray",
        annotation_text="Paper Benchmark Avg (35%)",
        row=3, col=1
    )

    fig.update_layout(
        height=720,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=40, r=40, t=50, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Walk-Forward Period Breakdown Table
    st.subheader("Walk-Forward Validation Breakdown (In-Sample Training vs. Out-of-Sample Test)")
    train_mask = ret_series.index < split_dt
    test_mask = ret_series.index >= split_dt

    def calc_period_stats(r_sub, label):
        if len(r_sub) == 0:
            return {}
        c_eq = (1.0 + r_sub).cumprod()
        tot_r = (c_eq.iloc[-1] - 1.0) * 100.0
        a_r = (c_eq.iloc[-1] ** (252.0 / len(r_sub)) - 1.0) * 100.0
        a_v = r_sub.std() * np.sqrt(252) * 100.0
        sr = a_r / a_v if a_v > 0 else 0.0
        m_dd = ((c_eq / c_eq.cummax()) - 1.0).min() * 100.0
        win_days = (r_sub > 0).mean() * 100.0
        return {
            "Period": label,
            "Start_Date": r_sub.index[0].strftime("%Y-%m-%d"),
            "End_Date": r_sub.index[-1].strftime("%Y-%m-%d"),
            "Trading_Days": len(r_sub),
            "Total_Return_Pct": tot_r,
            "Annualized_Return_Pct": a_r,
            "Annualized_Vol_Pct": a_v,
            "Sharpe_Ratio": sr,
            "Max_Drawdown_Pct": m_dd,
            "Positive_Days_Pct": win_days,
        }

    wf_df = pd.DataFrame([
        calc_period_stats(ret_series.loc[train_mask], "1. Training Window (Parameter Calibration)"),
        calc_period_stats(ret_series.loc[test_mask], "2. Out-of-Sample Test Window (Frozen Rules)"),
        calc_period_stats(ret_series, "3. Combined Full Horizon"),
    ])

    st.dataframe(
        wf_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Period": st.column_config.TextColumn("Validation Phase", help="Walk-forward stage: Training calibrates s* scaling; Testing runs out-of-sample with frozen parameters."),
            "Start_Date": st.column_config.TextColumn("Start Date", help="First trading day of the evaluation segment."),
            "End_Date": st.column_config.TextColumn("End Date", help="Last trading day of the evaluation segment."),
            "Trading_Days": st.column_config.NumberColumn("Trading Days", help="Total count of active market days in the window."),
            "Total_Return_Pct": st.column_config.NumberColumn("Total Return (%)", format="%.2f%%", help="Cumulative compounded return over the window."),
            "Annualized_Return_Pct": st.column_config.NumberColumn("CAGR (%)", format="%.2f%%", help="Annualized compound growth rate."),
            "Annualized_Vol_Pct": st.column_config.NumberColumn("Annual Vol (%)", format="%.2f%%", help="Annualized volatility of daily strategy returns."),
            "Sharpe_Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f", help="Risk-adjusted return per unit of annualized volatility."),
            "Max_Drawdown_Pct": st.column_config.NumberColumn("Max Drawdown (%)", format="%.2f%%", help="Worst peak-to-valley decline during the window."),
            "Positive_Days_Pct": st.column_config.NumberColumn("Win Days (%)", format="%.1f%%", help="Percentage of trading days with positive net portfolio return."),
        }
    )


# ==============================================================================
# TAB 2: S&P 500 LIVE SIGNAL SCANNER & TRADE TRACKER
# ==============================================================================
with tab_scan:
    st.subheader(f"S&P 500 Cross-Sectional Signal Scanner & Trade Book (As of {latest_market_date.strftime('%Y-%m-%d')})")
    st.caption("💡 **Tip:** Hover your mouse over **any column header** below to read a plain-English explanation of what that metric means and how it is calculated.")

    # Summary KPI cards for today's signals
    buy_cnt = (scanner_df["Action_Signal"].str.contains("BUY|HOLD LONG")).sum()
    new_buy_cnt = (scanner_df["Action_Signal"].str.contains("BUY")).sum()
    short_cnt = (scanner_df["Action_Signal"].str.contains("SHORT")).sum()
    new_short_cnt = (scanner_df["Action_Signal"].str.contains("SELL SHORT")).sum()
    exit_cnt = (scanner_df["Action_Signal"].str.contains("EXIT|COVER")).sum()
    regime_pass_cnt = scanner_df["Regime_Active"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Stocks in Drift Regime",
        f"{regime_pass_cnt} / {len(scanner_df)} ({regime_pass_cnt/len(scanner_df)*100:.1f}%)",
        help=f"Number of stocks with >{regime_thresh_pct:.0f}% positive return days over the trailing {regime_lookback} trading days."
    )
    c2.metric(
        "Active Long Positions",
        f"{buy_cnt} Stocks",
        delta=f"+{new_buy_cnt} New Buy Signals Today",
        help="Stocks passing the Regime Filter with above-average EDGE scores (Z > 0), allocated 50% total long exposure."
    )
    c3.metric(
        "Active Short Positions",
        f"{short_cnt} Stocks",
        delta=f"+{new_short_cnt} New Short Signals Today",
        delta_color="inverse",
        help="Stocks passing the Regime Filter with below-average EDGE scores (Z < 0), allocated 50% total short exposure."
    )
    c4.metric(
        "Exits / Short Covers Today",
        f"{exit_cnt} Stocks",
        help="Stocks that were held yesterday but dropped out of the Long/Short buckets today (either due to regime failure or score shifts)."
    )
    c5.metric(
        "Net / Gross Exposure (Scaled)",
        f"Net: {scanner_df['Target_Weight_Pct'].sum():+.1f}% | Gross: {scanner_df['Target_Weight_Pct'].abs().sum():.1f}%",
        help="Market-neutral portfolio aims for ~0% Net Exposure and 100% × s* Gross Exposure."
    )

    # Filter Controls for the Table
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        signal_filter = st.multiselect(
            "Filter by Action Signal",
            options=scanner_df["Action_Signal"].unique().tolist(),
            default=scanner_df["Action_Signal"].unique().tolist(),
            help="Filter the S&P 500 table by specific trading instructions (Buy, Hold Long, Sell Short, Exit, Cover, etc.)."
        )
    with f_col2:
        regime_only = st.selectbox(
            "Filter by Drift Regime Status",
            options=["All S&P 500 Stocks", "Only Stocks Passing Regime Filter (Active)", "Only Stocks Filtered Out (Inactive)"],
            index=0,
            help="Show all constituents or isolate only those currently inside the 63-day >60% upward drift regime."
        )
    with f_col3:
        ticker_search = st.text_input(
            "Search Ticker or Company Name",
            value="",
            placeholder="e.g., AAPL, NVDA, Microsoft...",
            help="Type a ticker symbol or company name to filter the table immediately."
        )

    filtered_df = scanner_df[scanner_df["Action_Signal"].isin(signal_filter)].copy()
    if "Passing" in regime_only:
        filtered_df = filtered_df[filtered_df["Regime_Active"] == True]
    elif "Filtered Out" in regime_only:
        filtered_df = filtered_df[filtered_df["Regime_Active"] == False]

    if ticker_search.strip():
        q = ticker_search.strip().upper()
        filtered_df = filtered_df[
            filtered_df["Ticker"].str.upper().str.contains(q) |
            filtered_df["Company"].str.upper().str.contains(q)
        ]

    # Display Interactive Dataframe with Rich Hover Descriptions on Every Column
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=560,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                help="Overall signal rank across the S&P 500 today (#1 = highest conviction Long among active drift-regime stocks)."
            ),
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="S&P 500 stock exchange symbol."
            ),
            "Company": st.column_config.TextColumn(
                "Company",
                help="Company name."
            ),
            "Sector": st.column_config.TextColumn(
                "Sector",
                help="GICS sector classification."
            ),
            "Action_Signal": st.column_config.TextColumn(
                "Signal Action",
                help="Today's trading instruction: BUY (enter new long), HOLD LONG (maintain existing long), SELL SHORT (enter new short), HOLD SHORT (maintain existing short), EXIT LONG (sell long to close), COVER SHORT (buy back short to close), or FLAT (ignored because Regime = 0)."
            ),
            "Open_Trade_Status": st.column_config.TextColumn(
                "Trade Status",
                help="Indicates whether there is currently an active open position (Open Long / Open Short), a position closed today, or no trade."
            ),
            "Current_Price": st.column_config.NumberColumn(
                "Latest Price ($)",
                format="$%.2f",
                help="Most recent daily closing share price."
            ),
            "Up_Fraction_Pct": st.column_config.ProgressColumn(
                f"Up Days % ({regime_lookback}d)",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
                help=f"Equation (2): Percentage of days over the last {regime_lookback} trading days (~3 months) that the stock closed higher than the previous day. Must exceed {regime_thresh_pct:.0f}% to activate the strategy."
            ),
            "Regime_Active": st.column_config.CheckboxColumn(
                "Drift Regime?",
                help=f"Equation (3): Checked (True) if Up Days % > {regime_thresh_pct:.0f}%. If unchecked (False), the stock's EDGE score is multiplied by 0 and it is not traded."
            ),
            "Value_Percentile": st.column_config.NumberColumn(
                "Value Rank (0-100)",
                format="%.1f",
                help="Cross-sectional percentile rank of Inverse Share Price (1 / Price). Closer to 100 means a lower nominal share price ('cheaper'); closer to 0 means a very high nominal share price."
            ),
            "Ret_10d_Pct": st.column_config.NumberColumn(
                f"{rev_lookback}d Return (%)",
                format="%+.2f%%",
                help=f"Trailing {rev_lookback}-trading-day return (~2 weeks). The strategy looks for recent dip-losers (negative return) to buy and recent surging winners (positive return) to short."
            ),
            "Reversal_Z": st.column_config.NumberColumn(
                "Reversal Z-Score",
                format="%+.2f",
                help=f"Standardized cross-sectional Z-score of the negated {rev_lookback}-day return. Positive values indicate recent short-term losers expected to bounce back."
            ),
            "BASE_Score": st.column_config.NumberColumn(
                "BASE Score",
                format="%+.3f",
                help=f"Equation (1): Combined score = ({val_weight:.2f} × Value Rank) + ({1-val_weight:.2f} × Reversal Z-Score) before applying the Regime filter."
            ),
            "EDGE_Score": st.column_config.NumberColumn(
                "EDGE Raw",
                format="%+.3f",
                help="Equation (4): BASE Score × Regime Filter (1 or 0). Equals 0 if the stock is not in the upward drift regime."
            ),
            "EDGE_Z_Score": st.column_config.NumberColumn(
                "EDGE Z-Score",
                format="%+.2f",
                help="Standardized Z-score computed strictly across stocks currently passing the Drift Regime filter. Positive Z (> 0) goes into the Long bucket; Negative Z (< 0) goes into the Short bucket."
            ),
            "Target_Weight_Pct": st.column_config.NumberColumn(
                "Target Weight (%)",
                format="%+.2f%%",
                help="Recommended portfolio allocation percentage for today (Longs sum to +50% × s*, Shorts sum to -50% × s*)."
            ),
            "Latest_Entry_Date": st.column_config.TextColumn(
                "Latest Entry Date",
                help="Date when the current open trade (or most recent historical trade) was initiated."
            ),
            "Latest_Entry_Price": st.column_config.NumberColumn(
                "Latest Entry Price ($)",
                format="$%.2f",
                help="Closing stock price on the Latest Entry Date when the signal first triggered."
            ),
            "Days_Held": st.column_config.NumberColumn(
                "Days in Trade",
                help="Number of consecutive trading days the current open Long or Short trade has been active."
            ),
            "Open_Trade_PnL_Pct": st.column_config.NumberColumn(
                "Open Trade PnL (%)",
                format="%+.2f%%",
                help="Unrealized percentage gain/loss on the currently open trade since Latest Entry Date (accounts for direction: Long profits when price rises; Short profits when price falls)."
            ),
            "Total_Historical_Trades": st.column_config.NumberColumn(
                "Total Trades",
                help="Total number of discrete Long or Short trades triggered for this stock over the entire backtest period."
            ),
        }
    )

    # Download CSV Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered S&P 500 Signal Table (CSV)",
        data=csv_bytes,
        file_name=f"unicorn_edge_sp500_signals_{latest_market_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    st.divider()
    # Single-Stock Deep-Dive Inspector
    st.subheader("🔍 Individual Stock Signal & Regime Inspector")
    selected_stock = st.selectbox(
        "Select any S&P 500 Stock to inspect its historical Price, Drift Regime Window, and Buy/Short Signals:",
        options=scanner_df["Ticker"].tolist(),
        index=0,
        help="Plots the stock's price, 63-day Up-Day Fraction vs the 60% threshold, and active Long/Short portfolio weights over time."
    )

    if selected_stock:
        valid_idx = engine_results["scaled_weights"].index
        s_price = engine_results["prices"].loc[valid_idx, selected_stock]
        s_upfrac = engine_results["up_fraction"].loc[valid_idx, selected_stock] * 100.0
        s_weight = engine_results["scaled_weights"].loc[valid_idx, selected_stock] * 100.0

        fig_stock = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.45, 0.30, 0.25],
            subplot_titles=(
                f"{selected_stock} — Share Price & Active Regime Shading",
                f"{selected_stock} — Trailing {regime_lookback}-Day Positive Return Fraction (%) vs. {regime_thresh_pct:.0f}% Gate",
                f"{selected_stock} — Portfolio Target Weight (%) [+Long / -Short]"
            )
        )

        fig_stock.add_trace(
            go.Scatter(x=s_price.index, y=s_price.values, name="Share Price ($)", line=dict(color="#00CC96", width=2)),
            row=1, col=1
        )
        fig_stock.add_trace(
            go.Scatter(x=s_upfrac.index, y=s_upfrac.values, name="Up-Day %", line=dict(color="#AB63FA", width=1.8)),
            row=2, col=1
        )
        fig_stock.add_hline(
            y=regime_thresh_pct, line_dash="dash", line_color="yellow",
            annotation_text=f"Drift Regime Threshold ({regime_thresh_pct:.0f}%)",
            row=2, col=1
        )
        fig_stock.add_trace(
            go.Bar(
                x=s_weight.index,
                y=s_weight.values,
                name="Target Weight (%)",
                marker_color=np.where(s_weight >= 0, "#00CC96", "#EF553B")
            ),
            row=3, col=1
        )

        fig_stock.update_layout(
            height=620,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=30)
        )
        st.plotly_chart(fig_stock, use_container_width=True)

# ==============================================================================
# TAB 3: STRATEGY EQUATIONS & COLUMN GLOSSARY
# ==============================================================================
with tab_guide:
    st.subheader("📘 Quick-Reference Guide: Mathematical Formulas & Signal Rules")
    st.markdown(
        r"""
        ### 1. How the Daily Signals Are Computed
        Every day $t$, for each S&P 500 stock $i$:

        1. **Value Rank ($\text{value}_{i,t}$):** Calculate inverse share price $\left(\frac{1}{P_{i,t}}\right)$ and convert into a cross-sectional percentile score between $0$ and $1$. Stocks with lower nominal share prices receive scores closer to $1$.
        2. **Short-Term Reversal ($\text{reversal}_{i,t}$):** Calculate the stock's trailing 10-day return, negate it $\left(-R_{i,t-10 \to t}\right)$, and standardize across all stocks to a Z-score (mean $0$, standard deviation $1$). Recent 2-week losers get positive scores; recent 2-week winners get negative scores.
        3. **BASE Factor — Equation (1):**
           $$\text{BASE}_{i,t} = 0.7 \times \text{value}_{i,t} + 0.3 \times \text{reversal}_{i,t}$$
        4. **63-Day Drift Regime Filter — Equations (2) & (3):**
           $$\text{UpFraction}_{i,t} = \frac{1}{63}\sum_{k=1}^{63} \mathbb{I}\left[r_{i,t-k} > 0\right], \qquad \text{REGIME}_{i,t} = \mathbb{I}\left[\text{UpFraction}_{i,t} > 0.60\right]$$
           A stock only qualifies if more than **60%** of its last 63 trading days closed positive.
        5. **The Unicorn Edge Score — Equation (4):**
           $$\text{EDGE}_{i,t} = \text{BASE}_{i,t} \times \text{REGIME}_{i,t}$$

        ---
        ### 2. How Signals Translate into Portfolio Actions (Tab 2)
        - **🟢 BUY (New Long) / 🔵 HOLD LONG:** Stock is inside the Drift Regime ($\text{REGIME}_{i,t} = 1$) AND its standardized $\text{EDGE}$ Z-score is **greater than 0** (cheaper price + recent 10-day pullover within a 3-month uptrend).
        - **🔴 SELL SHORT (New Short) / 🟠 HOLD SHORT:** Stock is inside the Drift Regime ($\text{REGIME}_{i,t} = 1$) AND its standardized $\text{EDGE}$ Z-score is **less than 0** (higher nominal price + recent 10-day surge).
        - **⚪ EXIT LONG / 🟣 COVER SHORT:** The stock was held yesterday as a Long or Short, but today either its 63-day $\text{UpFraction}$ dropped to $\le 60\%$ ($\text{REGIME}_{i,t} = 0$) or its Z-score crossed zero.
        """
    )
