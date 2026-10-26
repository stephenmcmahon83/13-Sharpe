import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import urllib.request
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Unicorn Edge Strategy — 20-Year Backtester & S&P 500 Scanner",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦄 The 'Unicorn Edge' Quantitative Strategy Suite (20-Year Edition)")
st.markdown(
    """
    **Market-Neutral Regime-Conditioned Value & Short-Term Reversal System**  
    *Hover over any parameter `(?)` icon in the sidebar or **any column header** in the tables below for a plain-English explanation of what each metric means.*
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
    """Fetches current S&P 500 constituents from Wikipedia using a browser User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
        tables = pd.read_html(html)
        df = tables[0]
        df["Symbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
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
def download_price_data_20y(tickers, start_date, end_date):
    """
    Downloads 20-year daily price history in safe chunks.
    Returns BOTH:
      - adj_prices: Adjusted Close for accurate daily total return calculations
      - raw_prices: Unadjusted Close for accurate historical (1 / Nominal Price) Value ranking
    """
    chunk_size = 125
    adj_list = []
    raw_list = []

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]
        raw_dl = yf.download(
            batch,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            threads=True
        )
        if raw_dl.empty:
            continue

        if isinstance(raw_dl.columns, pd.MultiIndex):
            adj_p = raw_dl["Adj Close"] if "Adj Close" in raw_dl.columns.get_level_values(0) else raw_dl["Close"]
            nom_p = raw_dl["Close"]
        else:
            t_name = batch[0]
            adj_p = raw_dl[["Adj Close"]].rename(columns={"Adj Close": t_name}) if "Adj Close" in raw_dl.columns else raw_dl[["Close"]].rename(columns={"Close":
