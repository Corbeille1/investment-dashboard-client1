import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import json
from fpdf import FPDF
from datetime import datetime

# 🔧 THIS LINE MUST COME RIGHT AFTER IMPORTS
st.set_page_config(page_title="Investment Dashboard", layout="wide")

# --- Initialize session keys ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

# --- LANGUAGE SWITCH ---
lang = st.sidebar.selectbox("Language / 언어 / Langue", ["English", "Français", "한국어"])
texts = {
    "English": {
        "login": "Login to Your Investment Dashboard",
        "email": "Email",
        "password": "Password",
        "warning": "Please enter your credentials to log in.",
        "success": "Logged in successfully!",
        "title": "Investment Portfolio Tracker",
        "tickers": "Enter tickers (comma separated)",
        "shares": "Enter number of shares (same order)",
        "buy_prices": "Enter buy prices (same order)",
        "track": "Track Portfolio",
        "summary": "Portfolio Summary",
        "allocation": "Asset Allocation",
        "compare": "Portfolio vs. S&P 500",
        "metrics": "Performance Metrics"
    }
}
t = texts[lang]

# --- LOGIN SYSTEM ---
EMAIL = st.secrets.get("EMAIL", "amahali.we@gmail.com")
PASSWORD = st.secrets.get("PASSWORD", "changeme")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title(f"🔒 {t['login']}")
    with st.form("login_form", clear_on_submit=False):
        email_input = st.text_input(t['email'])
        password_input = st.text_input(t['password'], type="password")
        submit_login = st.form_submit_button("Access Dashboard")

    if submit_login:
        if email_input.strip().lower() == EMAIL.lower() and password_input.strip() == PASSWORD:
            st.session_state.logged_in = True
            st.session_state.show_dashboard = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")
    st.stop()

# LOGGED IN: Add Logout in Sidebar
with st.sidebar:
    st.markdown("### 👤 Account")
    if st.button("🚪 Log out"):
        st.session_state.logged_in = False
        st.session_state.show_dashboard = False
        st.rerun()

# Show dashboard if logged in
if st.session_state.show_dashboard:
    st.title(f"📊 {t['title']}")

    portfolio = []

    # Upload a file to pre-fill input
    st.subheader("📥 Load a portfolio (to auto-fill)")
    uploaded_file = st.file_uploader("Upload your portfolio file (JSON or CSV)", type=["json", "csv"], key="upload_prefill")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".json"):
                portfolio_loaded = json.load(uploaded_file)
            else:
                df_uploaded = pd.read_csv(uploaded_file)
                portfolio_loaded = df_uploaded.to_dict(orient="records")

            # Auto-fill values
            tickers_default = ', '.join([item['ticker'] for item in portfolio_loaded])
            shares_default = ', '.join([str(item['shares']) for item in portfolio_loaded])
            buy_prices_default = ', '.join([str(item['buy_price']) for item in portfolio_loaded])
            st.success("✅ Portfolio loaded and fields pre-filled!")
        except Exception as e:
            st.error("❌ Failed to load portfolio: {}".format(e))
            tickers_default = shares_default = buy_prices_default = ""
    else:
        tickers_default = "AAPL, TSLA, VOO"
        shares_default = "10, 5, 7"
        buy_prices_default = "145, 700, 380"

    # Manual fields (populated with above if available)
    st.subheader(t['tickers'])
    tickers = st.text_input("", tickers_default)
    st.subheader(t['shares'])
    shares = st.text_input("", shares_default)
    st.subheader(t['buy_prices'])
    buy_prices = st.text_input("", buy_prices_default)

    if st.button(t['track']):
        tickers = [x.strip().upper() for x in tickers.split(",")]
        shares = [int(x.strip()) for x in shares.split(",")]
        buy_prices = [float(x.strip()) for x in buy_prices.split(",")]

        if not (len(tickers) == len(shares) == len(buy_prices)):
            st.error("⚠️ The number of tickers, shares, and buy prices must match.")
            st.stop()

        for i in range(len(tickers)):
            portfolio.append({
                'ticker': tickers[i],
                'shares': shares[i],
                'buy_price': buy_prices[i]
            })

        # Add collapsible block to merge more portfolios
        with st.expander("📁 Add another portfolio (merge)", expanded=False):
            uploaded_file_post = st.file_uploader("📂 Upload to merge with current portfolio", type=["json", "csv"], key="upload_merge")
            if uploaded_file_post:
                try:
                    if uploaded_file_post.name.endswith(".json"):
                        additional_portfolio = json.load(uploaded_file_post)
                    else:
                        df_extra = pd.read_csv(uploaded_file_post)
                        additional_portfolio = df_extra.to_dict(orient="records")

                    if isinstance(additional_portfolio, list) and all('ticker' in x and 'shares' in x and 'buy_price' in x for x in additional_portfolio):
                        for item in additional_portfolio:
                            item['ticker'] = item['ticker'].strip().upper()
                        portfolio.extend(additional_portfolio)
                        st.success("✅ Portfolio successfully merged!")
                    else:
                        st.error("⚠️ Invalid format.")
                except Exception as e:
                    st.error("❌ Failed to merge portfolio: {}".format(e))

        # --- Price Fetching ---
        price_data = yf.download([item['ticker'] for item in portfolio], period="5d")

        # Handle single vs. multi-ticker
        if isinstance(price_data.columns, pd.MultiIndex):
            if 'Adj Close' in price_data.columns:
                data = price_data['Adj Close'].dropna().iloc[-1]
            elif 'Close' in price_data.columns:
                data = price_data['Close'].dropna().iloc[-1]
            else:
                st.error("❌ Neither 'Adj Close' nor 'Close' was found in the price data.")
                st.stop()
        else:
            # Single-ticker fallback
            if 'Adj Close' in price_data.columns:
                ticker = [item['ticker'] for item in portfolio][0]
                data = pd.Series({ticker: price_data['Adj Close'].dropna().iloc[-1]})
            elif 'Close' in price_data.columns:
                ticker = [item['ticker'] for item in portfolio][0]
                data = pd.Series({ticker: price_data['Close'].dropna().iloc[-1]})
            else:
                st.error("❌ Price column missing. Please check the ticker.")
                st.stop()

        results = []
        total_value = 0
        total_cost = 0

        for item in portfolio:
            ticker = item['ticker']
            shares = item['shares']
            buy_price = item['buy_price']
            try:
                current_price = data[ticker]
            except KeyError:
                st.error(f"❌ Price for {ticker} not found.")
                st.stop()

            value = shares * current_price
            cost = shares * buy_price
            pnl = value - cost
            return_pct = (pnl / cost) * 100

            total_value += value
            total_cost += cost

            results.append({
                'Ticker': ticker,
                'Shares': shares,
                'Buy Price': buy_price,
                'Current Price': round(current_price, 2),
                'Current Value': round(value, 2),
                'P&L': round(pnl, 2),
                'Return %': round(return_pct, 2)
            })

        df = pd.DataFrame(results)
        st.subheader(t['summary'])
        st.dataframe(df)

        st.markdown(f"**Total Cost:** ${round(total_cost, 2)}")
        st.markdown(f"**Total Value:** ${round(total_value, 2)}")
        st.markdown(f"**Total P&L:** ${round(total_value - total_cost, 2)}")
